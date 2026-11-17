"""Storage backends for checkpoints.

A backend stores opaque checkpoint blobs under string keys and can list them, which
lets retention and resume work from what is *actually persisted* rather than from
in-memory bookkeeping that is lost when a job restarts.

Integrity. ``torch.load`` does not notice a flipped byte inside tensor data: the archive
still parses and the weights are silently wrong. Every backend therefore writes a SHA-256
digest of the serialized bytes next to the checkpoint (``<name>.sha256``, in the format
``sha256sum -c`` / ``shasum -a 256 -c`` understand) and verifies it on load. The digest is
written only after the checkpoint itself is durable, and a digest left over from an earlier
checkpoint of the same name is removed *before* the new bytes replace the old ones. A crash in
between therefore leaves a complete checkpoint with no digest - never one paired with a digest
of different bytes. A checkpoint without a digest (or written before digests existed) loads
unverified.

Errors. ``load`` raises ``CheckpointCorruptedError`` when the stored bytes are wrong (digest
mismatch, or they do not deserialize) and lets every other error - the file system or the
network failing - propagate unchanged. Callers rely on that distinction: a corrupted checkpoint
can be skipped, an unreachable one must not be mistaken for a missing one.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import posixpath
import re
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

import torch

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

DIGEST_SUFFIX = ".sha256"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK = 4 * 1024 * 1024


class CheckpointCorruptedError(RuntimeError):
    """The stored bytes of a checkpoint are wrong: they fail their digest or cannot be deserialized."""


class UntrustedCheckpointError(RuntimeError):
    """The checkpoint is intact but holds pickled Python objects that safe loading refuses to build."""


def _deserialize(stream: BinaryIO, path: PathLike, allow_pickle: bool = False) -> Dict[str, Any]:
    """Turn checkpoint bytes into a state dict.

    Unpickling can run arbitrary code, and a checkpoint directory is writable by more parties
    than the training code is (a shared bucket, a shared file system). So by default only
    tensors and plain containers are reconstructed (``torch.load(weights_only=True)``), which is
    all a FlexTrain checkpoint contains. The digest does not change this: it sits next to the
    data, so it shows the bytes are what was written, not who wrote them.
    """
    try:
        return torch.load(stream, map_location="cpu", weights_only=not allow_pickle)
    except pickle.UnpicklingError as e:
        # torch words the refusal differently across versions ("Unsupported class X" in 2.3,
        # "Unsupported global: GLOBAL X was not an allowed global" later).
        refused = ("Unsupported global", "Unsupported class", "allowed global")
        if not allow_pickle and any(marker in str(e) for marker in refused):
            raise UntrustedCheckpointError(
                f"{path} contains pickled Python objects, which are not loaded by default because unpickling "
                f"can execute code. (FlexTrain <= 0.2 checkpoints do; so do checkpoints holding custom scheduler "
                f"or optimizer state.) If you trust where this file came from, set checkpoint.allow_pickle=true."
            ) from e
        raise CheckpointCorruptedError(f"{path} is not a readable checkpoint: {type(e).__name__}: {e}") from e
    except (OSError, MemoryError):
        raise  # the machine's problem, not the checkpoint's
    except Exception as e:
        # The bytes were read; they just are not a checkpoint (truncated archive, damaged pickle
        # stream). Unpicklers signal that with whatever exception the damage happens to trigger.
        raise CheckpointCorruptedError(f"{path} is not a readable checkpoint: {type(e).__name__}: {e}") from e


def digest_path(path: PathLike) -> str:
    """Where the digest of the checkpoint at ``path`` is stored."""
    return f"{path}{DIGEST_SUFFIX}"


def _hash_stream(stream: BinaryIO) -> str:
    """SHA-256 of ``stream`` from its current position to the end (hashlib releases the GIL)."""
    sha = hashlib.sha256()
    buffer = bytearray(_HASH_CHUNK)
    view = memoryview(buffer)
    while True:
        n = stream.readinto(view)
        if not n:
            return sha.hexdigest()
        sha.update(view[:n])


class _HashingWriter:
    """Write-only view of a file that hashes the bytes on their way through.

    ``torch.save`` writes a checkpoint front to back, so hashing what it writes gives the
    digest of the file without reading it again: one pass over the data instead of two. The
    hash is updated before ``write`` returns because torch reuses the buffer it passes in.
    There is deliberately no ``seek``: a writer that went back to patch earlier bytes would
    make the digest wrong, and must fail here instead.
    """

    def __init__(self, stream: BinaryIO):
        self._stream = stream
        self._sha = hashlib.sha256()

    def write(self, data) -> int:
        self._sha.update(data)  # releases the GIL for all but tiny buffers
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def hexdigest(self) -> str:
        return self._sha.hexdigest()


def _dump(state: Dict[str, Any], stream: BinaryIO, checksum: bool) -> Optional[str]:
    """``torch.save`` ``state`` to ``stream``; returns the SHA-256 of what was written if asked to."""
    if not checksum:
        torch.save(state, stream)
        return None
    writer = _HashingWriter(stream)
    torch.save(state, writer)
    return writer.hexdigest()


class StorageBackend(ABC):
    """Abstract checkpoint store."""

    checksum: bool = True
    allow_pickle: bool = False

    @abstractmethod
    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        """Persist ``state`` at ``path`` atomically. Returns the number of bytes written."""

    @abstractmethod
    def load(self, path: PathLike) -> Dict[str, Any]:
        """Load a checkpoint, raising ``CheckpointCorruptedError`` if its digest does not match."""

    @abstractmethod
    def exists(self, path: PathLike) -> bool:
        ...

    @abstractmethod
    def delete(self, path: PathLike) -> None:
        ...

    @abstractmethod
    def list(self, directory: PathLike) -> List[str]:
        """Return the full paths of the entries directly inside ``directory``."""

    @abstractmethod
    def write_bytes(self, path: PathLike, data: bytes) -> None:
        """Store a small blob atomically."""

    @abstractmethod
    def read_bytes(self, path: PathLike) -> Optional[bytes]:
        """Return a small blob, or ``None`` if it does not exist."""

    def join(self, directory: PathLike, name: str) -> str:
        return posixpath.join(str(directory), name)

    # ------------------------------------------------------------------ integrity
    def write_digest(self, path: PathLike, digest: str) -> None:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        self.write_bytes(digest_path(path), f"{digest}  {name}\n".encode())

    def read_digest(self, path: PathLike) -> Optional[str]:
        """The recorded SHA-256 of the checkpoint at ``path``; ``None`` if none was recorded."""
        data = self.read_bytes(digest_path(path))
        if data is None:
            return None
        fields = data.split()
        digest = fields[0].decode("ascii", "replace") if fields else ""
        if not _DIGEST_RE.match(digest):
            raise CheckpointCorruptedError(f"unreadable digest file {digest_path(path)}")
        return digest

    def _verify(self, path: PathLike, stream: BinaryIO) -> None:
        """Check ``stream`` (positioned at 0) against the recorded digest, then rewind it."""
        expected = self.read_digest(path) if self.checksum else None
        if expected is None:
            return
        actual = _hash_stream(stream)
        if actual != expected:
            raise CheckpointCorruptedError(
                f"checksum mismatch for {path}: stored bytes hash to {actual[:12]}..., "
                f"expected {expected[:12]}... (bit rot, or a truncated / partial copy)"
            )
        stream.seek(0)

    def delete_digest(self, path: PathLike) -> None:
        if self.exists(digest_path(path)):
            self.delete(digest_path(path))

    def delete_checkpoint(self, path: PathLike) -> None:
        """Delete a checkpoint and its digest (digest first, so no orphaned digest can remain)."""
        self.delete_digest(path)
        self.delete(path)


def _sync(fd: int) -> None:
    """Flush ``fd`` to stable storage.

    On macOS ``fsync`` only hands the data to the drive, which may still hold it in a volatile
    cache; ``F_FULLFSYNC`` is what asks the drive to flush. Without it "survives power loss"
    would be true on Linux only.
    """
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except OSError:  # not supported by this file system: fall back to a plain fsync
            pass
    os.fsync(fd)


def _fsync_dir(directory: Path) -> None:
    """Persist a rename by fsyncing the parent directory (POSIX only)."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        _sync(fd)
    finally:
        os.close(fd)


class LocalStorage(StorageBackend):
    """Local / shared-filesystem backend with crash-safe writes.

    Writes go to a unique temp file in the target directory, are ``fsync``-ed, then
    ``os.replace``-d into place and the directory is ``fsync``-ed. A reader therefore
    sees either the previous complete file or the new complete file - never a torn one,
    even if the node loses power mid-write.
    """

    def __init__(self, fsync: bool = True, checksum: bool = True, allow_pickle: bool = False):
        self.fsync = fsync
        self.checksum = checksum
        self.allow_pickle = allow_pickle

    def _replace_into(self, tmp_path: Path, path: Path) -> None:
        os.replace(tmp_path, path)
        if self.fsync:
            _fsync_dir(path.parent)

    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        digest = None
        try:
            with open(tmp_path, "wb") as f:
                digest = _dump(state, f, self.checksum)
                f.flush()
                nbytes = f.tell()
                if self.fsync:
                    _sync(f.fileno())
            self.delete_digest(path)  # never let an older digest describe the new bytes
            self._replace_into(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        if digest is not None:
            self.write_digest(path, digest)
        return nbytes

    def load(self, path: PathLike) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(path, "rb") as f:
            self._verify(path, f)
            return _deserialize(f, path, self.allow_pickle)

    def exists(self, path: PathLike) -> bool:
        return Path(path).exists()

    def delete(self, path: PathLike) -> None:
        Path(path).unlink(missing_ok=True)

    def list(self, directory: PathLike) -> List[str]:
        directory = Path(directory)
        if not directory.is_dir():
            return []
        return [str(p) for p in directory.iterdir() if p.is_file()]

    def write_bytes(self, path: PathLike, data: bytes) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                if self.fsync:
                    _sync(f.fileno())
            self._replace_into(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def read_bytes(self, path: PathLike) -> Optional[bytes]:
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None

    def join(self, directory: PathLike, name: str) -> str:
        return str(Path(directory) / name)


def _serialize(state: Dict[str, Any], checksum: bool = False):
    """Serialize to an anonymous temp file. Returns it rewound, and its SHA-256 if asked to.

    Writing to a real file lets ``torch.save`` release the GIL during I/O, so a background
    upload does not stall the training thread; ``io.BytesIO`` writes hold the GIL.
    """
    spool = tempfile.TemporaryFile()
    try:
        digest = _dump(state, spool, checksum)
        spool.flush()
        spool.seek(0)
    except BaseException:
        spool.close()
        raise
    return spool, digest


def _key(path: PathLike) -> str:
    return str(path).replace("\\", "/").lstrip("/")


class _ObjectStorage(StorageBackend):
    """Shared save / load logic for object stores (a single-object upload is atomic)."""

    def __init__(self, checksum: bool = True, allow_pickle: bool = False):
        self.checksum = checksum
        self.allow_pickle = allow_pickle

    @abstractmethod
    def _upload(self, spool: BinaryIO, path: PathLike, nbytes: int) -> None:
        ...

    @abstractmethod
    def _download(self, path: PathLike, buffer: BinaryIO) -> None:
        ...

    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        spool, digest = _serialize(state, self.checksum)
        with spool:
            nbytes = os.fstat(spool.fileno()).st_size
            self.delete_digest(path)  # never let an older digest describe the new bytes
            self._upload(spool, path, nbytes)
        if digest is not None:
            self.write_digest(path, digest)
        return nbytes

    def load(self, path: PathLike) -> Dict[str, Any]:
        buffer = io.BytesIO()
        self._download(path, buffer)
        buffer.seek(0)
        self._verify(path, buffer)
        return _deserialize(buffer, path, self.allow_pickle)


class GCSStorage(_ObjectStorage):
    """Google Cloud Storage backend."""

    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None, client: Any = None,
                 checksum: bool = True, allow_pickle: bool = False):
        super().__init__(checksum, allow_pickle)
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as e:
                raise ImportError("google-cloud-storage required: pip install 'flextrain[gcs]'") from e
            client = (storage.Client.from_service_account_json(credentials_path)
                      if credentials_path else storage.Client())
        self.client = client
        self.bucket = client.bucket(bucket_name)

    def _upload(self, spool: BinaryIO, path: PathLike, nbytes: int) -> None:
        self.bucket.blob(_key(path)).upload_from_file(spool, size=nbytes)

    def _download(self, path: PathLike, buffer: BinaryIO) -> None:
        self.bucket.blob(_key(path)).download_to_file(buffer)

    def exists(self, path: PathLike) -> bool:
        return self.bucket.blob(_key(path)).exists()

    def delete(self, path: PathLike) -> None:
        blob = self.bucket.blob(_key(path))
        if blob.exists():  # deleting a missing object is an error in GCS; make it idempotent
            blob.delete()

    def list(self, directory: PathLike) -> List[str]:
        prefix = _key(directory).rstrip("/") + "/"
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix, delimiter="/")]

    def write_bytes(self, path: PathLike, data: bytes) -> None:
        self.bucket.blob(_key(path)).upload_from_string(data)

    def read_bytes(self, path: PathLike) -> Optional[bytes]:
        blob = self.bucket.blob(_key(path))
        return blob.download_as_bytes() if blob.exists() else None


class S3Storage(_ObjectStorage):
    """Amazon S3 backend."""

    def __init__(self, bucket_name: str, region: Optional[str] = None, client: Any = None, checksum: bool = True,
                 allow_pickle: bool = False):
        super().__init__(checksum, allow_pickle)
        if client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError("boto3 required: pip install 'flextrain[s3]'") from e
            client = boto3.client("s3", region_name=region)
        self.bucket_name = bucket_name
        self.s3 = client

    def _upload(self, spool: BinaryIO, path: PathLike, nbytes: int) -> None:
        self.s3.upload_fileobj(spool, self.bucket_name, _key(path))

    def _download(self, path: PathLike, buffer: BinaryIO) -> None:
        self.s3.download_fileobj(self.bucket_name, _key(path), buffer)

    def exists(self, path: PathLike) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=_key(path))
            return True
        except Exception as e:  # botocore.exceptions.ClientError
            code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def delete(self, path: PathLike) -> None:
        self.s3.delete_object(Bucket=self.bucket_name, Key=_key(path))

    def list(self, directory: PathLike) -> List[str]:
        prefix = _key(directory).rstrip("/") + "/"
        keys: List[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def write_bytes(self, path: PathLike, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket_name, Key=_key(path), Body=data)

    def read_bytes(self, path: PathLike) -> Optional[bytes]:
        if not self.exists(path):
            return None
        return self.s3.get_object(Bucket=self.bucket_name, Key=_key(path))["Body"].read()


def create_storage_backend(backend_type: str, bucket_name: Optional[str] = None, **kwargs) -> StorageBackend:
    """Create a storage backend by name."""
    backend_type = backend_type.lower()
    if backend_type == "local":
        return LocalStorage(**kwargs)
    if backend_type in ("gcs", "s3"):
        if not bucket_name:
            raise ValueError(f"bucket_name is required for the {backend_type} backend")
        cls = GCSStorage if backend_type == "gcs" else S3Storage
        return cls(bucket_name, **kwargs)
    raise ValueError(f"Unknown storage backend: {backend_type}")
