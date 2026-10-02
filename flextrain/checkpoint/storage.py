"""Storage backends for checkpoints.

A backend stores opaque checkpoint blobs under string keys and can list them, which
lets retention and resume work from what is *actually persisted* rather than from
in-memory bookkeeping that is lost when a job restarts.
"""

from __future__ import annotations

import io
import logging
import os
import posixpath
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class StorageBackend(ABC):
    """Abstract checkpoint store."""

    @abstractmethod
    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        """Persist ``state`` at ``path`` atomically. Returns the number of bytes written."""

    @abstractmethod
    def load(self, path: PathLike) -> Dict[str, Any]:
        ...

    @abstractmethod
    def exists(self, path: PathLike) -> bool:
        ...

    @abstractmethod
    def delete(self, path: PathLike) -> None:
        ...

    @abstractmethod
    def list(self, directory: PathLike) -> List[str]:
        """Return the full paths of the entries directly inside ``directory``."""

    def size(self, path: PathLike) -> Optional[int]:
        return None

    def join(self, directory: PathLike, name: str) -> str:
        return posixpath.join(str(directory), name)


def _fsync_dir(directory: Path) -> None:
    """Persist a rename by fsyncing the parent directory (POSIX only)."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LocalStorage(StorageBackend):
    """Local / shared-filesystem backend with crash-safe writes.

    Writes go to a unique temp file in the target directory, are ``fsync``-ed, then
    ``os.replace``-d into place and the directory is ``fsync``-ed. A reader therefore
    sees either the previous complete file or the new complete file - never a torn one,
    even if the node loses power mid-write.
    """

    def __init__(self, fsync: bool = True):
        self.fsync = fsync

    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp_path, "wb") as f:
                torch.save(state, f)
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())
                nbytes = f.tell()
            os.replace(tmp_path, path)
            if self.fsync:
                _fsync_dir(path.parent)
            return nbytes
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def load(self, path: PathLike) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        # weights_only=False: checkpoints also carry Python RNG state and metadata.
        # Only load checkpoints you (or your job) wrote.
        return torch.load(path, map_location="cpu", weights_only=False)

    def exists(self, path: PathLike) -> bool:
        return Path(path).exists()

    def delete(self, path: PathLike) -> None:
        Path(path).unlink(missing_ok=True)

    def list(self, directory: PathLike) -> List[str]:
        directory = Path(directory)
        if not directory.is_dir():
            return []
        return [str(p) for p in directory.iterdir() if p.is_file()]

    def size(self, path: PathLike) -> Optional[int]:
        try:
            return Path(path).stat().st_size
        except FileNotFoundError:
            return None

    def join(self, directory: PathLike, name: str) -> str:
        return str(Path(directory) / name)


def _serialize(state: Dict[str, Any]):
    """Serialize to an anonymous temp file (returns it rewound).

    Writing to a real file lets ``torch.save`` release the GIL during I/O, so a background
    upload does not stall the training thread; ``io.BytesIO`` writes hold the GIL.
    """
    spool = tempfile.TemporaryFile()
    try:
        torch.save(state, spool)
        spool.flush()
        spool.seek(0)
    except BaseException:
        spool.close()
        raise
    return spool


def _key(path: PathLike) -> str:
    return str(path).replace("\\", "/").lstrip("/")


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend. A single-object upload is atomic in GCS."""

    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None, client: Any = None):
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as e:
                raise ImportError("google-cloud-storage required: pip install 'flextrain[gcs]'") from e
            client = (storage.Client.from_service_account_json(credentials_path)
                      if credentials_path else storage.Client())
        self.client = client
        self.bucket = client.bucket(bucket_name)

    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        with _serialize(state) as spool:
            nbytes = os.fstat(spool.fileno()).st_size
            self.bucket.blob(_key(path)).upload_from_file(spool, size=nbytes)
        return nbytes

    def load(self, path: PathLike) -> Dict[str, Any]:
        buffer = io.BytesIO()
        self.bucket.blob(_key(path)).download_to_file(buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def exists(self, path: PathLike) -> bool:
        return self.bucket.blob(_key(path)).exists()

    def delete(self, path: PathLike) -> None:
        self.bucket.blob(_key(path)).delete()

    def list(self, directory: PathLike) -> List[str]:
        prefix = _key(directory).rstrip("/") + "/"
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix, delimiter="/")]


class S3Storage(StorageBackend):
    """Amazon S3 backend. A single PUT / multipart upload is atomic in S3."""

    def __init__(self, bucket_name: str, region: Optional[str] = None, client: Any = None):
        if client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError("boto3 required: pip install 'flextrain[s3]'") from e
            client = boto3.client("s3", region_name=region)
        self.bucket_name = bucket_name
        self.s3 = client

    def save(self, state: Dict[str, Any], path: PathLike) -> int:
        with _serialize(state) as spool:
            nbytes = os.fstat(spool.fileno()).st_size
            self.s3.upload_fileobj(spool, self.bucket_name, _key(path))
        return nbytes

    def load(self, path: PathLike) -> Dict[str, Any]:
        buffer = io.BytesIO()
        self.s3.download_fileobj(self.bucket_name, _key(path), buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)

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
