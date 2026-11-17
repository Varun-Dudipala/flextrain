"""Storage backend tests (local for real, GCS/S3 against in-memory fakes)."""

import hashlib
import io
from pathlib import Path

import pytest
import torch

from flextrain.checkpoint.storage import (
    CheckpointCorruptedError,
    GCSStorage,
    LocalStorage,
    S3Storage,
    UntrustedCheckpointError,
    create_storage_backend,
    digest_path,
)


class Payload:
    """Stands in for anything that needs real unpickling (and so could run code on load)."""

    def __init__(self):
        self.value = 7


def flip_byte_in_tensor_data(raw: bytes, tensor: torch.Tensor) -> bytes:
    """Corrupt one byte of ``tensor``'s payload inside a serialized checkpoint (simulated bit rot)."""
    offset = raw.find(tensor.numpy().tobytes())
    assert offset > 0, "tensor payload not found in the checkpoint bytes"
    corrupted = bytearray(raw)
    corrupted[offset + 5] ^= 0xFF
    return bytes(corrupted)


class TestLocalStorage:
    def test_round_trip(self, tmp_path):
        storage = LocalStorage()
        state = {"key": "value", "tensor": torch.randn(10)}
        path = tmp_path / "sub" / "ckpt.pt"
        nbytes = storage.save(state, path)
        assert nbytes == path.stat().st_size > 0
        loaded = storage.load(path)
        assert loaded["key"] == "value" and torch.equal(loaded["tensor"], state["tensor"])

    def test_exists_delete_list(self, tmp_path):
        storage = LocalStorage()
        assert not storage.exists(tmp_path / "a.pt")
        storage.save({"a": 1}, tmp_path / "a.pt")
        storage.save({"b": 2}, tmp_path / "b.pt")
        assert storage.exists(tmp_path / "a.pt")
        assert sorted(Path(p).name for p in storage.list(tmp_path)) == ["a.pt", "a.pt.sha256", "b.pt", "b.pt.sha256"]
        storage.delete_checkpoint(tmp_path / "a.pt")
        storage.delete_checkpoint(tmp_path / "a.pt")  # idempotent
        assert sorted(Path(p).name for p in storage.list(tmp_path)) == ["b.pt", "b.pt.sha256"]
        assert storage.list(tmp_path / "missing") == []

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalStorage().load(tmp_path / "missing.pt")

    def test_overwrite_is_atomic_replace(self, tmp_path):
        storage = LocalStorage()
        path = tmp_path / "c.pt"
        storage.save({"v": 1}, path)
        storage.save({"v": 2}, path)
        assert storage.load(path)["v"] == 2

    def test_failed_write_leaves_no_partial_file_and_keeps_old(self, tmp_path, monkeypatch):
        storage = LocalStorage()
        path = tmp_path / "c.pt"
        storage.save({"v": 1}, path)

        def boom(obj, f, *a, **k):
            f.write(b"partial garbage")
            raise OSError("disk full")

        monkeypatch.setattr(torch, "save", boom)
        with pytest.raises(OSError):
            storage.save({"v": 2}, path)
        monkeypatch.undo()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["c.pt", "c.pt.sha256"]  # no temp file left behind
        assert storage.load(path)["v"] == 1  # previous checkpoint intact, and it still matches its digest

    def test_fsync_is_called(self, tmp_path, monkeypatch):
        from flextrain.checkpoint import storage as storage_module

        calls = []
        real_sync = storage_module._sync
        monkeypatch.setattr(storage_module, "_sync", lambda fd: (calls.append(fd), real_sync(fd)))
        LocalStorage(fsync=True).save({"a": 1}, tmp_path / "a.pt")
        assert len(calls) >= 4  # the checkpoint, its directory, the digest, its directory
        calls.clear()
        LocalStorage(fsync=False).save({"a": 1}, tmp_path / "b.pt")
        assert calls == []


class TestIntegrity:
    """torch.load alone accepts a checkpoint with a flipped byte in its weights; the digest does not."""

    def test_torch_load_alone_misses_flipped_bytes(self, tmp_path):
        weights = torch.arange(4096, dtype=torch.float32)
        path = tmp_path / "c.pt"
        LocalStorage(checksum=False).save({"w": weights}, path)
        path.write_bytes(flip_byte_in_tensor_data(path.read_bytes(), weights))
        loaded = LocalStorage(checksum=False).load(path)  # no error: this is the failure mode
        assert not torch.equal(loaded["w"], weights)

    def test_flipped_byte_is_detected(self, tmp_path):
        weights = torch.arange(4096, dtype=torch.float32)
        path = tmp_path / "c.pt"
        storage = LocalStorage()
        storage.save({"w": weights}, path)
        path.write_bytes(flip_byte_in_tensor_data(path.read_bytes(), weights))
        with pytest.raises(CheckpointCorruptedError, match="checksum mismatch"):
            storage.load(path)

    def test_digest_file_is_sha256sum_compatible(self, tmp_path):
        path = tmp_path / "checkpoint_step00000001.pt"
        LocalStorage().save({"w": torch.ones(8)}, path)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert Path(digest_path(path)).read_text() == f"{expected}  checkpoint_step00000001.pt\n"

    def test_digest_is_computed_while_writing_not_by_reading_the_file_back(self, tmp_path, monkeypatch):
        """One pass over the data instead of two. The state has many tensors of very different
        sizes, so the digest covers every kind of write torch.save makes (headers of a few bytes,
        records of megabytes), and it must still be the SHA-256 of the file as it is on disk."""
        import flextrain.checkpoint.storage as storage_module

        def no_reading_back(stream):
            raise AssertionError("the checkpoint was read back to hash it")

        state = {"model": {f"w{i}": torch.randn(n) for i, n in enumerate([1, 7, 1000, 300_000, 2_000_000])},
                 "trainer": {"global_step": 5, "note": "x" * 5000}}
        path = tmp_path / "c.pt"
        with monkeypatch.context() as m:
            m.setattr(storage_module, "_hash_stream", no_reading_back)
            nbytes = LocalStorage().save(state, path)
        assert nbytes == path.stat().st_size
        assert Path(digest_path(path)).read_text().split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = LocalStorage().load(path)  # verifies the digest against the stored bytes
        assert all(torch.equal(loaded["model"][k], v) for k, v in state["model"].items())

    def test_the_hashing_writer_cannot_be_rewound(self):
        """A serializer that went back to patch earlier bytes would leave a digest that does not
        describe the file. It has to fail instead."""
        from flextrain.checkpoint.storage import _HashingWriter

        writer = _HashingWriter(io.BytesIO())
        writer.write(b"abc")
        assert writer.hexdigest() == hashlib.sha256(b"abc").hexdigest()
        assert not hasattr(writer, "seek") and not hasattr(writer, "tell")

    def test_object_store_digest_describes_the_uploaded_bytes(self):
        client = FakeS3Client()
        storage = S3Storage("b", client=client)
        storage.save({"w": torch.randn(50_000), "small": torch.ones(3)}, "ckpt/k.pt")
        uploaded = client.store["ckpt/k.pt"]
        assert storage.read_digest("ckpt/k.pt") == hashlib.sha256(uploaded).hexdigest()
        assert torch.equal(storage.load("ckpt/k.pt")["small"], torch.ones(3))

    def test_checkpoint_without_digest_loads_unverified(self, tmp_path):
        """Checkpoints written before digests existed, or whose writer died before the digest landed."""
        path = tmp_path / "c.pt"
        storage = LocalStorage()
        storage.save({"v": 1}, path)
        Path(digest_path(path)).unlink()
        assert storage.read_digest(path) is None and storage.load(path)["v"] == 1

    def test_garbled_digest_file_counts_as_corruption(self, tmp_path):
        path = tmp_path / "c.pt"
        storage = LocalStorage()
        storage.save({"v": 1}, path)
        Path(digest_path(path)).write_text("not a digest")
        with pytest.raises(CheckpointCorruptedError, match="unreadable digest"):
            storage.load(path)

    def test_overwrite_refreshes_the_digest(self, tmp_path):
        path = tmp_path / "c.pt"
        storage = LocalStorage()
        storage.save({"v": torch.zeros(64)}, path)
        storage.save({"v": torch.ones(64)}, path)
        assert torch.equal(storage.load(path)["v"], torch.ones(64))

    def test_overwriting_never_pairs_new_bytes_with_the_old_digest(self, tmp_path, monkeypatch):
        """Crash between replacing the data and writing its digest: the result must be a good
        checkpoint without a digest, not a good checkpoint that fails an older one's digest."""
        path = tmp_path / "c.pt"
        storage = LocalStorage()
        storage.save({"v": torch.zeros(64)}, path)

        class Crash(BaseException):
            pass

        def die(self, path, digest):
            raise Crash

        monkeypatch.setattr(LocalStorage, "write_digest", die)
        with pytest.raises(Crash):
            storage.save({"v": torch.ones(64)}, path)
        monkeypatch.undo()
        assert storage.read_digest(path) is None
        assert torch.equal(storage.load(path)["v"], torch.ones(64))

    def test_bytes_that_do_not_deserialize_are_corruption_not_a_crash(self, tmp_path):
        path = tmp_path / "c.pt"
        path.write_bytes(b"this is not a checkpoint")
        with pytest.raises(CheckpointCorruptedError, match="not a readable checkpoint"):
            LocalStorage().load(path)

    def test_io_errors_are_not_reported_as_corruption(self, tmp_path, monkeypatch):
        """An unreachable checkpoint must stay distinguishable from a damaged one."""
        path = tmp_path / "c.pt"
        LocalStorage().save({"v": 1}, path)

        def stale_handle(*args, **kwargs):
            raise OSError("stale NFS file handle")

        monkeypatch.setattr(torch, "load", stale_handle)
        with pytest.raises(OSError, match="stale NFS"):
            LocalStorage().load(path)

    def test_checksum_can_be_disabled(self, tmp_path):
        LocalStorage(checksum=False).save({"v": 1}, tmp_path / "c.pt")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["c.pt"]


class TestSafeLoading:
    """Unpickling can execute code, and checkpoint locations are often writable by other parties."""

    def test_arbitrary_objects_are_not_unpickled_by_default(self, tmp_path):
        path = tmp_path / "c.pt"
        LocalStorage().save({"step": 1, "payload": Payload()}, path)
        with pytest.raises(UntrustedCheckpointError, match="allow_pickle"):
            LocalStorage().load(path)
        assert LocalStorage(allow_pickle=True).load(path)["payload"].value == 7

    def test_refusing_to_unpickle_is_not_reported_as_corruption(self, tmp_path):
        """A corrupted checkpoint is skipped in favour of an older one; an untrusted one must not be."""
        path = tmp_path / "c.pt"
        LocalStorage().save({"payload": Payload()}, path)
        try:
            LocalStorage().load(path)
        except CheckpointCorruptedError:
            pytest.fail("intact but untrusted checkpoint was classified as corrupted")
        except UntrustedCheckpointError:
            pass

    def test_everything_flextrain_itself_writes_loads_safely(self, tmp_path):
        """Tensors, optimizer and scheduler state, and all three RNG states (NumPy's as plain lists)."""
        import random

        import numpy as np

        from flextrain.checkpoint import capture_rng_state, capture_state, restore_rng_state

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.randn(3, 4)).sum().backward()
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        state = capture_state(model, optimizer, scheduler)
        state["rng"] = [capture_rng_state()]
        path = tmp_path / "c.pt"
        LocalStorage().save(state, path)
        expected = (random.random(), float(np.random.rand()), torch.rand(1).item())

        loaded = LocalStorage().load(path)  # safe mode
        restore_rng_state(loaded["rng"][0])
        assert (random.random(), float(np.random.rand()), torch.rand(1).item()) == expected

    def test_numpy_rng_state_written_by_older_versions_still_restores(self):
        import numpy as np

        from flextrain.checkpoint import capture_rng_state, restore_rng_state

        state = capture_rng_state()
        state["numpy"] = np.random.get_state()  # the FlexTrain <= 0.2 representation: a tuple holding an ndarray
        expected = float(np.random.rand())
        restore_rng_state(state)
        assert float(np.random.rand()) == expected


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_file(self, f, size=None):
        self.store[self.name] = f.read()

    def download_to_file(self, f):
        f.write(self.store[self.name])

    def exists(self):
        return self.name in self.store

    def delete(self):
        del self.store[self.name]

    def upload_from_string(self, data):
        self.store[self.name] = bytes(data)

    def download_as_bytes(self):
        return self.store[self.name]


class FakeGCSClient:
    def __init__(self):
        self.store = {}

    def bucket(self, name):
        client = self

        class Bucket:
            def blob(self, key):
                return FakeBlob(client.store, key)

        return Bucket()

    def list_blobs(self, bucket, prefix, delimiter):
        return [FakeBlob(self.store, k) for k in self.store
                if k.startswith(prefix) and "/" not in k[len(prefix):]]


class FakeS3Client:
    class _NotFound(Exception):
        response = {"Error": {"Code": "404"}}

    def __init__(self):
        self.store = {}

    def upload_fileobj(self, f, bucket, key):
        self.store[key] = f.read()

    def download_fileobj(self, bucket, key, f):
        f.write(self.store[key])

    def head_object(self, Bucket, Key):
        if Key not in self.store:
            raise self._NotFound()

    def delete_object(self, Bucket, Key):
        self.store.pop(Key, None)

    def put_object(self, Bucket, Key, Body):
        self.store[Key] = bytes(Body)

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.store[Key])}

    def get_paginator(self, name):
        store = self.store

        class Paginator:
            def paginate(self, Bucket, Prefix, Delimiter):
                keys = [k for k in store if k.startswith(Prefix) and "/" not in k[len(Prefix):]]
                yield {"Contents": [{"Key": k} for k in keys]}

        return Paginator()


@pytest.mark.parametrize("make", [lambda: GCSStorage("bucket", client=FakeGCSClient()),
                                  lambda: S3Storage("bucket", client=FakeS3Client())], ids=["gcs", "s3"])
def test_cloud_backends(make):
    storage = make()
    path = storage.join("runs/exp/checkpoints", "checkpoint_step00000001.pt")
    assert not storage.exists(path)
    nbytes = storage.save({"t": torch.ones(3)}, path)
    assert nbytes > 0 and storage.exists(path)
    storage.save({"t": torch.ones(1)}, "runs/exp/checkpoints/nested/other.pt")
    assert sorted(storage.list("runs/exp/checkpoints")) == [path, digest_path(path)]  # direct children only
    assert torch.equal(storage.load(path)["t"], torch.ones(3))
    storage.delete_checkpoint(path)
    assert not storage.exists(path) and not storage.exists(digest_path(path))


@pytest.mark.parametrize("make", [lambda: GCSStorage("bucket", client=FakeGCSClient()),
                                  lambda: S3Storage("bucket", client=FakeS3Client())], ids=["gcs", "s3"])
def test_cloud_backends_detect_corrupted_objects(make):
    storage = make()
    weights = torch.arange(4096, dtype=torch.float32)
    storage.save({"w": weights}, "ckpt/a.pt")
    store = storage.client.store if isinstance(storage, GCSStorage) else storage.s3.store
    store["ckpt/a.pt"] = flip_byte_in_tensor_data(store["ckpt/a.pt"], weights)
    with pytest.raises(CheckpointCorruptedError, match="checksum mismatch"):
        storage.load("ckpt/a.pt")


def test_s3_exists_reraises_non_404():
    client = FakeS3Client()

    class Denied(Exception):
        response = {"Error": {"Code": "403"}}

    def head(**kwargs):
        raise Denied()

    client.head_object = head
    with pytest.raises(Denied):
        S3Storage("b", client=client).exists("k")


def test_factory():
    assert isinstance(create_storage_backend("local"), LocalStorage)
    with pytest.raises(ValueError):
        create_storage_backend("invalid")
    with pytest.raises(ValueError, match="bucket_name"):
        create_storage_backend("s3")


def test_serialized_size_matches_bytes():
    buf = io.BytesIO()
    torch.save({"a": torch.zeros(100)}, buf)
    storage = S3Storage("b", client=FakeS3Client())
    assert storage.save({"a": torch.zeros(100)}, "k.pt") == len(buf.getvalue())
