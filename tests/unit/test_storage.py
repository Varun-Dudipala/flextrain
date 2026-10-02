"""Storage backend tests (local for real, GCS/S3 against in-memory fakes)."""

import io
from pathlib import Path

import pytest
import torch

from flextrain.checkpoint.storage import GCSStorage, LocalStorage, S3Storage, create_storage_backend


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
        assert sorted(Path(p).name for p in storage.list(tmp_path)) == ["a.pt", "b.pt"]
        storage.delete(tmp_path / "a.pt")
        storage.delete(tmp_path / "a.pt")  # idempotent
        assert [Path(p).name for p in storage.list(tmp_path)] == ["b.pt"]
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
        assert sorted(p.name for p in tmp_path.iterdir()) == ["c.pt"]  # no temp file left behind
        assert storage.load(path)["v"] == 1  # previous checkpoint intact

    def test_fsync_is_called(self, tmp_path, monkeypatch):
        calls = []
        import os

        real_fsync = os.fsync
        monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd)))
        LocalStorage(fsync=True).save({"a": 1}, tmp_path / "a.pt")
        assert len(calls) >= 1
        calls.clear()
        LocalStorage(fsync=False).save({"a": 1}, tmp_path / "b.pt")
        assert calls == []


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
    assert storage.list("runs/exp/checkpoints") == [path]  # direct children only
    assert torch.equal(storage.load(path)["t"], torch.ones(3))
    storage.delete(path)
    assert not storage.exists(path)


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
