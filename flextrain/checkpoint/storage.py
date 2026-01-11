"""Storage backends for checkpoints."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import torch

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    def save(self, state: Dict[str, Any], path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        pass

    @abstractmethod
    def delete(self, path: Union[str, Path]) -> None:
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def save(self, state: Dict[str, Any], path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(state, temp_path)
            temp_path.rename(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def exists(self, path: Union[str, Path]) -> bool:
        return Path(path).exists()

    def delete(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.exists():
            path.unlink()


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError("google-cloud-storage required: pip install google-cloud-storage")

        self.client = storage.Client() if not credentials_path else storage.Client.from_service_account_json(credentials_path)
        self.bucket = self.client.bucket(bucket_name)

    def save(self, state: Dict[str, Any], path: Union[str, Path]) -> None:
        import io
        blob = self.bucket.blob(str(path).lstrip("/"))
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer)

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        import io
        blob = self.bucket.blob(str(path).lstrip("/"))
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def exists(self, path: Union[str, Path]) -> bool:
        return self.bucket.blob(str(path).lstrip("/")).exists()

    def delete(self, path: Union[str, Path]) -> None:
        self.bucket.blob(str(path).lstrip("/")).delete()


class S3Storage(StorageBackend):
    """Amazon S3 storage backend."""

    def __init__(self, bucket_name: str, region: Optional[str] = None):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required: pip install boto3")

        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3", region_name=region)

    def save(self, state: Dict[str, Any], path: Union[str, Path]) -> None:
        import io
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket_name, str(path).lstrip("/"))

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        import io
        buffer = io.BytesIO()
        self.s3.download_fileobj(self.bucket_name, str(path).lstrip("/"), buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)

    def exists(self, path: Union[str, Path]) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=str(path).lstrip("/"))
            return True
        except:
            return False

    def delete(self, path: Union[str, Path]) -> None:
        self.s3.delete_object(Bucket=self.bucket_name, Key=str(path).lstrip("/"))


def create_storage_backend(backend_type: str, bucket_name: Optional[str] = None, **kwargs) -> StorageBackend:
    """Create a storage backend."""
    backend_type = backend_type.lower()
    if backend_type == "local":
        return LocalStorage()
    elif backend_type == "gcs":
        return GCSStorage(bucket_name, **kwargs)
    elif backend_type == "s3":
        return S3Storage(bucket_name, **kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")
