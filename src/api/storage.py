"""Asset storage backends for violation crop images.

Two concrete implementations:
  - LocalStorage  → writes to ASSETS_DIR on the local filesystem
  - MinIOStorage  → writes to an S3-compatible MinIO bucket

Usage:
    storage = get_storage()
    asset_info = storage.save(image_bytes, "violations", "v_42_7_20260426.jpg")
    url = storage.get_url(asset_info["object_key"])
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.api.database import (
    ASSETS_DIR,
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_FOLDER,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    STORAGE_BACKEND,
)

logger = logging.getLogger("traffic-api.storage")


# ── Protocol ──────────────────────────────────────────────────────

@runtime_checkable
class AssetStorage(Protocol):
    """Interface for storing binary assets (images, video clips)."""

    def save(
        self,
        data: bytes,
        prefix: str,
        filename: str,
        mime_type: str = "image/jpeg",
    ) -> dict:
        """Save binary data.  Returns metadata dict with keys:
        storage_backend, bucket_name, object_key, local_path,
        mime_type, sha256, file_size_bytes.
        """
        ...

    def get_url(self, object_key: str) -> str:
        """Return a URL (or local path) to retrieve the asset."""
        ...

    def delete(self, object_key: str) -> bool:
        """Delete an asset.  Returns True if successful."""
        ...


# ── Local Filesystem ─────────────────────────────────────────────

class LocalStorage:
    """Stores assets under ASSETS_DIR/{prefix}/{date}/{filename}."""

    def __init__(self, root: Path | None = None):
        self.root = root or ASSETS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        data: bytes,
        prefix: str,
        filename: str,
        mime_type: str = "image/jpeg",
    ) -> dict:
        date_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rel_path = Path(prefix) / date_dir / filename
        abs_path = self.root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        abs_path.write_bytes(data)
        sha256 = hashlib.sha256(data).hexdigest()

        logger.info("Saved asset locally: %s (%d bytes)", abs_path, len(data))
        return {
            "storage_backend": "local",
            "bucket_name": None,
            "object_key": str(rel_path),
            "local_path": str(abs_path),
            "mime_type": mime_type,
            "sha256": sha256,
            "file_size_bytes": len(data),
        }

    def get_url(self, object_key: str) -> str:
        return f"/api/assets/{object_key}"

    def delete(self, object_key: str) -> bool:
        path = self.root / object_key
        if path.exists():
            path.unlink()
            return True
        return False


# ── MinIO (S3-compatible) ─────────────────────────────────────────

class MinIOStorage:
    """Stores assets in a MinIO bucket.

    Requires the ``minio`` Python package.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool | None = None,
    ):
        from minio import Minio  # lazy import — only needed if backend=minio

        self.endpoint = endpoint or MINIO_ENDPOINT
        self.access_key = access_key or MINIO_ACCESS_KEY
        self.secret_key = secret_key or MINIO_SECRET_KEY
        self.bucket = bucket or MINIO_BUCKET
        self.secure = secure if secure is not None else MINIO_SECURE

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

        # Ensure bucket exists
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info("Created MinIO bucket: %s", self.bucket)

    def save(
        self,
        data: bytes,
        prefix: str,
        filename: str,
        mime_type: str = "image/jpeg",
    ) -> dict:
        date_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        object_key = f"{prefix}/{date_dir}/{filename}"
        sha256 = hashlib.sha256(data).hexdigest()

        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=mime_type,
        )

        logger.info(
            "Saved asset to MinIO: %s/%s (%d bytes)",
            self.bucket,
            object_key,
            len(data),
        )
        return {
            "storage_backend": "minio",
            "bucket_name": self.bucket,
            "object_key": object_key,
            "local_path": None,
            "mime_type": mime_type,
            "sha256": sha256,
            "file_size_bytes": len(data),
        }

    def get_url(self, object_key: str) -> str:
        """Generate a presigned URL valid for 1 hour."""
        from datetime import timedelta

        return self.client.presigned_get_object(
            self.bucket,
            object_key,
            expires=timedelta(hours=1),
        )

    def delete(self, object_key: str) -> bool:
        try:
            self.client.remove_object(self.bucket, object_key)
            return True
        except Exception:
            logger.exception("Failed to delete object %s", object_key)
            return False


# ── Cloudinary ────────────────────────────────────────────────────

class CloudinaryStorage:
    """Stores evidence images in Cloudinary."""

    def __init__(
        self,
        cloud_name: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        folder: str | None = None,
    ):
        import cloudinary

        self.cloud_name = cloud_name or CLOUDINARY_CLOUD_NAME
        self.api_key = api_key or CLOUDINARY_API_KEY
        self.api_secret = api_secret or CLOUDINARY_API_SECRET
        self.folder = (folder or CLOUDINARY_FOLDER).strip("/")

        missing = [
            name
            for name, value in {
                "CLOUDINARY_CLOUD_NAME": self.cloud_name,
                "CLOUDINARY_API_KEY": self.api_key,
                "CLOUDINARY_API_SECRET": self.api_secret,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Cloudinary config: {', '.join(missing)}")

        cloudinary.config(
            cloud_name=self.cloud_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
            secure=True,
        )

    def save(
        self,
        data: bytes,
        prefix: str,
        filename: str,
        mime_type: str = "image/jpeg",
    ) -> dict:
        import cloudinary.uploader

        date_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stem = Path(filename).stem
        folder = f"{self.folder}/{prefix}/{date_dir}" if self.folder else f"{prefix}/{date_dir}"
        sha256 = hashlib.sha256(data).hexdigest()

        result = cloudinary.uploader.upload(
            io.BytesIO(data),
            folder=folder,
            public_id=stem,
            resource_type="image",
            overwrite=True,
        )
        public_id = result["public_id"]
        secure_url = result.get("secure_url")

        logger.info("Saved asset to Cloudinary: %s (%d bytes)", public_id, len(data))
        return {
            "storage_backend": "cloudinary",
            "bucket_name": self.cloud_name,
            "object_key": public_id,
            "local_path": secure_url,
            "mime_type": mime_type,
            "sha256": sha256,
            "file_size_bytes": len(data),
        }

    def get_url(self, object_key: str) -> str:
        import cloudinary.utils

        url, _ = cloudinary.utils.cloudinary_url(
            object_key,
            resource_type="image",
            secure=True,
        )
        return url

    def delete(self, object_key: str) -> bool:
        import cloudinary.uploader

        result = cloudinary.uploader.destroy(object_key, resource_type="image")
        return result.get("result") in {"ok", "not found"}


# ── Null Storage (No-op) ──────────────────────────────────────────

class NullStorage:
    """A no-op storage backend that does not save anything."""

    def save(
        self,
        data: bytes,
        prefix: str,
        filename: str,
        mime_type: str = "image/jpeg",
    ) -> dict:
        logger.info("NullStorage: Simulating save of %s (%d bytes)", filename, len(data))
        sha256 = hashlib.sha256(data).hexdigest()
        return {
            "storage_backend": "none",
            "bucket_name": None,
            "object_key": f"{prefix}/{filename}",
            "local_path": None,
            "mime_type": mime_type,
            "sha256": sha256,
            "file_size_bytes": len(data),
        }

    def get_url(self, object_key: str) -> str:
        return ""

    def delete(self, object_key: str) -> bool:
        return True


# ── Factory ───────────────────────────────────────────────────────

_storage_instance: AssetStorage | None = None


def get_storage() -> AssetStorage:
    """Return the configured storage backend (singleton)."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    if STORAGE_BACKEND == "minio":
        _storage_instance = MinIOStorage()
    elif STORAGE_BACKEND == "cloudinary":
        _storage_instance = CloudinaryStorage()
    elif STORAGE_BACKEND in ("none", "noop"):
        _storage_instance = NullStorage()
    else:
        _storage_instance = LocalStorage()

    logger.info("Initialized storage backend: %s", STORAGE_BACKEND)
    return _storage_instance
