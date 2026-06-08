"""Database configuration for the Traffic AI persistence layer.

Supports PostgreSQL (production) and SQLite (local development) via
the DATABASE_URL environment variable.  Storage backend for violation
crop images is configured through STORAGE_BACKEND / ASSETS_DIR / MINIO_*
env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Database URL ──────────────────────────────────────────────────
# Default: PostgreSQL for production.
# Override with "sqlite:///./traffic_system.db" for local dev.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://traffic:traffic@localhost:5432/traffic_ai",
)

engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL connection pool tuning
    engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "5"))
    engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "10"))

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ── Asset / storage config ────────────────────────────────────────
# "local", "minio", "cloudinary", or "none"
STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")

# Local filesystem root for violation crops
ASSETS_DIR: Path = Path(os.getenv("ASSETS_DIR", "data/assets"))

# MinIO connection parameters
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "traffic-violations")
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Cloudinary connection parameters
CLOUDINARY_CLOUD_NAME: str | None = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY: str | None = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET: str | None = os.getenv("CLOUDINARY_API_SECRET")
CLOUDINARY_FOLDER: str = os.getenv("CLOUDINARY_FOLDER", "traffic-ai")


def get_db():
    """FastAPI dependency — yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
