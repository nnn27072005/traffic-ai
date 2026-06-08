"""SQLAlchemy models for the Traffic AI persistence layer.

Tables
------
cameras              – Camera / source registration
processing_sessions  – Video upload & stream processing jobs
traffic_counts       – Per-frame vehicle counting snapshots
violations           – Helmet (and future) violation events
violation_assets     – Binary asset metadata for violation crops
"""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.api.database import Base


# ── Severity auto-classification ──────────────────────────────────
# Used by crud.auto_classify_severity() to derive severity from
# violation_type and confidence.
SEVERITY_RULES: dict[str, str] = {
    "WithoutHelmet": "high",
}
CONFIDENCE_THRESHOLDS = {
    "critical": 0.90,
    "high": 0.70,
    "medium": 0.40,
    "low": 0.0,
}


def auto_classify_severity(violation_type: str, confidence: float) -> str:
    """Derive severity from violation type and detection confidence.

    Rules (applied in order):
      1. If violation_type is in SEVERITY_RULES and confidence ≥ 0.90 → "critical"
      2. If violation_type is in SEVERITY_RULES and confidence ≥ 0.70 → "high"
      3. If confidence ≥ 0.40 → "medium"
      4. Otherwise → "low"
    """
    base = SEVERITY_RULES.get(violation_type)
    if base:
        if confidence >= CONFIDENCE_THRESHOLDS["critical"]:
            return "critical"
        if confidence >= CONFIDENCE_THRESHOLDS["high"]:
            return "high"
    if confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ── Helper: cross-dialect JSON type ──────────────────────────────
# PostgreSQL uses JSONB (indexed); SQLite falls back to plain JSON.
from sqlalchemy import JSON as _JSON
from sqlalchemy.types import TypeDecorator


class JSONType(TypeDecorator):
    """Uses JSONB on PostgreSQL, plain JSON on SQLite."""
    impl = _JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB)
        return dialect.type_descriptor(_JSON)


# ══════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════


class Camera(Base):
    """A registered camera / video source."""

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    location = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    stream_url = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    sessions = relationship("ProcessingSession", back_populates="camera")
    traffic_counts = relationship("TrafficCount", back_populates="camera")
    violations = relationship("Violation", back_populates="camera")

    def __repr__(self) -> str:
        return f"<Camera id={self.id} name={self.name!r}>"


class ProcessingSession(Base):
    """Tracks a single video-processing job (upload or live stream)."""

    __tablename__ = "processing_sessions"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Source info
    source_type = Column(
        String(32), nullable=False, default="upload",
        comment="upload | stream | rtsp",
    )
    source_path = Column(Text, nullable=True)

    # Progress
    status = Column(
        String(32), nullable=False, default="pending", index=True,
        comment="pending | processing | completed | failed",
    )
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, nullable=True, default=0)
    fps = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Processing config snapshot (line_ratio, confidence thresholds, etc.)
    config = Column(JSONType, nullable=True)

    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    output_video_path = Column(Text, nullable=True)

    # Relationships
    camera = relationship("Camera", back_populates="sessions")
    traffic_counts = relationship("TrafficCount", back_populates="session")
    violations = relationship("Violation", back_populates="session")

    @property
    def total_violations(self) -> int:
        return len(self.violations)

    def __repr__(self) -> str:
        return f"<ProcessingSession id={self.id} status={self.status!r}>"


class TrafficCount(Base):
    """Per-frame (or batched) vehicle counting snapshot."""

    __tablename__ = "traffic_counts"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id = Column(
        Integer,
        ForeignKey("processing_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
    )
    frame_id = Column(Integer, nullable=True, index=True)

    # Aggregate counts
    vehicle_count = Column(Integer, nullable=False, default=0)
    count_in = Column(Integer, nullable=False, default=0)
    count_out = Column(Integer, nullable=False, default=0)

    # Per-class breakdown: {"Car": 12, "Bike": 5, "Bus": 1, ...}
    per_class_count = Column(JSONType, nullable=True)

    latency_ms = Column(Float, nullable=False, default=0.0)
    extra_data = Column("metadata", JSONType, nullable=True)

    # Relationships
    camera = relationship("Camera", back_populates="traffic_counts")
    session = relationship("ProcessingSession", back_populates="traffic_counts")
    violations = relationship("Violation", back_populates="traffic_count")

    def __repr__(self) -> str:
        return f"<TrafficCount id={self.id} vehicles={self.vehicle_count}>"


class ViolationAsset(Base):
    """Binary asset metadata for a violation crop image (or video clip)."""

    __tablename__ = "violation_assets"

    id = Column(Integer, primary_key=True, index=True)
    storage_backend = Column(String(32), nullable=False, default="local")
    bucket_name = Column(String(255), nullable=True)
    object_key = Column(Text, nullable=False)
    local_path = Column(Text, nullable=True)
    mime_type = Column(String(128), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    sha256 = Column(String(64), nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ViolationAsset id={self.id} key={self.object_key!r}>"


class Violation(Base):
    """A detected traffic violation event."""

    __tablename__ = "violations"
    __table_args__ = (
        UniqueConstraint(
            "camera_id", "track_id", "frame_id", "violation_type",
            name="uq_violation_event",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id = Column(
        Integer,
        ForeignKey("processing_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    traffic_count_id = Column(
        Integer,
        ForeignKey("traffic_counts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    asset_id = Column(
        Integer,
        ForeignKey("violation_assets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Detection info
    track_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
    )
    violation_type = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False, default=0.0)
    class_name = Column(String(64), nullable=False, default="unknown")
    frame_id = Column(Integer, nullable=True, index=True)
    bbox = Column(JSONType, nullable=False)
    image_path = Column(Text, nullable=True)

    # Review workflow
    reviewed = Column(Boolean, nullable=False, default=False, index=True)
    severity = Column(
        String(16), nullable=False, default="medium", index=True,
        comment="low | medium | high | critical",
    )
    notes = Column(Text, nullable=True)

    extra_data = Column("metadata", JSONType, nullable=True)

    # Relationships
    camera = relationship("Camera", back_populates="violations")
    session = relationship("ProcessingSession", back_populates="violations")
    traffic_count = relationship("TrafficCount", back_populates="violations")
    asset = relationship("ViolationAsset")

    @property
    def asset_url(self) -> str | None:
        if self.asset is None:
            return None
        if self.asset.storage_backend == "local":
            return f"/api/assets/{self.asset.object_key}"
        if self.asset.storage_backend == "cloudinary":
            return self.asset.local_path
        if self.asset.storage_backend == "minio":
            try:
                from src.api.storage import get_storage

                return get_storage().get_url(self.asset.object_key)
            except Exception:
                return None
        return None

    def __repr__(self) -> str:
        return f"<Violation id={self.id} type={self.violation_type!r} severity={self.severity!r}>"


# ── User & Authentication ──────────────────────────────────────────

class User(Base):
    """System operator account for secure access control."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    email = Column(String(255), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=True)
    google_id = Column(String(255), unique=True, index=True, nullable=True)
    avatar_url = Column(Text, nullable=True)
    role = Column(String(32), default="operator")  # admin | operator
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role!r}>"


# ── Backward-compatible aliases ───────────────────────────────────
# The existing CRUD / API layer references these names.
TrafficStats = TrafficCount
ViolationRecord = Violation
