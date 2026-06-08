"""CRUD operations for the Traffic AI persistence layer.

Provides service-level functions for:
  - Camera management
  - Processing session lifecycle
  - Traffic count recording
  - Violation logging with automatic crop storage and severity classification
  - Aggregated statistics and time-series queries
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from sqlalchemy import func, case
from sqlalchemy.orm import Session

from src.api import models
from src.api.models import auto_classify_severity
from src.api.storage import get_storage

logger = logging.getLogger("traffic-api.crud")


# ══════════════════════════════════════════════════════════════════
#  CAMERAS
# ══════════════════════════════════════════════════════════════════


def create_camera(db: Session, *, name: str, **kwargs) -> models.Camera:
    """Register a new camera source."""
    camera = models.Camera(name=name, **kwargs)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def get_camera(db: Session, camera_id: int) -> models.Camera | None:
    return db.query(models.Camera).filter(models.Camera.id == camera_id).first()


def get_cameras(db: Session, *, active_only: bool = False) -> list[models.Camera]:
    q = db.query(models.Camera)
    if active_only:
        q = q.filter(models.Camera.is_active.is_(True))
    return q.order_by(models.Camera.name).all()


def get_or_create_default_camera(db: Session) -> models.Camera:
    """Return the default camera, creating it if it doesn't exist."""
    camera = db.query(models.Camera).filter(models.Camera.name == "default").first()
    if camera is None:
        camera = create_camera(
            db,
            name="default",
            location="Default Camera",
            description="Auto-created default camera for uploads",
        )
    return camera


# ══════════════════════════════════════════════════════════════════
#  PROCESSING SESSIONS
# ══════════════════════════════════════════════════════════════════


def create_session(
    db: Session,
    *,
    camera_id: int | None = None,
    source_type: str = "upload",
    source_path: str | None = None,
    config: dict | None = None,
) -> models.ProcessingSession:
    """Start a new processing session."""
    session = models.ProcessingSession(
        camera_id=camera_id,
        source_type=source_type,
        source_path=source_path,
        status="pending",
        config=config,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info("Created session %d (source=%s)", session.id, source_type)
    return session


def update_session(
    db: Session,
    session_id: int,
    **kwargs,
) -> models.ProcessingSession | None:
    """Update session fields (status, processed_frames, etc.)."""
    session = (
        db.query(models.ProcessingSession)
        .filter(models.ProcessingSession.id == session_id)
        .first()
    )
    if session is None:
        return None

    for key, value in kwargs.items():
        if hasattr(session, key) and value is not None:
            setattr(session, key, value)

    # Auto-set completed_at when status transitions to terminal
    if kwargs.get("status") in ("completed", "failed") and session.completed_at is None:
        session.completed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: int) -> models.ProcessingSession | None:
    return (
        db.query(models.ProcessingSession)
        .filter(models.ProcessingSession.id == session_id)
        .first()
    )


def get_sessions(
    db: Session, *, skip: int = 0, limit: int = 20
) -> list[models.ProcessingSession]:
    return (
        db.query(models.ProcessingSession)
        .order_by(models.ProcessingSession.started_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def stop_session(db: Session, session_id: int) -> models.ProcessingSession | None:
    """Manually stop a processing session."""
    return update_session(db, session_id, status="stopped")


# ══════════════════════════════════════════════════════════════════
#  TRAFFIC COUNTS
# ══════════════════════════════════════════════════════════════════


def create_traffic_stats(
    db: Session,
    stats: dict[str, Any],
    *,
    camera_id: int | None = None,
    session_id: int | None = None,
) -> models.TrafficCount:
    """Record a traffic count snapshot."""
    db_stats = models.TrafficCount(
        camera_id=camera_id,
        session_id=session_id,
        vehicle_count=stats["vehicle_count"],
        count_in=stats["count_in"],
        count_out=stats["count_out"],
        latency_ms=stats["latency_ms"],
        per_class_count=stats.get("per_class_count"),
        frame_id=stats.get("frame_id"),
    )
    db.add(db_stats)
    db.commit()
    db.refresh(db_stats)
    return db_stats


def get_latest_stats(db: Session) -> models.TrafficCount | None:
    return (
        db.query(models.TrafficCount)
        .order_by(models.TrafficCount.timestamp.desc())
        .first()
    )


def get_traffic_time_series(
    db: Session,
    *,
    camera_id: int | None = None,
    session_id: int | None = None,
    limit: int = 100,
) -> list[models.TrafficCount]:
    """Return recent traffic count records for charting."""
    q = db.query(models.TrafficCount)
    if camera_id is not None:
        q = q.filter(models.TrafficCount.camera_id == camera_id)
    if session_id is not None:
        q = q.filter(models.TrafficCount.session_id == session_id)
    return q.order_by(models.TrafficCount.timestamp.desc()).limit(limit).all()


# ══════════════════════════════════════════════════════════════════
#  VIOLATIONS
# ══════════════════════════════════════════════════════════════════


def create_violation(
    db: Session,
    violation: dict[str, Any],
    *,
    camera_id: int | None = None,
    session_id: int | None = None,
    frame: np.ndarray | None = None,
) -> models.Violation:
    """Create a violation record with optional crop storage.

    If *frame* is provided, the bbox region is cropped, saved to the
    configured storage backend, and linked via a ViolationAsset row.
    Severity is auto-classified from violation_type and confidence.
    """
    violation_type = ",".join(violation["violations"])
    confidence = violation["confidence"]

    # Auto-classify severity
    severity = auto_classify_severity(violation_type, confidence)

    # Prepare the violation record
    db_violation = models.Violation(
        camera_id=camera_id,
        session_id=session_id,
        track_id=violation["track_id"],
        violation_type=violation_type,
        confidence=confidence,
        class_name=violation.get("class_name", "unknown"),
        frame_id=violation.get("frame_id"),
        bbox=violation["bbox"],
        severity=severity,
        reviewed=False,
    )

    # Save crop if frame is provided
    if frame is not None:
        asset = _save_violation_crop(
            db,
            frame=frame,
            bbox=violation["bbox"],
            track_id=violation["track_id"],
            frame_id=violation.get("frame_id"),
        )
        if asset is not None:
            db_violation.asset_id = asset.id
            db_violation.image_path = asset.object_key

    db.add(db_violation)
    db.commit()
    db.refresh(db_violation)
    return db_violation


def _save_violation_crop(
    db: Session,
    *,
    frame: np.ndarray,
    bbox: dict,
    track_id: int,
    frame_id: int | None,
) -> models.ViolationAsset | None:
    """Crop the violation region from the frame and save to storage."""
    from src.api.database import STORAGE_BACKEND
    if STORAGE_BACKEND in ("none", "noop"):
        return None

    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox["x1"]))
        y1 = max(0, int(bbox["y1"]))
        x2 = min(w, int(bbox["x2"]))
        y2 = min(h, int(bbox["y2"]))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        # Add 10% padding
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_bytes = buf.tobytes()

        # Generate filename
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"v_{track_id}_{frame_id or 0}_{ts}.jpg"

        storage = get_storage()
        info = storage.save(image_bytes, "violations", fname)

        asset = models.ViolationAsset(
            storage_backend=info["storage_backend"],
            bucket_name=info.get("bucket_name"),
            object_key=info["object_key"],
            local_path=info.get("local_path"),
            mime_type=info["mime_type"],
            width=x2 - x1,
            height=y2 - y1,
            file_size_bytes=info["file_size_bytes"],
            sha256=info["sha256"],
        )
        db.add(asset)
        db.flush()  # Get ID without committing (caller will commit)
        return asset

    except Exception:
        logger.exception("Failed to save violation crop")
        return None


def get_violations(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 100,
    camera_id: int | None = None,
    session_id: int | None = None,
    violation_type: str | None = None,
    severity: str | None = None,
    reviewed: bool | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[models.Violation]:
    """Query violations with comprehensive filtering."""
    q = db.query(models.Violation)

    if camera_id is not None:
        q = q.filter(models.Violation.camera_id == camera_id)
    if session_id is not None:
        q = q.filter(models.Violation.session_id == session_id)
    if violation_type is not None:
        q = q.filter(models.Violation.violation_type == violation_type)
    if severity is not None:
        q = q.filter(models.Violation.severity == severity)
    if reviewed is not None:
        q = q.filter(models.Violation.reviewed == reviewed)
    if date_from is not None:
        q = q.filter(models.Violation.timestamp >= date_from)
    if date_to is not None:
        q = q.filter(models.Violation.timestamp <= date_to)

    return (
        q.order_by(models.Violation.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_violation(db: Session, violation_id: int) -> models.Violation | None:
    return (
        db.query(models.Violation)
        .filter(models.Violation.id == violation_id)
        .first()
    )


def review_violation(
    db: Session,
    violation_id: int,
    *,
    reviewed: bool = True,
    severity: str | None = None,
    notes: str | None = None,
) -> models.Violation | None:
    """Mark a violation as reviewed and optionally update severity/notes."""
    violation = get_violation(db, violation_id)
    if violation is None:
        return None

    violation.reviewed = reviewed
    if severity is not None:
        violation.severity = severity
    if notes is not None:
        violation.notes = notes

    db.commit()
    db.refresh(violation)
    return violation


# ══════════════════════════════════════════════════════════════════
#  AGGREGATED STATISTICS
# ══════════════════════════════════════════════════════════════════


def get_summary_stats(db: Session) -> dict[str, Any]:
    """Return aggregated summary statistics for the dashboard."""
    # Total violations count
    total_violations = db.query(func.count(models.Violation.id)).scalar() or 0

    # Violations by severity
    severity_rows = (
        db.query(models.Violation.severity, func.count(models.Violation.id))
        .group_by(models.Violation.severity)
        .all()
    )
    violations_by_severity = {row[0]: row[1] for row in severity_rows}

    # Violations by type
    type_rows = (
        db.query(models.Violation.violation_type, func.count(models.Violation.id))
        .group_by(models.Violation.violation_type)
        .all()
    )
    violations_by_type = {row[0]: row[1] for row in type_rows}

    # Latest traffic stats
    latest = get_latest_stats(db)

    # Average latency across all counts
    avg_latency = db.query(func.avg(models.TrafficCount.latency_ms)).scalar() or 0.0

    # Total counts
    total_in = db.query(func.max(models.TrafficCount.count_in)).scalar() or 0
    total_out = db.query(func.max(models.TrafficCount.count_out)).scalar() or 0

    return {
        "total_vehicles": latest.vehicle_count if latest else 0,
        "total_violations": total_violations,
        "avg_latency": round(float(avg_latency), 2),
        "count_in": total_in,
        "count_out": total_out,
        "violations_by_severity": violations_by_severity,
        "violations_by_type": violations_by_type,
    }


def get_daily_summary(db: Session, *, limit: int = 24) -> list[models.TrafficCount]:
    """Return recent traffic count records for daily charts."""
    return (
        db.query(models.TrafficCount)
        .order_by(models.TrafficCount.timestamp.desc())
        .limit(limit)
        .all()
    )
