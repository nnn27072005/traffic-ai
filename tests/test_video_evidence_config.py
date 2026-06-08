import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import crud, models, schemas
from src.api.database import Base
from src.api.video_processor import violation_worker


def make_violation_payload(frame_id=1):
    return {
        "track_id": 7,
        "violations": ["WithoutHelmet"],
        "confidence": 0.91,
        "bbox": {"x1": 5, "y1": 5, "x2": 35, "y2": 35},
        "class_name": "Bike",
        "frame_id": frame_id,
    }


def make_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def test_create_violation_without_frame_does_not_create_asset():
    SessionLocal = make_db_session()
    with SessionLocal() as db:
        violation = crud.create_violation(db, make_violation_payload(), frame=None)

        assert violation.image_path is None
        assert violation.asset_id is None
        assert db.query(models.ViolationAsset).count() == 0


def test_create_violation_with_frame_creates_asset(monkeypatch):
    SessionLocal = make_db_session()

    class FakeStorage:
        def save(self, data, prefix, filename, mime_type="image/jpeg"):
            return {
                "storage_backend": "local",
                "bucket_name": None,
                "object_key": f"{prefix}/{filename}",
                "local_path": "fake-local-path.jpg",
                "mime_type": mime_type,
                "file_size_bytes": len(data),
                "sha256": "a" * 64,
            }

    monkeypatch.setattr(crud, "get_storage", lambda: FakeStorage())
    monkeypatch.setattr("src.api.database.STORAGE_BACKEND", "local")
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    with SessionLocal() as db:
        violation = crud.create_violation(db, make_violation_payload(), frame=frame)

        assert violation.image_path is not None
        assert violation.asset_id is not None
        assert db.query(models.ViolationAsset).count() == 1


def test_video_url_request_defaults_disable_output_and_evidence():
    request = schemas.VideoUrlRequest(url="https://example.com/video.mp4")

    assert request.save_output is False
    assert request.save_evidence is False


def test_violation_asset_url_supports_cloudinary_and_local():
    local_violation = models.Violation(
        track_id=1,
        violation_type="WithoutHelmet",
        confidence=0.8,
        class_name="Bike",
        bbox={"x1": 1, "y1": 1, "x2": 10, "y2": 10},
    )
    local_violation.asset = models.ViolationAsset(
        storage_backend="local",
        object_key="violations/2026-06-08/v_1.jpg",
    )

    cloudinary_violation = models.Violation(
        track_id=2,
        violation_type="WithoutHelmet",
        confidence=0.9,
        class_name="Bike",
        bbox={"x1": 1, "y1": 1, "x2": 10, "y2": 10},
    )
    cloudinary_violation.asset = models.ViolationAsset(
        storage_backend="cloudinary",
        object_key="traffic-ai/violations/v_2",
        local_path="https://res.cloudinary.com/demo/image/upload/v_2.jpg",
    )

    assert local_violation.asset_url == "/api/assets/violations/2026-06-08/v_1.jpg"
    assert cloudinary_violation.asset_url == "https://res.cloudinary.com/demo/image/upload/v_2.jpg"


def test_violation_worker_respects_save_evidence_false(monkeypatch):
    captured_frames = []

    @contextmanager
    def fake_session():
        yield object()

    def fake_create_violation(db, violation, **kwargs):
        captured_frames.append(kwargs.get("frame"))

    pipeline = SimpleNamespace(
        violation_queue=asyncio.Queue(),
        analyzer=SimpleNamespace(
            analyze=lambda frame, tracks, frame_idx: [
                SimpleNamespace(
                    track_id=7,
                    violations=["WithoutHelmet"],
                    confidence=0.91,
                    bbox_xyxy=np.array([5, 5, 35, 35]),
                )
            ]
        ),
    )
    track = SimpleNamespace(track_id=7, class_name="Motorcycle")
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    monkeypatch.setattr("src.api.video_processor.database.SessionLocal", fake_session)
    monkeypatch.setattr("src.api.video_processor.crud.create_violation", fake_create_violation)

    async def run_worker():
        task = asyncio.create_task(violation_worker(pipeline, session_id=1, save_evidence=False))
        await pipeline.violation_queue.put({
            "frame": frame,
            "tracks": [track],
            "frame_idx": 1,
            "save_evidence": False,
        })
        await pipeline.violation_queue.put(None)
        await task

    asyncio.run(run_worker())

    assert captured_frames == [None]


def test_violation_worker_respects_save_evidence_true(monkeypatch):
    captured_frames = []

    @contextmanager
    def fake_session():
        yield object()

    def fake_create_violation(db, violation, **kwargs):
        captured_frames.append(kwargs.get("frame"))

    pipeline = SimpleNamespace(
        violation_queue=asyncio.Queue(),
        analyzer=SimpleNamespace(
            analyze=lambda frame, tracks, frame_idx: [
                SimpleNamespace(
                    track_id=7,
                    violations=["WithoutHelmet"],
                    confidence=0.91,
                    bbox_xyxy=np.array([5, 5, 35, 35]),
                )
            ]
        ),
    )
    track = SimpleNamespace(track_id=7, class_name="Motorcycle")
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    monkeypatch.setattr("src.api.video_processor.database.SessionLocal", fake_session)
    monkeypatch.setattr("src.api.video_processor.crud.create_violation", fake_create_violation)

    async def run_worker():
        task = asyncio.create_task(violation_worker(pipeline, session_id=1, save_evidence=True))
        await pipeline.violation_queue.put({
            "frame": frame,
            "tracks": [track],
            "frame_idx": 1,
            "save_evidence": True,
        })
        await pipeline.violation_queue.put(None)
        await task

    asyncio.run(run_worker())

    assert len(captured_frames) == 1
    assert captured_frames[0] is frame
