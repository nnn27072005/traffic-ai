# tests/test_tracker.py
import numpy as np
import pytest
from src.detector.yolo_wrapper     import Detection
from src.tracker.bytetrack_wrapper import ByteTracker
from src.analyzer.counter          import LineCounter


def make_detection(x1, y1, x2, y2, cls_id=2, cls_name="Car", conf=0.9):
    return Detection(
        bbox_xyxy  = np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence = conf,
        class_id   = cls_id,
        class_name = cls_name,
    )


def test_tracker_returns_tracks():
    tracker = ByteTracker()
    dets = [make_detection(100, 100, 200, 200)]
    tracks = tracker.update(dets, frame_shape=(720, 1280))
    assert len(tracks) > 0


def test_tracker_assigns_ids():
    tracker = ByteTracker()
    dets = [
        make_detection(100, 100, 200, 200),
        make_detection(300, 300, 400, 400),
    ]
    tracks = tracker.update(dets, frame_shape=(720, 1280))
    ids = [t.track_id for t in tracks]
    assert len(set(ids)) == len(ids), "Track IDs phải unique"


def test_tracker_maintains_id_across_frames():
    tracker = ByteTracker()
    # Frame 1
    dets1 = [make_detection(100, 100, 200, 200)]
    tracks1 = tracker.update(dets1, frame_shape=(720, 1280))

    # Frame 2 — xe di chuyển nhẹ
    dets2 = [make_detection(105, 105, 205, 205)]
    tracks2 = tracker.update(dets2, frame_shape=(720, 1280))

    if tracks1 and tracks2:
        assert tracks1[0].track_id == tracks2[0].track_id, \
            "ID phải ổn định khi xe di chuyển nhẹ"


def test_counter_counts_crossing():
    counter = LineCounter(
        line_start = (0,   300),
        line_end   = (640, 300),
    )
    tracker = ByteTracker()

    # Simulate xe đi từ y=100 xuống y=500 (cross line y=300)
    positions = [100, 150, 200, 250, 300, 350, 400, 450, 500]

    for y in positions:
        dets = [make_detection(200, y, 260, y+60)]
        tracks = tracker.update(dets, frame_shape=(600, 640))
        counter.update(tracks)

    stats = counter.get_stats()
    assert stats.total_in + stats.total_out == 1, \
        f"Phải đếm đúng 1 lần crossing, got {stats.total}"


def test_counter_no_double_count():
    """Track đã cross rồi không được đếm lại."""
    counter = LineCounter(line_start=(0, 300), line_end=(640, 300))
    tracker = ByteTracker()

    # Xe cross xong rồi tiếp tục đi
    for y in range(100, 600, 20):
        dets = [make_detection(200, y, 260, y+60)]
        tracks = tracker.update(dets, frame_shape=(600, 640))
        counter.update(tracks)

    stats = counter.get_stats()
    assert stats.total == 1, \
        f"Không được double count, got {stats.total}"


# Chạy:
# !cd /kaggle/working/traffic-ai && python -m pytest tests/test_tracker.py -v