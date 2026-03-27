# src/analyzer/violation.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.detector.yolo_wrapper import YOLODetector, Detection
from src.tracker.bytetrack_wrapper import Track


# ── Class IDs — khớp với configs/helmet.yaml ─────────────────────
PLATE_CLASS_ID     = 0
HELMET_CLASS_ID    = 1
NO_HELMET_CLASS_ID = 2
VIOLATION_CLASS_IDS = {NO_HELMET_CLASS_ID}

# Classes từ traffic model (fisheye8k) cần analyze
BIKE_CLASS_NAMES = {"Bike"}


@dataclass
class ViolationEvent:
    track_id:    int
    bbox_xyxy:   np.ndarray
    violations:  list[str]        # ["WithoutHelmet"]
    plate_text:  str | None       # future: OCR on Plate bbox
    confidence:  float            # max violation confidence
    frame_idx:   int


@dataclass
class ViolationStats:
    total_violations:  int = 0
    total_compliant:   int = 0    # WithHelmet detected
    plates_detected:   int = 0
    per_type: dict[str, int] = field(default_factory=dict)

    @property
    def violation_rate(self) -> float:
        total = self.total_violations + self.total_compliant
        return self.total_violations / max(total, 1)


class HelmetViolationAnalyzer:
    """
    Cascade detection pipeline:
      1. Nhận motorcycle tracks từ ByteTracker
      2. Crop region quanh mỗi xe, expand để capture người ngồi
      3. Chạy helmet model trên crop
      4. Classify: WithHelmet / WithoutHelmet / Plate

    Dataset note: HelmetViolations dùng top-view + grayscale.
    Model sẽ hoạt động tốt hơn với camera góc cao.
    Với video góc thấp, confidence threshold nên giảm xuống 0.25.
    """

    def __init__(
        self,
        helmet_model_path: str | Path,
        confidence:        float = 0.30,
        device:            str   = "cuda",
        imgsz:             int   = 640,
        pad_ratio:         float = 0.30,
        cooldown_frames:   int   = 15,
    ):
        self.detector = YOLODetector(
            model_path = helmet_model_path,
            confidence = confidence,
            device     = device,
            imgsz      = imgsz,
        )
        self.pad_ratio       = pad_ratio
        self.cooldown_frames = cooldown_frames
        self._stats          = ViolationStats()
        self._last_alert:  dict[int, int] = {}

    def analyze(
        self,
        frame:     np.ndarray,
        tracks:    list[Track],
        frame_idx: int = 0,
    ) -> list[ViolationEvent]:

        events = []

        for track in tracks:
            if track.class_name not in BIKE_CLASS_NAMES:
                continue

            # Cooldown
            last = self._last_alert.get(track.track_id, -self.cooldown_frames)
            if frame_idx - last < self.cooldown_frames:
                continue

            crop, valid = self._extract_crop(frame, track.bbox_xyxy)
            if not valid:
                continue

            detections = self.detector.detect(crop)
            if not detections:
                continue

            # Tách theo loại
            violations = [d for d in detections
                          if d.class_id == NO_HELMET_CLASS_ID]
            helmets    = [d for d in detections
                          if d.class_id == HELMET_CLASS_ID]
            plates     = [d for d in detections
                          if d.class_id == PLATE_CLASS_ID]

            # Update compliance stats
            if helmets:
                self._stats.total_compliant += 1
            if plates:
                self._stats.plates_detected += 1

            if violations:
                max_conf = max(d.confidence for d in violations)
                events.append(ViolationEvent(
                    track_id   = track.track_id,
                    bbox_xyxy  = track.bbox_xyxy,
                    violations = [d.class_name for d in violations],
                    plate_text = None,
                    confidence = max_conf,
                    frame_idx  = frame_idx,
                ))
                self._stats.total_violations += 1
                self._stats.per_type["WithoutHelmet"] = \
                    self._stats.per_type.get("WithoutHelmet", 0) + 1
                self._last_alert[track.track_id] = frame_idx

        return events

    def _extract_crop(
        self,
        frame: np.ndarray,
        bbox:  np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        h, w   = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = x2 - x1, y2 - y1

        px = int(bw * self.pad_ratio)
        py = int(bh * self.pad_ratio * 1.5)

        x1c = max(0, x1 - px)
        y1c = max(0, y1 - py)
        x2c = min(w, x2 + px)
        y2c = min(h, y2 + py)

        if (x2c - x1c) < 20 or (y2c - y1c) < 20:
            return np.zeros((1, 1, 3), np.uint8), False

        return frame[y1c:y2c, x1c:x2c], True

    def get_stats(self) -> ViolationStats:
        return self._stats


# ── Annotation ────────────────────────────────────────────────────

def draw_violations(
    frame:      np.ndarray,
    violations: list[ViolationEvent],
) -> np.ndarray:
    for event in violations:
        x1, y1, x2, y2 = map(int, event.bbox_xyxy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = (f"NO HELMET #{event.track_id} "
                 f"({event.confidence:.0%})")

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(frame,
                      (x1, max(0, y1 - th - 12)),
                      (x1 + tw + 4, y1),
                      (0, 0, 255), -1)
        cv2.putText(frame, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame