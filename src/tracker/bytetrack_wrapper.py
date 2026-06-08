# src/tracker/bytetrack_wrapper.py
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import supervision as sv

from src.detector.yolo_wrapper import Detection


@dataclass
class Track:
    """Một tracked object với ID ổn định qua các frames."""
    track_id:   int
    bbox_xyxy:  np.ndarray   # shape (4,)
    confidence: float
    class_id:   int
    class_name: str


class ByteTracker:
    """
    Wrapper quanh supervision.ByteTrack.

    Tại sao ByteTrack thay vì DeepSORT:
    - Không cần ReID model riêng → đơn giản hơn, nhanh hơn
    - Tận dụng low-confidence detections (0.1–0.5) để duy trì
      track khi xe bị che khuất — quan trọng với traffic Việt Nam
    - IDF1 tương đương DeepSORT nhưng latency thấp hơn ~3x
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 25,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold = track_activation_threshold,
            lost_track_buffer          = lost_track_buffer,
            minimum_matching_threshold = minimum_matching_threshold,
            frame_rate                 = frame_rate,
        )
        self._frame_count = 0

    def update(self, detections: list[Detection], frame_shape: tuple) -> list[Track]:
        """
        Cập nhật tracker với detections mới.

        Args:
            detections : output từ YOLODetector.detect()
            frame_shape: (height, width) của frame hiện tại

        Returns:
            list[Track] — chỉ các track đang active
        """
        self._frame_count += 1

        if len(detections) == 0:
            return []

        # Convert sang supervision Detections format
        sv_detections = sv.Detections(
            xyxy       = np.array([d.bbox_xyxy  for d in detections]),
            confidence = np.array([d.confidence for d in detections]),
            class_id   = np.array([d.class_id   for d in detections]),
        )

        # Update tracker — ByteTrack internally handles
        # low-confidence detections để tránh ID switches
        tracked = self.tracker.update_with_detections(sv_detections)

        if len(tracked) == 0:
            return []

        # Lấy class names từ detections gốc bằng IoU matching
        class_names = self._match_class_names(tracked, detections)

        tracks = []
        for i in range(len(tracked)):
            tracks.append(Track(
                track_id   = int(tracked.tracker_id[i]),
                bbox_xyxy  = tracked.xyxy[i],
                confidence = float(tracked.confidence[i]),
                class_id   = int(tracked.class_id[i]),
                class_name = class_names[i],
            ))

        return tracks

    def reset(self) -> None:
        """Reset tracker state — dùng khi chuyển sang video mới."""
        self.tracker.reset()
        self._frame_count = 0

    @staticmethod
    def _match_class_names(
        tracked: sv.Detections,
        original: list[Detection],
    ) -> list[str]:
        """Map class_id về class_name."""
        id_to_name = {d.class_id: d.class_name for d in original}
        return [
            id_to_name.get(int(cid), "unknown")
            for cid in tracked.class_id
        ]