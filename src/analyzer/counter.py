# src/analyzer/counter.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Literal

import numpy as np
import supervision as sv

from src.tracker.bytetrack_wrapper import Track


@dataclass
class CounterStats:
    """Thống kê tại một thời điểm."""
    total_in:    int
    total_out:   int
    per_class_in:  dict[str, int]
    per_class_out: dict[str, int]
    active_tracks: int

    @property
    def total(self) -> int:
        return self.total_in + self.total_out


class LineCounter:
    """
    Đếm xe đi qua một đường thẳng (counting line).

    Cách hoạt động:
    - Mỗi track được theo dõi vị trí trung tâm bbox qua các frames
    - Khi centroid chuyển từ một phía sang phía kia của line → count
    - "in" = đi từ trên xuống (hoặc trái sang phải tùy orientation)
    - "out" = chiều ngược lại

    Lý do dùng centroid thay vì bbox edge:
    - Bbox edge bị ảnh hưởng bởi partial occlusion
    - Centroid ổn định hơn khi xe bị che khuất một phần
    """

    def __init__(
        self,
        line_start: tuple[int, int],
        line_end:   tuple[int, int],
    ):
        self.line = sv.LineZone(
            start = sv.Point(*line_start),
            end   = sv.Point(*line_end),
        )

        # Track lịch sử crossing để tránh đếm trùng
        self._counted_ids: set[int] = set()
        self._in_count:  int = 0
        self._out_count: int = 0
        self._per_class_in:  dict[str, int] = defaultdict(int)
        self._per_class_out: dict[str, int] = defaultdict(int)

        # Map track_id → class_name (để count per class khi cross)
        self._id_to_class: dict[int, str] = {}

    def update(self, tracks: list[Track]) -> CounterStats:
        """
        Cập nhật counter với tracks mới.
        Gọi mỗi frame sau ByteTracker.update().
        """
        if not tracks:
            return self.get_stats(active_tracks=0)

        # Convert tracks sang sv.Detections để dùng LineZone
        sv_detections = sv.Detections(
            xyxy       = np.array([t.bbox_xyxy  for t in tracks]),
            confidence = np.array([t.confidence for t in tracks]),
            class_id   = np.array([t.class_id   for t in tracks]),
            tracker_id = np.array([t.track_id   for t in tracks]),
        )

        # Update class map
        for t in tracks:
            self._id_to_class[t.track_id] = t.class_name

        # LineZone.trigger returns (crossed_in, crossed_out) boolean arrays
        crossed_in, crossed_out = self.line.trigger(sv_detections)

        for i, track in enumerate(tracks):
            tid = track.track_id
            cls = track.class_name

            if crossed_in[i] and tid not in self._counted_ids:
                self._in_count += 1
                self._per_class_in[cls] += 1
                self._counted_ids.add(tid)

            elif crossed_out[i] and tid not in self._counted_ids:
                self._out_count += 1
                self._per_class_out[cls] += 1
                self._counted_ids.add(tid)

        return self.get_stats(active_tracks=len(tracks))

    def get_stats(self, active_tracks: int = 0) -> CounterStats:
        return CounterStats(
            total_in       = self._in_count,
            total_out      = self._out_count,
            per_class_in   = dict(self._per_class_in),
            per_class_out  = dict(self._per_class_out),
            active_tracks  = active_tracks,
        )

    def reset(self) -> None:
        self._counted_ids.clear()
        self._in_count  = 0
        self._out_count = 0
        self._per_class_in.clear()
        self._per_class_out.clear()
        self._id_to_class.clear()