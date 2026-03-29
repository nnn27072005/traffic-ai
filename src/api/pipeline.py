# src/api/pipeline.py
from __future__ import annotations
import time
from pathlib import Path

import cv2
import numpy as np

from src.detector.yolo_wrapper     import YOLODetector
from src.tracker.bytetrack_wrapper import ByteTracker
from src.analyzer.counter          import LineCounter
from src.analyzer.violation        import HelmetViolationAnalyzer, draw_violations
from src.api.schemas import (
    AnalysisResult, BBoxSchema, TrackSchema, ViolationSchema
)


class TrafficPipeline:
    """
    Singleton pipeline — load model 1 lần, reuse cho mọi request.
    Tránh overhead load model mỗi request (~2–3s mỗi lần).
    """

    def __init__(
        self,
        traffic_model_path: str | Path,
        helmet_model_path:  str | Path,
        device:             str   = "cuda",
        traffic_conf:       float = 0.25,
        helmet_conf:        float = 0.30,
    ):
        print("[Pipeline] Loading traffic model...")
        self.detector = YOLODetector(
            model_path = traffic_model_path,
            confidence = traffic_conf,
            device     = device,
            imgsz      = 1280,
        )

        print("[Pipeline] Loading helmet model...")
        self.analyzer = HelmetViolationAnalyzer(
            helmet_model_path = helmet_model_path,
            confidence        = helmet_conf,
            device            = device,
        )

        self.tracker = ByteTracker()
        self.counter = None   # init per-request với frame size
        self._frame_idx = 0

        print("[Pipeline] Ready.")

    def process_frame(
        self,
        frame:      np.ndarray,
        line_ratio: float = 0.5,   # counting line ở giữa frame
    ) -> AnalysisResult:

        t0 = time.perf_counter()
        self._frame_idx += 1
        h, w = frame.shape[:2]

        # Init counter nếu chưa có hoặc frame size thay đổi
        if self.counter is None:
            self.counter = LineCounter(
                line_start = (0, int(h * line_ratio)),
                line_end   = (w, int(h * line_ratio)),
            )

        # Stage 1: Detect + Track
        detections = self.detector.detect(frame)
        tracks     = self.tracker.update(detections, (h, w))
        stats      = self.counter.update(tracks)

        # Stage 2: Violation detection
        violations = self.analyzer.analyze(frame, tracks, self._frame_idx)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Build response
        track_schemas = [
            TrackSchema(
                track_id   = t.track_id,
                class_name = t.class_name,
                confidence = round(t.confidence, 3),
                bbox       = BBoxSchema(
                    x1=float(t.bbox_xyxy[0]), y1=float(t.bbox_xyxy[1]),
                    x2=float(t.bbox_xyxy[2]), y2=float(t.bbox_xyxy[3]),
                ),
            )
            for t in tracks
        ]

        violation_schemas = [
            ViolationSchema(
                track_id   = v.track_id,
                violations = v.violations,
                confidence = round(v.confidence, 3),
                bbox       = BBoxSchema(
                    x1=float(v.bbox_xyxy[0]), y1=float(v.bbox_xyxy[1]),
                    x2=float(v.bbox_xyxy[2]), y2=float(v.bbox_xyxy[3]),
                ),
            )
            for v in violations
        ]

        return AnalysisResult(
            frame_id      = self._frame_idx,
            vehicle_count = len(tracks),
            tracks        = track_schemas,
            violations    = violation_schemas,
            count_in      = stats.total_in,
            count_out     = stats.total_out,
            latency_ms    = round(latency_ms, 2),
        )

    def reset(self) -> None:
        """Reset tracker và counter — dùng khi chuyển video source."""
        self.tracker.reset()
        self.counter = None
        self._frame_idx = 0
        self.analyzer._last_alert.clear()