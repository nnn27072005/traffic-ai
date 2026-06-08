# src/api/pipeline.py
from __future__ import annotations
import os
import time, logging, asyncio
from pathlib import Path

import cv2
import numpy as np

from src.detector.yolo_wrapper     import YOLODetector
from src.tracker.bytetrack_wrapper import ByteTracker
from src.analyzer.counter          import LineCounter
from src.analyzer.violation        import BIKE_CLASS_NAMES, HelmetViolationAnalyzer, draw_violations
from src.api.schemas import (
    AnalysisResult, BBoxSchema, TrackSchema, ViolationSchema
)

logger = logging.getLogger("traffic-api.pipeline")

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
        violation_queue:    asyncio.Queue | None = None,
    ):
        logger.info("Loading traffic model...")
        self.detector = YOLODetector(
            model_path = traffic_model_path,
            confidence = traffic_conf,
            device     = device,
            imgsz      = 640,
        )

        logger.info("Loading helmet model...")
        self.analyzer = HelmetViolationAnalyzer(
            helmet_model_path = helmet_model_path,
            confidence        = helmet_conf,
            device            = device,
        )

        self.tracker = ByteTracker()
        self.counter = None   # init per-request với frame size
        self._frame_idx = 0
        self.violation_queue = violation_queue
        self.violation_interval = max(1, int(os.getenv("VIOLATION_ANALYSIS_INTERVAL", "3")))

        logger.info("Pipeline initialized and ready.")

    def process_frame(
        self,
        frame:      np.ndarray,
        line_ratio: float = 0.5,   # counting line ở giữa frame
        session_id: int | None = None,
        camera_id:  int | None = None,
        save_evidence: bool = False,
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
        violations = []
        if self.violation_queue is not None:
            # Async mode: push to queue, don't block
            # We filter for bikes here to avoid pushing unnecessary frames
            bike_tracks = [t for t in tracks if t.class_name in BIKE_CLASS_NAMES]
            if bike_tracks and self._frame_idx % self.violation_interval == 0:
                try:
                    evidence_frame = frame.copy() if save_evidence else frame
                    self.violation_queue.put_nowait({
                        "frame": evidence_frame,
                        "tracks": bike_tracks,
                        "frame_idx": self._frame_idx,
                        "session_id": session_id,
                        "save_evidence": save_evidence,
                    })
                except asyncio.QueueFull:
                    # Skip if queue is full to prevent pipeline blockage
                    pass
        else:
            # Sync mode: block (legacy/fallback)
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

        result = AnalysisResult(
            frame_id      = self._frame_idx,
            vehicle_count = len(tracks),
            tracks        = track_schemas,
            violations    = violation_schemas,
            count_in      = stats.total_in,
            count_out     = stats.total_out,
            latency_ms    = round(latency_ms, 2),
        )

        return result


    def render(
        self,
        frame:      np.ndarray,
        result:     AnalysisResult,
        mode:       str   = "preview",
        line_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Multi-mode renderer:
          - 'preview': Low-res (720p), lightweight annotations, high throughput.
          - 'export':  Full-res, detailed annotations, high quality.
        """
        import supervision as sv
        from src.analyzer.violation import draw_violations, ViolationEvent

        h_orig, w_orig = frame.shape[:2]
        
        # 1. Handle mode-specific scaling
        if mode == "preview" and w_orig > 1280:
            scale = 1280 / w_orig
            target_w = 1280
            target_h = int(h_orig * scale)
            canvas = cv2.resize(frame, (target_w, target_h))
            h, w = target_h, target_w
        else:
            scale = 1.0
            canvas = frame.copy()
            h, w = h_orig, w_orig

        # 2. Prepare detections for supervision
        if result.tracks:
            xyxy = np.array([[
                t.bbox.x1 * scale, t.bbox.y1 * scale,
                t.bbox.x2 * scale, t.bbox.y2 * scale
            ] for t in result.tracks])
            
            sv_det = sv.Detections(
                xyxy=xyxy,
                tracker_id=np.array([t.track_id for t in result.tracks]),
                class_id=np.array([0] * len(result.tracks)) # Dummy class ID
            )
            
            # Preview uses thinner lines
            thickness = 1 if mode == "preview" else 2
            box_ann = sv.BoxAnnotator(thickness=thickness)
            
            if mode == "export":
                label_ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
                labels = [f"#{t.track_id} {t.class_name}" for t in result.tracks]
                canvas = box_ann.annotate(canvas, sv_det)
                canvas = label_ann.annotate(canvas, sv_det, labels=labels)
            else:
                # Preview just draws boxes to save CPU
                canvas = box_ann.annotate(canvas, sv_det)

        # 3. Draw violations
        if result.violations:
            events = [
                ViolationEvent(
                    track_id=v.track_id,
                    bbox_xyxy=np.array([
                        v.bbox.x1 * scale, v.bbox.y1 * scale,
                        v.bbox.x2 * scale, v.bbox.y2 * scale
                    ]),
                    violations=v.violations,
                    confidence=v.confidence,
                    frame_idx=result.frame_id
                ) for v in result.violations
            ]
            canvas = draw_violations(canvas, events)
        
        # 4. Draw counting line & HUD
        line_y = int(h * line_ratio)
        cv2.line(canvas, (0, line_y), (w, line_y), (0, 255, 255), 2 if mode == "export" else 1)
        
        hud_text = f"IN: {result.count_in} OUT: {result.count_out}"
        if mode == "export":
            hud_text += f" | FPS: {round(1000/result.latency_ms, 1) if result.latency_ms > 0 else 0}"
            
        cv2.putText(canvas, hud_text, (10, 30 if mode == "export" else 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1 if mode == "export" else 0.6, 
                    (0, 255, 255), 2 if mode == "export" else 1)

        return canvas


    def reset(self) -> None:
        """Reset tracker và counter — dùng khi chuyển video source."""
        self.tracker.reset()
        self.counter = None
        self._frame_idx = 0
        self.analyzer._last_alert.clear()
