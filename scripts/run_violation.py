# scripts/run_violation.py
from __future__ import annotations
import argparse
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from src.detector.yolo_wrapper  import YOLODetector
from src.tracker.bytetrack_wrapper import ByteTracker
from src.analyzer.counter       import LineCounter
from src.analyzer.violation     import HelmetViolationAnalyzer, draw_violations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("violation-cli")

def run_violation(
    source:       str,
    traffic_path: str,
    helmet_path:  str,
    output:       str,
    line_y:       float = 0.5,
    confidence:   float = 0.25,
    device:       str   = "cuda",
    show:         bool  = False,
) -> None:
    # 1. Init
    detector = YOLODetector(model_path=traffic_path, confidence=confidence, device=device)
    analyzer = HelmetViolationAnalyzer(helmet_model_path=helmet_path, device=device)
    tracker  = ByteTracker()
    
    cap = cv2.VideoCapture(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    counter = LineCounter(line_start=(0, int(h * line_y)), line_end=(w, int(h * line_y)))
    
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    box_ann = sv.BoxAnnotator(thickness=2)
    line_ann = sv.LineZoneAnnotator()
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # Detect + Track
        detections = detector.detect(frame)
        tracks = tracker.update(detections, (h, w))
        counter.update(tracks)
        
        # Violations
        violations = analyzer.analyze(frame, tracks, frame_idx)
        
        # Annotate
        if tracks:
            sv_det = sv.Detections(
                xyxy=np.array([t.bbox_xyxy for t in tracks]),
                class_id=np.array([t.class_id for t in tracks]),
                tracker_id=np.array([t.track_id for t in tracks])
            )
            frame = box_ann.annotate(frame, sv_det)
        
        frame = draw_violations(frame, violations)
        frame = line_ann.annotate(frame, counter.line)
        
        writer.write(frame)
        if show:
            cv2.imshow("Violation Detection", frame)
            if cv2.waitKey(1) == ord('q'): break
            
        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--traffic", required=True)
    parser.add_argument("--helmet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_violation(args.source, args.traffic, args.helmet, args.output, device=args.device)
