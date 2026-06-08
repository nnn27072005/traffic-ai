# src/detector/yolo_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """Một detection result chuẩn hóa, độc lập với Ultralytics."""
    bbox_xyxy: np.ndarray   # shape (4,) — x1 y1 x2 y2 pixel coords
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    Thin wrapper quanh Ultralytics YOLO.
    Chuẩn hóa output thành list[Detection] để
    các module khác không phụ thuộc trực tiếp vào Ultralytics.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda",
        imgsz: int = 1280,
    ):
        self.model      = YOLO(str(model_path))
        self.confidence = confidence
        self.iou        = iou
        self.device     = device
        self.imgsz      = imgsz
        self.class_names: dict[int, str] = self.model.names

        print(f"[YOLODetector] Loaded: {Path(model_path).name}")
        print(f"[YOLODetector] Classes: {self.class_names}")
        print(f"[YOLODetector] Device : {device}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Chạy inference trên 1 frame.
        Returns list[Detection] sorted by confidence descending.
        """
        results = self.model(
            frame,
            conf    = self.confidence,
            iou     = self.iou,
            imgsz   = self.imgsz,
            device  = self.device,
            verbose = False,
        )[0]

        detections = []
        for box in results.boxes:
            detections.append(Detection(
                bbox_xyxy  = box.xyxy[0].cpu().numpy(),
                confidence = float(box.conf[0]),
                class_id   = int(box.cls[0]),
                class_name = self.class_names[int(box.cls[0])],
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Batch inference — dùng khi có nhiều camera."""
        results = self.model(
            frames,
            conf    = self.confidence,
            iou     = self.iou,
            imgsz   = self.imgsz,
            device  = self.device,
            verbose = False,
        )
        return [
            [
                Detection(
                    bbox_xyxy  = box.xyxy[0].cpu().numpy(),
                    confidence = float(box.conf[0]),
                    class_id   = int(box.cls[0]),
                    class_name = self.class_names[int(box.cls[0])],
                )
                for box in r.boxes
            ]
            for r in results
        ]