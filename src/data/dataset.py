# src/data/dataset.py

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ── Data structures ──────────────────────────────────────────────

@dataclass
class BoundingBox:
    """YOLO format: cx, cy, w, h — tất cả normalized [0, 1]."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / (self.h + 1e-6)

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        x1 = int((self.cx - self.w / 2) * img_w)
        y1 = int((self.cy - self.h / 2) * img_h)
        x2 = int((self.cx + self.w / 2) * img_w)
        y2 = int((self.cy + self.h / 2) * img_h)
        return x1, y1, x2, y2


@dataclass
class Sample:
    image_path: Path
    label_path: Optional[Path]
    boxes: list[BoundingBox] = field(default_factory=list)

    def load_image(self) -> np.ndarray:
        img = cv2.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {self.image_path}")
        return img


# ── Dataset class ─────────────────────────────────────────────────

class FishEye8KDataset:
    """
    Dataset loader cho FishEye8K.
    Hỗ trợ cả YOLO directory format và custom path.
    """

    CLASS_NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        skip_empty: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / split / "images"
        self.lbl_dir = self.root / split / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")

        self.samples = self._load_samples(skip_empty)
        print(f"[FishEye8K/{split}] Loaded {len(self.samples)} samples.")

    def _load_samples(self, skip_empty: bool) -> list[Sample]:
        samples = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue

            lbl_path = self.lbl_dir / img_path.with_suffix(".txt").name
            boxes = self._parse_label(lbl_path) if lbl_path.exists() else []

            if skip_empty and len(boxes) == 0:
                continue

            samples.append(Sample(img_path, lbl_path if lbl_path.exists() else None, boxes))

        return samples

    @staticmethod
    def _parse_label(lbl_path: Path) -> list[BoundingBox]:
        boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                continue  # skip malformed lines
            boxes.append(BoundingBox(
                class_id=int(parts[0]),
                cx=float(parts[1]),
                cy=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
            ))
        return boxes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def get_class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in self.CLASS_NAMES}
        for sample in self.samples:
            for box in sample.boxes:
                if box.class_id < len(self.CLASS_NAMES):
                    counts[self.CLASS_NAMES[box.class_id]] += 1
        return counts

    def get_box_stats(self) -> dict[str, np.ndarray]:
        areas, ratios, cls_ids = [], [], []
        for sample in self.samples:
            for box in sample.boxes:
                areas.append(box.area)
                ratios.append(box.aspect_ratio)
                cls_ids.append(box.class_id)
        return {
            "areas":   np.array(areas),
            "ratios":  np.array(ratios),
            "cls_ids": np.array(cls_ids),
        }