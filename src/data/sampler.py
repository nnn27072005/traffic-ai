# src/data/sampler.py
from __future__ import annotations
import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np


def oversample_minority_classes(
    src_label_dir: str | Path,
    src_image_dir: str | Path,
    dst_label_dir: str | Path,
    dst_image_dir: str | Path,
    target_classes: list[int],   # class ids cần oversample
    multiplier: int = 3,         # copy bao nhiêu lần
    min_instances: int = 2,      # ảnh phải có ít nhất N instances của class đó
) -> None:
    """
    Copy thêm ảnh chứa minority classes vào training set.
    Chạy 1 lần trước khi train, output vào folder riêng.
    """
    src_lbl = Path(src_label_dir)
    src_img = Path(src_image_dir)
    dst_lbl = Path(dst_label_dir)
    dst_img = Path(dst_image_dir)

    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    # First: copy toàn bộ original data
    for f in src_lbl.glob("*.txt"):
        shutil.copy2(f, dst_lbl / f.name)
    for f in src_img.glob("*"):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(f, dst_img / f.name)

    # Find ảnh có minority class
    minority_files: list[Path] = []
    for lbl_file in src_lbl.glob("*.txt"):
        lines = lbl_file.read_text().strip().splitlines()
        cls_counts = defaultdict(int)
        for line in lines:
            if line:
                cls_counts[int(line.split()[0])] += 1

        # Chỉ lấy ảnh có đủ instances của target class
        for tc in target_classes:
            if cls_counts[tc] >= min_instances:
                minority_files.append(lbl_file)
                break

    print(f"Found {len(minority_files)} images with minority classes")

    # Copy thêm multiplier lần
    img_extensions = [".jpg", ".jpeg", ".png"]
    copied = 0

    for lbl_file in minority_files:
        # Tìm image file tương ứng
        img_file = None
        for ext in img_extensions:
            candidate = src_img / lbl_file.with_suffix(ext).name
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        for i in range(1, multiplier):
            new_stem = f"{lbl_file.stem}_aug{i}"
            shutil.copy2(lbl_file, dst_lbl / f"{new_stem}.txt")
            shutil.copy2(img_file, dst_img / f"{new_stem}{img_file.suffix}")
            copied += 1

    print(f"Oversampled: added {copied} extra images")
    print(f"New training set size: {len(list(dst_img.glob('*.*')))}")