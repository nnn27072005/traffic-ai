# scripts/train_detector.py
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import torch
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────

import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config("/kaggle/working/traffic-ai/configs/train_detector.yaml")


# ── Class weight calculator ───────────────────────────────────────

def compute_class_weights(dataset_yaml: str) -> list[float]:
    """
    Tính class weights từ distribution để compensate imbalance.
    Formula: w_i = total / (n_classes * count_i), normalized to mean=1.
    """
    import yaml
    from pathlib import Path
    import numpy as np

    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["path"])
    label_dir = data_root / "train" / "labels"

    counts = np.zeros(cfg["nc"])
    for lbl_file in label_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            if line:
                cls = int(line.split()[0])
                if cls < cfg["nc"]:
                    counts[cls] += 1

    # Inverse frequency, normalized
    weights = counts.sum() / (cfg["nc"] * counts + 1e-6)
    weights = weights / weights.mean()

    print("\nClass weights:")
    for i, (name, w) in enumerate(zip(cfg["names"].values(), weights)):
        print(f"  {name:<12}: {counts[i]:>6.0f} instances → weight {w:.3f}")

    return weights.tolist()


# ── Checkpoint resume helper ──────────────────────────────────────

def get_resume_path(run_dir: str) -> str | None:
    """Tìm checkpoint mới nhất để resume nếu runtime bị reset."""
    run_path = Path(run_dir)
    last_ckpt = run_path / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Found checkpoint: {last_ckpt}")
        return str(last_ckpt)
    return None


# ── Main training function ────────────────────────────────────────

def train(resume: bool = False):
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    run_dir = Path(CFG["project"]) / CFG["name"]

    # Resume logic
    if resume:
        ckpt = get_resume_path(str(run_dir))
        if ckpt:
            model = YOLO(ckpt)
            model.train(resume=True)
            return

    # Fresh training
    model = YOLO(CFG["model"])

    # Compute class weights từ actual distribution
    class_weights = compute_class_weights(CFG["data"])
    # YOLOv10 không có cls_pw per-class trực tiếp,
    # nhưng ta dùng để inform augmentation strategy
    # (xem bước 2.3 — oversampling)

    results = model.train(
        data         = CFG["data"],
        epochs       = CFG["epochs"],
        imgsz        = CFG["imgsz"],
        batch        = CFG["batch"],
        workers      = CFG["workers"],

        optimizer    = CFG["optimizer"],
        lr0          = CFG["lr0"],
        lrf          = CFG["lrf"],
        momentum     = CFG["momentum"],
        weight_decay = CFG["weight_decay"],
        warmup_epochs     = CFG["warmup_epochs"],
        warmup_momentum   = CFG["warmup_momentum"],

        hsv_h        = CFG["hsv_h"],
        hsv_s        = CFG["hsv_s"],
        hsv_v        = CFG["hsv_v"],
        degrees      = CFG["degrees"],
        translate    = CFG["translate"],
        scale        = CFG["scale"],
        shear        = CFG["shear"],
        perspective  = CFG["perspective"],
        flipud       = CFG["flipud"],
        fliplr       = CFG["fliplr"],
        mosaic       = CFG["mosaic"],
        mixup        = CFG["mixup"],
        copy_paste   = CFG["copy_paste"],

        box          = CFG["box"],
        cls          = CFG["cls"],
        dfl          = CFG["dfl"],

        dropout      = CFG["dropout"],
        label_smoothing = CFG["label_smoothing"],

        project      = CFG["project"],
        name         = CFG["name"],
        save_period  = CFG["save_period"],
        plots        = CFG["plots"],

        # Tránh mất checkpoint khi Kaggle session hết
        save         = True,
        exist_ok     = True,
    )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)