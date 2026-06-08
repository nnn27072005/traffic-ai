# scripts/train_detector.py
from __future__ import annotations
import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.training.callbacks import CheckpointSyncCallback


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_class_weights_callback(cls_weights: list[float]):
    """
    Patch cls loss weights sau khi model được khởi tạo.
    Ultralytics lưu class weights trong model.model.criterion.cls_pw
    (với detection loss dùng BCEWithLogitsLoss).
    """
    weights_tensor = torch.tensor(cls_weights, dtype=torch.float32)

    def on_train_start(trainer):
        try:
            # Ultralytics v8/v10: criterion được init trong trainer
            criterion = trainer.model.criterion
            if hasattr(criterion, "cls_pw"):
                criterion.cls_pw = weights_tensor.to(trainer.device)
                print(f"  [ClassWeights] Applied: {cls_weights}")
            else:
                # Fallback: patch hyp dict
                trainer.model.hyp["cls_pw"] = weights_tensor.to(trainer.device)
                print(f"  [ClassWeights] Applied via hyp: {cls_weights}")
        except Exception as e:
            print(f"  [ClassWeights] Warning — could not apply: {e}")
            print(f"  Training continues with uniform weights.")

    return on_train_start


def train(config_path: str, resume: bool = False):
    CFG = load_config(config_path)

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_dir = Path(CFG["project"]) / CFG["name"]

    # Resume logic
    if resume:
        last_ckpt = run_dir / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
            return

    model = YOLO(CFG["model"])

    # Register callbacks
    sync_cb = CheckpointSyncCallback(
        drive_dir  = "/kaggle/working/traffic-ai/checkpoints",
        sync_every = 5,
    )
    model.add_callback("on_train_epoch_end", sync_cb.on_train_epoch_end)

    if "cls_weights" in CFG:
        model.add_callback(
            "on_train_start",
            apply_class_weights_callback(CFG["cls_weights"]),
        )

    # Loại bỏ keys không phải Ultralytics params trước khi pass vào train()
    EXCLUDED_KEYS = {"model", "cls_weights"}
    train_args = {k: v for k, v in CFG.items() if k not in EXCLUDED_KEYS}

    results = model.train(**train_args, exist_ok=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_detector.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        config_path = f"/kaggle/working/traffic-ai/{args.config}",
        resume      = args.resume,
    )