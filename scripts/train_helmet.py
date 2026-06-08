# scripts/train_helmet.py
from __future__ import annotations
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from src.training.callbacks import CheckpointSyncCallback


def train(config_path: str, resume: bool = False):
    with open(config_path) as f:
        CFG = yaml.safe_load(f)

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    run_dir = Path(CFG["project"]) / CFG["name"]

    if resume:
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            print(f"Resuming from {last}")
            YOLO(str(last)).train(resume=True)
            return
        print("No checkpoint found, starting fresh.")

    model = YOLO(CFG["model"])

    sync_cb = CheckpointSyncCallback(
        drive_dir  = "/kaggle/working/traffic-ai/checkpoints/helmet",
        sync_every = 5,
    )
    model.add_callback("on_train_epoch_end", sync_cb.on_train_epoch_end)

    EXCLUDED = {"model"}
    train_args = {k: v for k, v in CFG.items() if k not in EXCLUDED}

    results = model.train(**train_args, exist_ok=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_helmet.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(
        config_path = f"/kaggle/working/traffic-ai/{args.config}",
        resume      = args.resume,
    )