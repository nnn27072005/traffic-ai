# src/training/callbacks.py
from __future__ import annotations
import os, shutil, time
from pathlib import Path


class CheckpointSyncCallback:
    """
    Sync checkpoint lên Google Drive sau mỗi N epochs.
    Dùng để resume khi Kaggle session bị reset.
    """

    def __init__(
        self,
        drive_dir: str,
        sync_every: int = 5,
        keep_last_n: int = 3,
    ):
        self.drive_dir = Path(drive_dir)
        self.sync_every = sync_every
        self.keep_last_n = keep_last_n
        self.epoch = 0
        self.drive_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer) -> None:
        self.epoch += 1

        if self.epoch % self.sync_every != 0:
            return

        weights_dir = Path(trainer.save_dir) / "weights"
        if not weights_dir.exists():
            return

        # Copy last.pt và best.pt
        for fname in ["last.pt", "best.pt"]:
            src = weights_dir / fname
            if src.exists():
                dst = self.drive_dir / fname
                shutil.copy2(src, dst)

        # Copy epoch checkpoint nếu có
        epoch_ckpt = weights_dir / f"epoch{self.epoch}.pt"
        if epoch_ckpt.exists():
            shutil.copy2(epoch_ckpt, self.drive_dir / epoch_ckpt.name)

        # Cleanup old epoch checkpoints, giữ lại keep_last_n
        old_ckpts = sorted(self.drive_dir.glob("epoch*.pt"))
        for old in old_ckpts[:-self.keep_last_n]:
            old.unlink()

        print(f"  [Checkpoint] Synced epoch {self.epoch} → {self.drive_dir}")


# ── Dùng trong training script ────────────────────────────────────

from google.colab import drive as gdrive   # chỉ cần trên Colab
# Trên Kaggle: mount drive qua kaggle secrets hoặc output folder

def train_with_sync():
    from ultralytics import YOLO

    model = YOLO("yolov10s.pt")

    # Mount Google Drive (Colab)
    # gdrive.mount("/content/drive")
    # drive_save_dir = "/content/drive/MyDrive/traffic-ai-checkpoints"

    # Trên Kaggle: save vào /kaggle/working (persistent trong session)
    drive_save_dir = "/kaggle/working/traffic-ai/checkpoints"

    sync_cb = CheckpointSyncCallback(
        drive_dir   = drive_save_dir,
        sync_every  = 5,
    )

    # Register callback
    model.add_callback("on_train_epoch_end", sync_cb.on_train_epoch_end)

    model.train(
        data    = "/kaggle/working/traffic-ai/configs/fisheye8k.yaml",
        epochs  = 80,
        imgsz   = 1280,
        batch   = 8,
        project = "/kaggle/working/traffic-ai/runs/train",
        name    = "fisheye8k-yolov10s-1280",
        exist_ok = True,
    )