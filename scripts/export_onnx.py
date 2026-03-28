# scripts/export_onnx.py
from __future__ import annotations
import sys
sys.path.insert(0, "/kaggle/working/traffic-ai")

import argparse
from src.optimizer.export import export_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to best.pt")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--imgsz",  default=640,   type=int)
    args = parser.parse_args()

    export_all(
        model_path = args.model,
        output_dir = args.output,
        imgsz      = args.imgsz,
    )