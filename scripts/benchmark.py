# scripts/benchmark.py
from __future__ import annotations
import sys
sys.path.insert(0, "/kaggle/working/traffic-ai")

import argparse
import json
from pathlib import Path
from src.optimizer.benchmark import run_full_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, help="Path to best.pt")
    parser.add_argument("--onnx-dir", required=True, help="ONNX files directory")
    parser.add_argument("--imgsz",    default=640,   type=int)
    parser.add_argument("--runs",     default=200,   type=int)
    parser.add_argument("--save",     default=None,  help="Save results as JSON")
    args = parser.parse_args()

    results = run_full_benchmark(
        pt_path  = args.model,
        onnx_dir = args.onnx_dir,
        imgsz    = args.imgsz,
        n_runs   = args.runs,
    )

    if args.save:
        out = [
            {
                "name":    r.name,
                "mean_ms": r.mean_ms,
                "p95_ms":  r.p95_ms,
                "p99_ms":  r.p99_ms,
                "fps":     r.fps,
            }
            for r in results
        ]
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {args.save}")