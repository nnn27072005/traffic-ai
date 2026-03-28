# src/optimizer/benchmark.py
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO


@dataclass
class BenchmarkResult:
    name:      str
    mean_ms:   float
    std_ms:    float
    p50_ms:    float
    p95_ms:    float
    p99_ms:    float
    fps:       float
    n_runs:    int

    def __str__(self) -> str:
        return (
            f"{self.name:<22} | "
            f"mean {self.mean_ms:>6.1f}ms | "
            f"p95 {self.p95_ms:>6.1f}ms | "
            f"p99 {self.p99_ms:>6.1f}ms | "
            f"{self.fps:>6.1f} FPS"
        )


def _make_dummy(imgsz: int, half: bool = False) -> np.ndarray:
    dtype = np.float16 if half else np.float32
    return np.random.randn(1, 3, imgsz, imgsz).astype(dtype)


def benchmark_pytorch(
    model_path: str | Path,
    imgsz:      int = 640,
    n_runs:     int = 200,
    device:     str = "cuda",
) -> BenchmarkResult:
    """Benchmark PyTorch .pt model."""
    model = YOLO(str(model_path))

    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model.model(dummy)

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return BenchmarkResult(
        name    = f"PyTorch FP32",
        mean_ms = float(np.mean(times)),
        std_ms  = float(np.std(times)),
        p50_ms  = float(np.percentile(times, 50)),
        p95_ms  = float(np.percentile(times, 95)),
        p99_ms  = float(np.percentile(times, 99)),
        fps     = 1000 / float(np.mean(times)),
        n_runs  = n_runs,
    )


def benchmark_onnx(
    model_path: str | Path,
    imgsz:      int  = 640,
    half:       bool = False,
    n_runs:     int  = 200,
) -> BenchmarkResult:
    """Benchmark ONNX Runtime — CUDAExecutionProvider."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess  = ort.InferenceSession(
        str(model_path),
        sess_options = sess_opts,
        providers    = providers,
    )
    input_name = sess.get_inputs()[0].name
    dummy      = _make_dummy(imgsz, half)

    # Warmup
    for _ in range(10):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    times      = np.array(times)
    precision  = "FP16" if half else "FP32"
    model_name = Path(model_path).stem

    return BenchmarkResult(
        name    = f"ONNX {precision}",
        mean_ms = float(np.mean(times)),
        std_ms  = float(np.std(times)),
        p50_ms  = float(np.percentile(times, 50)),
        p95_ms  = float(np.percentile(times, 95)),
        p99_ms  = float(np.percentile(times, 99)),
        fps     = 1000 / float(np.mean(times)),
        n_runs  = n_runs,
    )


def run_full_benchmark(
    pt_path:   str | Path,
    onnx_dir:  str | Path,
    imgsz:     int = 640,
    n_runs:    int = 200,
) -> list[BenchmarkResult]:
    """
    Chạy benchmark đầy đủ: PyTorch vs ONNX FP32 vs ONNX FP16.
    Đây là table sẽ paste vào README và CV.
    """
    results = []
    onnx_dir = Path(onnx_dir)

    print(f"\nBenchmarking on: {torch.cuda.get_device_name(0)}")
    print(f"Input size: {imgsz}×{imgsz}")
    print(f"Runs: {n_runs}")
    print("=" * 65)

    # PyTorch baseline
    r = benchmark_pytorch(pt_path, imgsz=imgsz, n_runs=n_runs)
    results.append(r)
    print(r)

    # ONNX FP32
    fp32_path = onnx_dir / f"{Path(pt_path).stem}_fp32.onnx"
    if fp32_path.exists():
        r = benchmark_onnx(fp32_path, imgsz=imgsz, half=False, n_runs=n_runs)
        results.append(r)
        print(r)

    # ONNX FP16
    fp16_path = onnx_dir / f"{Path(pt_path).stem}_fp16.onnx"
    if fp16_path.exists():
        r = benchmark_onnx(fp16_path, imgsz=imgsz, half=True, n_runs=n_runs)
        results.append(r)
        print(r)

    print("=" * 65)

    # Speedup summary
    if len(results) >= 2:
        baseline = results[0].mean_ms
        print("\nSpeedup vs PyTorch FP32:")
        for r in results[1:]:
            speedup = baseline / r.mean_ms
            print(f"  {r.name:<20}: {speedup:.2f}×")

    return results