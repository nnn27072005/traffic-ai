# src/optimizer/benchmark.py — sửa hàm benchmark_pytorch

def benchmark_pytorch(
    model_path: str | Path,
    imgsz:      int = 640,
    n_runs:     int = 200,
    device:     str = "cuda",
) -> BenchmarkResult:
    """Benchmark PyTorch .pt model."""
    model = YOLO(str(model_path))
    
    # ← Fix: chuyển model sang đúng device trước khi inference
    model.model = model.model.to(device)
    model.model.eval()

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
        name    = "PyTorch FP32",
        mean_ms = float(np.mean(times)),
        std_ms  = float(np.std(times)),
        p50_ms  = float(np.percentile(times, 50)),
        p95_ms  = float(np.percentile(times, 95)),
        p99_ms  = float(np.percentile(times, 99)),
        fps     = 1000 / float(np.mean(times)),
        n_runs  = n_runs,
    )