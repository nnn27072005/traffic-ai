# src/optimizer/export.py
from __future__ import annotations
from pathlib import Path

import torch
from ultralytics import YOLO


def export_onnx(
    model_path: str | Path,
    output_dir: str | Path,
    imgsz:      int  = 640,
    half:       bool = False,   # FP16
    simplify:   bool = True,
    opset:      int  = 17,
) -> Path:
    """
    Export YOLO .pt → ONNX.

    Args:
        model_path : path tới best.pt
        output_dir : nơi lưu file .onnx
        imgsz      : input size (phải khớp với lúc train)
        half       : True = FP16, False = FP32
        simplify   : dùng onnx-simplifier để giảm graph complexity
        opset      : ONNX opset version (17 tương thích rộng nhất)

    Returns:
        Path tới file .onnx đã export
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = "fp16" if half else "fp32"
    out_name  = f"{model_path.stem}_{precision}.onnx"
    out_path  = output_dir / out_name

    print(f"Exporting: {model_path.name} → {out_name}")
    print(f"  imgsz   : {imgsz}")
    print(f"  half    : {half} ({precision})")
    print(f"  simplify: {simplify}")

    model = YOLO(str(model_path))
    exported = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        half     = half,
        simplify = simplify,
        opset    = opset,
        dynamic  = False,
    )

    # Ultralytics export vào cùng folder với .pt
    # Move về output_dir
    exported_path = Path(exported)
    if exported_path.parent != output_dir:
        final_path = output_dir / out_name
        exported_path.rename(final_path)
        exported_path = final_path

    print(f"  Saved  : {exported_path}")
    print(f"  Size   : {exported_path.stat().st_size / 1e6:.1f} MB")

    return exported_path


def export_all(
    model_path: str | Path,
    output_dir: str | Path,
    imgsz:      int = 640,
) -> dict[str, Path]:
    """Export cả FP32 và FP16 để benchmark so sánh."""
    results = {}

    results["fp32"] = export_onnx(
        model_path, output_dir,
        imgsz=imgsz, half=False,
    )
    # FP16 chỉ support trên GPU
    if torch.cuda.is_available():
        results["fp16"] = export_onnx(
            model_path, output_dir,
            imgsz=imgsz, half=True,
        )
    else:
        print("FP16 skipped — CUDA not available")

    return results