# Traffic Safety Monitoring System

> End-to-end traffic analytics: NMS-free detection · ByteTrack MOT · Helmet violation detection · REST API

---

## Demo

<!-- Thay bằng GIF hoặc link video sau khi có demo video -->
![Demo](assets/qualitative_results.png)

---

## Results

### Traffic Detection — FishEye8K

| Metric | Value |
|--------|-------|
| mAP@50 | **0.534** |
| mAP@50-95 | **0.288** |
| Precision | **0.651** |
| Recall | **0.470** |

| Class | AP@50 |
|-------|-------|
| Bus | 0.705  |
| Bike | 0.574  |
| Car | 0.742  |
| Pedestrian | 0.241  |
| Truck | 0.410  |

### Helmet Violation Detection

| Metric | Value |
|--------|-------|
| mAP@50 | 0.969 |
| Precision | 0.791 |
| Recall | 0.928 |

| Class | AP@50 |
|-------|-------|
| Plate | 0.603 |
| WithHelmet | 0.779 |
| WithoutHelmet | 0.658 |

### Inference Benchmark — Tesla T4

| Model | Format | Latency | FPS |
|-------|--------|---------|-----|
| Traffic (1280×1280) | PyTorch GPU | 41.7ms | 24.0 |
| Traffic (1280×1280) | ONNX CPU | 592.3ms | 1.7 |
| Helmet (640×640) | PyTorch GPU | 13.1ms | 76.2 |
| Helmet (640×640) | ONNX CPU | 140.1ms | 7.1 |

**GPU is 14.2× faster than CPU** for traffic model inference.

---

## Architecture
```
Input Frame
    │
    ▼
┌─────────────────────┐
│  YOLOv10s Detector  │  NMS-free · imgsz=1280
│  (FishEye8K)        │  5 classes: Bus/Bike/Car/Ped/Truck
└──────────┬──────────┘
           │ Detections
           ▼
┌─────────────────────┐
│    ByteTrack MOT    │  Low-conf detection reuse
│                     │  Lost track buffer = 30 frames
└──────┬──────┬───────┘
       │      │
       ▼      ▼
┌──────────┐  ┌──────────────────────┐
│  Line    │  │  Helmet Violation    │
│  Counter │  │  Analyzer (cascade)  │
│  IN/OUT  │  │  Crop → YOLOv10s     │
└──────────┘  │  WithHelmet/Without  │
              └──────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   FastAPI     │
              │  REST API     │
              └───────────────┘
```

---

## Key Technical Decisions

### 1. NMS-free Detection (YOLOv10)
YOLOv10 sử dụng dual-label assignment trong training — one-to-one branch cho inference, one-to-many branch chỉ dùng khi train. Kết quả: **zero post-processing overhead**, mỗi frame tiết kiệm 3–8ms so với YOLOv8/v9 với NMS.

### 2. ByteTrack thay vì DeepSORT
ByteTrack tái sử dụng low-confidence detections (score 0.1–0.5) để maintain tracklet khi xe bị che khuất — quan trọng với traffic Việt Nam có mật độ cao. Không cần ReID model riêng → latency thấp hơn ~3×.

### 3. Cascade Architecture cho Helmet Detection
Thay vì train 1 model lớn xử lý toàn cảnh, dùng 2 stage:
- Stage 1: Detect motorcycle bbox (traffic model)
- Stage 2: Crop + expand → classify helmet status

Lý do: helmet model cần thấy rõ người trên xe → crop giúp tăng effective resolution.

### 4. ONNX Export cho Deployment Flexibility
Export sang ONNX cho phép deploy trên bất kỳ hardware nào không cần PyTorch:
- FP32: 29.7MB, portable, CPU-compatible
- FP16: 14.9MB, 50% nhỏ hơn, GPU-optimized

---

## Dataset

| Dataset | Images | Instances | Classes |
|---------|--------|-----------|---------|
| FishEye8K | 8,000 | 112,213 | 5 (Bus/Bike/Car/Ped/Truck) |
| HelmetViolations | 1,004 | 1,378 | 3 (Plate/WithHelmet/WithoutHelmet) |

**FishEye8K challenges:**
- Fish-eye geometric distortion
- Dense scenes: median 20+ objects/frame
- Class imbalance: Bike (62K) vs Bus (2K) instances

---

## Project Structure
```
traffic-ai/
├── configs/              # Dataset & training configs
├── src/
│   ├── data/             # Dataset loader, sampler
│   ├── detector/         # YOLOv10 wrapper
│   ├── tracker/          # ByteTrack wrapper
│   ├── analyzer/         # Counter + violation detection
│   ├── optimizer/        # ONNX export + benchmark
│   ├── training/         # Training callbacks
│   └── api/              # FastAPI application
├── scripts/              # Training & inference CLI
├── notebooks/            # EDA notebooks
├── tests/                # Unit tests
└── docker/               # Dockerfile + compose
```

---

## Installation
```bash
git clone https://github.com/nnn27072005/traffic-ai
cd traffic-ai
pip install -r requirements.txt
```

## Usage

### Run tracking pipeline
```bash
python scripts/run_tracking.py \
    --source video.mp4 \
    --model runs/train/fisheye8k-yolov10s-1280/weights/best.pt \
    --output output.mp4
```

### Run violation detection
```bash
python scripts/run_violation.py \
    --source video.mp4 \
    --traffic runs/train/fisheye8k-yolov10s-1280/weights/best.pt \
    --helmet runs/train/helmet-yolov10s-640/weights/best.pt \
    --output output_violation.mp4
```

### Start API server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### API endpoints
```
GET  /health                    — Model status
POST /analyze/image             — Analyze image, return JSON
POST /analyze/image/annotated   — Analyze image, return annotated JPEG
POST /reset                     — Reset tracker state
GET  /benchmark                 — Inference benchmark results
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv10s (Ultralytics) |
| Tracking | ByteTrack (Supervision) |
| Export | ONNX opset 17 |
| API | FastAPI + Uvicorn |
| Containerization | Docker |

---

## Training
```bash
# Train traffic detector
python scripts/train_detector.py --config configs/train_detector.yaml

# Train helmet classifier
python scripts/train_helmet.py --config configs/train_helmet.yaml

# Export to ONNX
python scripts/export_onnx.py \
    --model runs/train/fisheye8k-yolov10s-1280/weights/best.pt \
    --output runs/export/traffic \
    --imgsz 1280

# Benchmark
python scripts/benchmark.py \
    --model runs/train/fisheye8k-yolov10s-1280/weights/best.pt \
    --onnx-dir runs/export/traffic \
    --imgsz 1280
```

---

