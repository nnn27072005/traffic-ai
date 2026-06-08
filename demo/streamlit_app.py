# demo/streamlit_app.py
from __future__ import annotations
import sys, time, tempfile
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector.yolo_wrapper     import YOLODetector
from src.tracker.bytetrack_wrapper import ByteTracker
from src.analyzer.counter          import LineCounter
from src.analyzer.violation        import HelmetViolationAnalyzer, draw_violations
import supervision as sv


# ── Config ────────────────────────────────────────────────────────

TRAFFIC_MODEL = "runs/train/fisheye8k-yolov10s-1280/weights/best.pt"
HELMET_MODEL  = "runs/train/helmet-yolov10s-640/weights/best.pt"

st.set_page_config(
    page_title = "Traffic Safety Monitor",
    page_icon  = "🚦",
    layout     = "wide",
)


# ── Load models (cache để không load lại mỗi lần) ────────────────

@st.cache_resource
def load_models(device: str = "cpu"):
    detector = YOLODetector(
        model_path = TRAFFIC_MODEL,
        confidence = 0.25,
        device     = device,
        imgsz      = 640,   # dùng 640 cho CPU
    )
    analyzer = HelmetViolationAnalyzer(
        helmet_model_path = HELMET_MODEL,
        confidence        = 0.30,
        device            = device,
    )
    return detector, analyzer


# ── Annotate frame ────────────────────────────────────────────────

def annotate_frame(
    frame:      np.ndarray,
    tracks:     list,
    violations: list,
    counter,
) -> np.ndarray:
    if tracks:
        sv_det = sv.Detections(
            xyxy       = np.array([t.bbox_xyxy  for t in tracks]),
            confidence = np.array([t.confidence for t in tracks]),
            class_id   = np.array([t.class_id   for t in tracks]),
            tracker_id = np.array([t.track_id   for t in tracks]),
        )
        labels = [
            f"#{t.track_id} {t.class_name} {t.confidence:.0%}"
            for t in tracks
        ]
        box_ann   = sv.BoxAnnotator(thickness=2)
        label_ann = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
        trace_ann = sv.TraceAnnotator(thickness=2, trace_length=20)
        frame = trace_ann.annotate(frame, sv_det)
        frame = box_ann.annotate(frame, sv_det)
        frame = label_ann.annotate(frame, sv_det, labels)

    frame = draw_violations(frame, violations)

    if counter is not None:
        line_ann = sv.LineZoneAnnotator(
            thickness      = 3,
            color          = sv.Color.from_hex("#FFFF00"),
            text_scale     = 0.8,
            text_thickness = 2,
        )
        frame = line_ann.annotate(frame, counter.line)

    return frame


# ── UI ────────────────────────────────────────────────────────────

st.title("🚦 Traffic Safety Monitoring System")
st.caption("YOLOv10s · ByteTrack · Helmet Violation Detection")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    device = st.selectbox(
        "Device",
        ["cpu", "cuda"],
        index=0,
        help="Dùng CPU nếu không có GPU NVIDIA",
    )
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    imgsz      = st.selectbox("Image size", [640, 1280], index=0)
    line_ratio = st.slider("Counting line position", 0.2, 0.8, 0.5, 0.05)

    st.divider()
    st.header("📊 Benchmark")
    st.markdown("""
    | Model | Device | FPS |
    |-------|--------|-----|
    | Traffic | GPU T4 | 24.0 |
    | Traffic | CPU | 1.7 |
    | Helmet | GPU T4 | 76.2 |
    | Helmet | CPU | 7.1 |
    """)

    st.divider()
    st.caption("GPU is **14.2×** faster than CPU")

# Tabs
tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis", "📈 System Info"])


# ── Tab 1: Image ──────────────────────────────────────────────────
with tab1:
    st.subheader("Upload ảnh để phân tích")

    uploaded = st.file_uploader(
        "Chọn ảnh (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded:
        detector, analyzer = load_models(device)
        tracker = ByteTracker()

        # Decode image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w       = frame.shape[:2]

        # Init counter
        line_y   = int(h * line_ratio)
        counter  = LineCounter(line_start=(0, line_y), line_end=(w, line_y))

        # Run pipeline
        with st.spinner("Đang phân tích..."):
            t0         = time.perf_counter()
            detections = detector.detect(frame)
            tracks     = tracker.update(detections, (h, w))
            stats      = counter.update(tracks)
            violations = analyzer.analyze(frame, tracks, frame_idx=1)
            latency    = (time.perf_counter() - t0) * 1000

        # Annotate
        annotated = annotate_frame(
            frame.copy(), tracks, violations, counter
        )
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        original_rgb  = cv2.cvtColor(frame,     cv2.COLOR_BGR2RGB)

        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb,  caption="Original", use_container_width=True)
        with col2:
            st.image(annotated_rgb, caption="Detected",  use_container_width=True)

        # Metrics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vehicles",   len(tracks))
        m2.metric("Violations", len(violations))
        m3.metric("Latency",    f"{latency:.0f}ms")
        m4.metric("Count IN",   stats.total_in)

        # Per-class breakdown
        if tracks:
            from collections import Counter as Ctr
            cls_counts = Ctr(t.class_name for t in tracks)
            st.subheader("Per-class breakdown")
            cols = st.columns(len(cls_counts))
            for col, (cls, cnt) in zip(cols, cls_counts.items()):
                col.metric(cls, cnt)

        # Violations detail
        if violations:
            st.error(f"⚠️ {len(violations)} violation(s) detected!")
            for v in violations:
                st.warning(
                    f"Track #{v.track_id} — "
                    f"{', '.join(v.violations)} "
                    f"(confidence: {v.confidence:.0%})"
                )
        else:
            st.success("✅ No helmet violations detected")

        # Raw JSON
        with st.expander("Raw JSON response"):
            import json
            response = {
                "vehicle_count": len(tracks),
                "violations":    len(violations),
                "latency_ms":    round(latency, 2),
                "tracks": [
                    {
                        "track_id":   t.track_id,
                        "class_name": t.class_name,
                        "confidence": round(t.confidence, 3),
                        "bbox": {
                            "x1": round(float(t.bbox_xyxy[0]), 1),
                            "y1": round(float(t.bbox_xyxy[1]), 1),
                            "x2": round(float(t.bbox_xyxy[2]), 1),
                            "y2": round(float(t.bbox_xyxy[3]), 1),
                        },
                    }
                    for t in tracks
                ],
            }
            st.json(response)


# ── Tab 2: Video ──────────────────────────────────────────────────
with tab2:
    st.subheader("Upload video để tracking")
    st.info(
        "Xử lý video trên CPU sẽ chậm. "
        "Nên dùng video ngắn (< 30s) hoặc chọn 'Sample frames' để demo nhanh."
    )

    mode = st.radio(
        "Chọn source",
        ["Upload video", "Dùng sample frames từ test set"],
        horizontal=True,
    )

    if mode == "Upload video":
        video_file = st.file_uploader(
            "Chọn video (MP4)",
            type=["mp4", "avi", "mov"],
            key="video_upload",
        )
        video_path = None
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(video_file.read())
                video_path = f.name
    else:
        # Dùng video đã có sẵn
        sample_path = Path("data/test_video.mp4")
        if sample_path.exists():
            video_path = str(sample_path)
            st.success(f"Using: {sample_path.name}")
        else:
            st.warning("data/test_video.mp4 not found. Upload video thủ công.")
            video_path = None

    max_frames = st.slider(
        "Số frames tối đa",
        50, 500, 100, 50,
        help="Giới hạn để demo nhanh hơn",
    )

    if video_path and st.button("▶️ Run Tracking", type="primary"):
        detector, analyzer = load_models(device)
        tracker = ByteTracker()
        counter = None

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        preview_slot  = st.empty()
        metrics_slot  = st.empty()

        frame_idx   = 0
        total_viol  = 0
        fps_avg     = 0.0
        output_frames = []

        while frame_idx < min(max_frames, total):
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            h, w = frame.shape[:2]
            t0   = time.perf_counter()

            if counter is None:
                counter = LineCounter(
                    line_start = (0, int(h * line_ratio)),
                    line_end   = (w, int(h * line_ratio)),
                )

            detections = detector.detect(frame)
            tracks     = tracker.update(detections, (h, w))
            stats      = counter.update(tracks)
            violations = analyzer.analyze(frame, tracks, frame_idx)
            total_viol += len(violations)

            elapsed = time.perf_counter() - t0
            fps_avg = 0.9 * fps_avg + 0.1 / max(elapsed, 1e-6)

            annotated = annotate_frame(frame.copy(), tracks, violations, counter)

            # Overlay FPS
            cv2.putText(annotated, f"FPS: {fps_avg:.1f}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            cv2.putText(annotated, f"Violations: {total_viol}",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

            output_frames.append(annotated)

            # Update UI mỗi 10 frames
            if frame_idx % 10 == 0:
                progress_bar.progress(frame_idx / min(max_frames, total))
                status_text.text(
                    f"Frame {frame_idx} | "
                    f"FPS {fps_avg:.1f} | "
                    f"Tracks {len(tracks)} | "
                    f"Violations {total_viol}"
                )
                preview_slot.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_idx}",
                    use_container_width=True,
                )
                with metrics_slot.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("FPS",        f"{fps_avg:.1f}")
                    c2.metric("Tracks",     len(tracks))
                    c3.metric("Total IN",   stats.total_in)
                    c4.metric("Violations", total_viol)

        cap.release()
        progress_bar.progress(1.0)

        # Save output video
        if output_frames:
            out_path = "data/demo_output.mp4"
            h, w = output_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                min(fps_avg, fps), (w, h),
            )
            for f in output_frames:
                writer.write(f)
            writer.release()

            st.success(f"✅ Done! Processed {frame_idx} frames")
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download output video",
                    f,
                    file_name = "tracking_output.mp4",
                    mime      = "video/mp4",
                )


# ── Tab 3: System Info ────────────────────────────────────────────
with tab3:
    st.subheader("System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Models**")
        traffic_exists = Path(TRAFFIC_MODEL).exists()
        helmet_exists  = Path(HELMET_MODEL).exists()
        st.markdown(
            f"- Traffic model: {'✅' if traffic_exists else '❌'} `{TRAFFIC_MODEL}`"
        )
        st.markdown(
            f"- Helmet model:  {'✅' if helmet_exists else '❌'} `{HELMET_MODEL}`"
        )

        st.markdown("**Performance (Tesla T4)**")
        st.table({
            "Model":   ["Traffic 1280", "Traffic 1280", "Helmet 640", "Helmet 640"],
            "Format":  ["PyTorch GPU", "ONNX CPU", "PyTorch GPU", "ONNX CPU"],
            "Latency": ["41.7ms", "592.3ms", "13.1ms", "140.1ms"],
            "FPS":     ["24.0", "1.7", "76.2", "7.1"],
        })

    with col2:
        import torch, platform
        st.markdown("**Environment**")
        st.code(f"""
Python  : {platform.python_version()}
PyTorch : {torch.__version__}
CUDA    : {torch.version.cuda}
GPU     : {'Available ✅' if torch.cuda.is_available() else 'Not available ❌'}
Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
        """)

        st.markdown("**Dataset**")
        st.markdown("""
| Dataset | Images | Classes |
|---------|--------|---------|
| FishEye8K | 8,000 | 5 |
| HelmetViolations | 1,004 | 3 |
        """)