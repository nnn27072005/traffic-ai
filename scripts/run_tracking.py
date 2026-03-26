# scripts/run_tracking.py
from __future__ import annotations
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from src.detector.yolo_wrapper  import YOLODetector
from src.tracker.bytetrack_wrapper import ByteTracker
from src.analyzer.counter       import LineCounter


# ── Annotators ────────────────────────────────────────────────────

CLASS_COLORS = {
    "Bus":        sv.Color(hex="#FF6464"),
    "Bike":       sv.Color(hex="#64C8FF"),
    "Car":        sv.Color(hex="#64FF96"),
    "Pedestrian": sv.Color(hex="#FFB464"),
    "Truck":      sv.Color(hex="#C864FF"),
}
DEFAULT_COLOR = sv.Color(hex="#AAAAAA")


def build_annotators():
    box_ann   = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(
        text_scale     = 0.45,
        text_thickness = 1,
        text_padding   = 4,
    )
    trace_ann = sv.TraceAnnotator(
        thickness  = 2,
        trace_length = 30,   # trail dài 30 frames
    )
    line_ann  = sv.LineZoneAnnotator(
        thickness       = 3,
        color           = sv.Color(hex="#FFFF00"),
        text_scale      = 1.0,
        text_thickness  = 2,
        text_offset     = 1.5,
        text_orient_to_line = True,
    )
    return box_ann, label_ann, trace_ann, line_ann


def draw_stats_overlay(
    frame:  np.ndarray,
    stats,
    fps:    float,
    frame_n: int,
) -> np.ndarray:
    """Vẽ thống kê lên góc trái trên của frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_n}",
        f"Active tracks: {stats.active_tracks}",
        f"Total IN:  {stats.total_in}",
        f"Total OUT: {stats.total_out}",
    ]
    for cls, cnt in stats.per_class_in.items():
        lines.append(f"  {cls}: {cnt}")

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text,
            (18, 35 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


# ── Main pipeline ─────────────────────────────────────────────────

def run_tracking(
    source:     str,
    model_path: str,
    output:     str,
    line_start: tuple = (0,   540),
    line_end:   tuple = (1280, 540),
    confidence: float = 0.25,
    imgsz:      int   = 1280,
    device:     str   = "cuda",
    show:       bool  = False,
) -> None:

    # Init components
    detector = YOLODetector(
        model_path = model_path,
        confidence = confidence,
        imgsz      = imgsz,
        device     = device,
    )
    tracker = ByteTracker(
        track_activation_threshold = confidence,
        lost_track_buffer          = 30,
        frame_rate                 = 25,
    )
    counter = LineCounter(
        line_start = line_start,
        line_end   = line_end,
    )
    box_ann, label_ann, trace_ann, line_ann = build_annotators()

    # Video I/O
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (width, height),
    )

    print(f"Source      : {source}")
    print(f"Resolution  : {width}×{height} @ {fps_src:.1f}fps")
    print(f"Total frames: {total_frames}")
    print(f"Output      : {output}")
    print(f"Line        : {line_start} → {line_end}")

    frame_n  = 0
    fps_timer = time.perf_counter()
    fps_avg   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        t0 = time.perf_counter()

        # ── 1. Detect ────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── 2. Track ─────────────────────────────────────────────
        tracks = tracker.update(detections, frame.shape[:2])

        # ── 3. Count ─────────────────────────────────────────────
        stats = counter.update(tracks)

        # ── 4. Annotate ──────────────────────────────────────────
        if tracks:
            sv_det = sv.Detections(
                xyxy       = np.array([t.bbox_xyxy  for t in tracks]),
                confidence = np.array([t.confidence for t in tracks]),
                class_id   = np.array([t.class_id   for t in tracks]),
                tracker_id = np.array([t.track_id   for t in tracks]),
            )

            labels = [
                f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
                for t in tracks
            ]

            frame = trace_ann.annotate(frame, sv_det)
            frame = box_ann.annotate(frame, sv_det)
            frame = label_ann.annotate(frame, sv_det, labels)

        frame = line_ann.annotate(frame, counter.line)

        # FPS rolling average
        elapsed = time.perf_counter() - t0
        fps_avg = 0.9 * fps_avg + 0.1 * (1.0 / max(elapsed, 1e-6))

        frame = draw_stats_overlay(frame, stats, fps_avg, frame_n)

        writer.write(frame)

        if show:
            cv2.imshow("Traffic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_n % 100 == 0:
            print(f"  Frame {frame_n}/{total_frames} | "
                  f"FPS {fps_avg:.1f} | "
                  f"Tracks {stats.active_tracks} | "
                  f"IN {stats.total_in} OUT {stats.total_out}")

    # Cleanup
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # Final summary
    final = counter.get_stats()
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Total IN  : {final.total_in}")
    print(f"Total OUT : {final.total_out}")
    print("\nPer class (IN):")
    for cls, cnt in sorted(final.per_class_in.items()):
        print(f"  {cls:<14}: {cnt}")
    print(f"\nOutput saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic tracking pipeline")
    parser.add_argument("--source",     required=True,  help="Input video path")
    parser.add_argument("--model",      required=True,  help="Path to best.pt")
    parser.add_argument("--output",     required=True,  help="Output video path")
    parser.add_argument("--confidence", default=0.25,   type=float)
    parser.add_argument("--imgsz",      default=1280,   type=int)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--line-y",     default=0.5,    type=float,
                        help="Counting line Y position (0-1, relative to height)")
    parser.add_argument("--show",       action="store_true")
    args = parser.parse_args()

    # Tính line position từ video
    cap  = cv2.VideoCapture(args.source)
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    line_y = int(h * args.line_y)

    run_tracking(
        source     = args.source,
        model_path = args.model,
        output     = args.output,
        line_start = (0, line_y),
        line_end   = (w, line_y),
        confidence = args.confidence,
        imgsz      = args.imgsz,
        device     = args.device,
        show       = args.show,
    )