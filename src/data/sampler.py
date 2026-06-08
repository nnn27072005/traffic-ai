# src/data/sampler.py
from __future__ import annotations
import random
from collections import defaultdict
from pathlib import Path


def oversample_minority_classes(
    src_label_dir: str | Path,
    src_image_dir: str | Path,
    dst_image_list: str | Path,   # output: 1 file .txt chứa paths
    target_classes: list[int],
    multiplier: int = 3,
    min_instances: int = 2,
    seed: int = 42,
) -> None:
    """
    Tạo image list file cho Ultralytics thay vì copy ảnh vật lý.
    Ultralytics hỗ trợ train.txt chứa absolute paths — không tốn thêm disk.

    Output file format (1 path per line):
        /kaggle/input/.../train/images/camera1_A_01.png
        /kaggle/input/.../train/images/camera1_A_01.png   ← duplicate = oversample
        /kaggle/input/.../train/images/camera2_B_07.png
    """
    random.seed(seed)
    src_lbl = Path(src_label_dir)
    src_img = Path(src_image_dir)
    dst     = Path(dst_image_list)
    dst.parent.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Build label → image path map
    def find_image(lbl_path: Path) -> Path | None:
        for ext in img_extensions:
            candidate = src_img / lbl_path.with_suffix(ext).name
            if candidate.exists():
                return candidate
        return None

    all_paths: list[str] = []       # tất cả ảnh (base set)
    minority_paths: list[str] = []  # ảnh chứa minority class

    for lbl_file in sorted(src_lbl.glob("*.txt")):
        img_file = find_image(lbl_file)
        if img_file is None:
            continue

        lines = lbl_file.read_text().strip().splitlines()
        cls_counts: dict[int, int] = defaultdict(int)
        for line in lines:
            if line.strip():
                cls_counts[int(line.split()[0])] += 1

        all_paths.append(str(img_file))

        # Kiểm tra có đủ minority instances không
        for tc in target_classes:
            if cls_counts.get(tc, 0) >= min_instances:
                minority_paths.append(str(img_file))
                break

    # Ghi file: toàn bộ base + minority lặp lại (multiplier-1) lần
    total_lines = list(all_paths)
    for _ in range(multiplier - 1):
        total_lines.extend(minority_paths)

    # Shuffle để tránh model thấy minority liên tiếp
    random.shuffle(total_lines)

    dst.write_text("\n".join(total_lines) + "\n")

    print(f"Base images    : {len(all_paths)}")
    print(f"Minority images: {len(minority_paths)} "
          f"(class ids {target_classes}, min {min_instances} instances)")
    print(f"Total in list  : {len(total_lines)} "
          f"(×{len(total_lines)/len(all_paths):.1f} effective multiplier)")
    print(f"Saved to       : {dst}")
    print(f"Disk used      : ~0 MB (no files copied)")