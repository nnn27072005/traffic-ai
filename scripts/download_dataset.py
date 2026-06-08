# scripts/download_data.py — thêm function mới

from huggingface_hub import snapshot_download
from pathlib import Path


def download_fisheye8k_from_hf(
    dest: str | Path,
    repo_id: str = "hieupth/fisheye8k",
) -> Path:
    """
    Args:
        dest   : folder lưu dataset, vd: "./data/fisheye8k"
        repo_id: HF dataset repo, default hieupth/fisheye8k

    Returns:
        Path tới dataset root
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} → {dest}")

    local_path = snapshot_download(
        repo_id   = repo_id,
        repo_type = "dataset",
        local_dir = str(dest),
        ignore_patterns = ["*.git*", "*.md"],
    )

    print(f"Downloaded to: {local_path}")
    return Path(local_path)


def download_helmet_from_hf(
    dest: str | Path,
    repo_id: str = "helmet-nztej/helmetviolations",   # placeholder
) -> Path:
    """
    Helmet dataset — nếu không có trên HF thì dùng Roboflow.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id   = repo_id,
        repo_type = "dataset",
        local_dir = str(dest),
    )
    return Path(local_path)

def main():
    # Ví dụ: tải fisheye8k về ./data/fisheye8k
    download_fisheye8k_from_hf(dest="./data/fisheye8k")

    # Ví dụ: tải helmet dataset về ./data/helmet
    download_helmet_from_hf(dest="./data/helmet")

if __name__ == "__main__":
    main()