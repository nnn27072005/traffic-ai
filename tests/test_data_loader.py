# tests/test_data_loader.py
import pytest
from pathlib import Path
from src.data.dataset import FishEye8KDataset, BoundingBox

DATA_ROOT = "/kaggle/input/datasets/flap1812/fisheye8k/Fisheye8K"

@pytest.fixture
def small_dataset():
    ds = FishEye8KDataset(DATA_ROOT, split="test")
    return ds

def test_dataset_loads(small_dataset):
    assert len(small_dataset) > 0

def test_sample_has_image(small_dataset):
    sample = small_dataset[0]
    assert sample.image_path.exists()

def test_image_readable(small_dataset):
    import cv2
    img = small_dataset[0].load_image()
    assert img is not None
    assert img.ndim == 3

def test_labels_valid_range(small_dataset):
    for sample in small_dataset.samples[:100]:
        for box in sample.boxes:
            assert 0 <= box.class_id < 5
            assert 0 <= box.cx <= 1
            assert 0 <= box.cy <= 1
            assert 0 <  box.w  <= 1
            assert 0 <  box.h  <= 1

def test_class_distribution_reasonable(small_dataset):
    dist = small_dataset.get_class_distribution()
    # Car phải là class nhiều nhất
    assert dist["Car"] == max(dist.values())
    # Tất cả classes phải có ít nhất 1 instance
    assert all(v > 0 for v in dist.values())