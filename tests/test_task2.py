import os
import shutil
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Import Task 1 generator to create a tiny synthetic dataset for testing
from src.task1_synthetic.generator import SyntheticIndustrialGenerator

# Import Task 2 functions we want to test
from src.task2_opencv import preprocess, load_image, segment, extract_features, batch_extract

@pytest.fixture(scope="module")
def synthetic_dataset_dir():
    """Create a temporary directory with a small synthetic dataset (2 images per defect type)."""
    tmp_dir = Path(tempfile.mkdtemp())
    # Use the generator to create 2 samples per class → 10 images total
    generator = SyntheticIndustrialGenerator()
    generator.generate_dataset(output_dir=tmp_dir, samples_per_class=2, seed=123)
    yield tmp_dir
    # Cleanup after tests
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_preprocess_returns_uint8_and_same_shape(synthetic_dataset_dir):
    # Pick the first generated image
    img_path = next(synthetic_dataset_dir.glob("images/*.png"))
    img = load_image(str(img_path))
    pre = preprocess(img)
    assert isinstance(pre, np.ndarray)
    assert pre.dtype == np.uint8
    # Shape should be unchanged (height, width, 3)
    assert pre.shape == img.shape

def test_segment_produces_nonempty_mask(synthetic_dataset_dir):
    img_path = next(synthetic_dataset_dir.glob("images/*.png"))
    img = load_image(str(img_path))
    pre = preprocess(img)
    mask = segment(pre)
    # Mask should be a uint8 binary image
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    # Ensure at least one defect pixel is detected (not all zeros)
    assert mask.any()
    # Values should be only 0 or 255
    unique_vals = np.unique(mask)
    assert set(unique_vals).issubset({0, 255})

def test_extract_features_returns_expected_keys(synthetic_dataset_dir):
    img_path = next(synthetic_dataset_dir.glob("images/*.png"))
    img = load_image(str(img_path))
    pre = preprocess(img)
    mask = segment(pre)
    feats = extract_features(pre, mask)
    # Basic sanity checks on the dictionary
    assert isinstance(feats, dict)
    # Geometry keys
    for key in ["contour_count", "total_area", "mean_contour_area", "max_aspect_ratio"]:
        assert key in feats
    # Hu moment keys
    for i in range(1, 8):
        assert f"hu_{i}" in feats
    # Intensity stats
    for key in ["intensity_mean", "intensity_std", "intensity_min", "intensity_max"]:
        assert key in feats
    # Texture keys (skimage GLCM provides these six)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        assert f"texture_{prop}" in feats
    # Values should be numeric and not NaN
    for v in feats.values():
        assert isinstance(v, (int, float))
        assert not np.isnan(v)

def test_batch_extract_returns_dataframe(synthetic_dataset_dir):
    # Collect all image file paths
    image_paths = list(map(str, synthetic_dataset_dir.glob("images/*.png")))
    df: pd.DataFrame = batch_extract(image_paths)
    # DataFrame should have one row per image
    assert len(df) == len(image_paths)
    # Must contain the image_path column and all feature columns
    required_columns = {
        "image_path",
        "contour_count",
        "total_area",
        "mean_contour_area",
        "max_aspect_ratio",
    }
    # Add Hu moment columns
    required_columns.update({f"hu_{i}" for i in range(1, 8)})
    # Add intensity stats
    required_columns.update({"intensity_mean", "intensity_std", "intensity_min", "intensity_max"})
    # Add texture columns
    required_columns.update({f"texture_{p}" for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]})

    assert required_columns.issubset(set(df.columns))
    # Ensure no NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert not df[numeric_cols].isnull().any().any()
