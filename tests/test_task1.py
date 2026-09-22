"""
Unit and Integration Tests for Task 1: Synthetic Generator & Virtual Camera
"""

import os
import shutil
import tempfile
import pytest
import numpy as np

from src.task1_synthetic.generator import SyntheticIndustrialGenerator, DefectType
from src.task1_synthetic.virtual_camera import VirtualCameraCapture


@pytest.fixture
def generator():
    return SyntheticIndustrialGenerator(img_size=(256, 256), seed=123)


def test_base_surface_generation(generator):
    """Verifies that the generated metal surface matches expected shape, dtype, and intensity."""
    surface = generator.generate_base_surface()
    assert surface.shape == (256, 256, 3)
    assert surface.dtype == np.uint8
    mean_val = np.mean(surface)
    assert 120 < mean_val < 220, f"Expected realistic metal surface brightness, got {mean_val}"


def test_defect_injection_all_classes(generator):
    """Verifies generation of all 5 defect classes, checking metadata and mask consistency."""
    for dtype in DefectType:
        img, metadata, mask = generator.generate_sample(defect_type=dtype, severity=5.0)

        # Image checks
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8

        # Mask checks
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8

        # Metadata checks
        assert metadata["defect_type"] == dtype.value
        assert "severity_score" in metadata
        assert "severity_level" in metadata
        assert "bounding_box" in metadata
        assert "affected_pixel_ratio" in metadata

        if dtype == DefectType.NORMAL:
            assert metadata["severity_score"] == 0.0
            assert metadata["severity_level"] == "PASS"
            assert metadata["bounding_box"] is None
            assert np.count_nonzero(mask) == 0
        else:
            assert metadata["severity_score"] == 5.0
            assert metadata["severity_level"] in ["LOW", "MEDIUM", "CRITICAL"]
            assert np.count_nonzero(mask) > 0
            bbox = metadata["bounding_box"]
            assert bbox is not None and len(bbox) == 4
            x, y, w, h = bbox
            assert w > 0 and h > 0


def test_severity_scaling_behavior():
    """Confirms that higher severity injects proportionally larger/more severe defects."""
    gen = SyntheticIndustrialGenerator(img_size=(256, 256), seed=42)

    # Compare low vs high severity on corrosion
    _, meta_low, mask_low = gen.generate_sample(DefectType.CORROSION, severity=2.0)
    _, meta_high, mask_high = gen.generate_sample(DefectType.CORROSION, severity=9.0)

    area_low = np.count_nonzero(mask_low)
    area_high = np.count_nonzero(mask_high)
    assert area_high > area_low, f"Expected high severity area ({area_high}) > low severity area ({area_low})"


def test_dataset_generation_and_metadata():
    """Tests generating a batch dataset on disk and verifies metadata.json."""
    temp_dir = tempfile.mkdtemp()
    try:
        gen = SyntheticIndustrialGenerator(img_size=(128, 128), seed=99)
        summary = gen.generate_dataset(output_dir=temp_dir, samples_per_class=2, seed=99)

        assert summary["total_samples"] == 10  # 5 classes * 2 samples
        assert os.path.exists(summary["metadata_path"])
        assert os.path.exists(summary["images_dir"])

        # Verify image files
        img_files = os.listdir(summary["images_dir"])
        assert len(img_files) == 10

        # Verify mask files
        mask_files = os.listdir(os.path.join(temp_dir, "masks"))
        assert len(mask_files) == 10
    finally:
        shutil.rmtree(temp_dir)


def test_virtual_camera_folder_stream():
    """Tests streaming frames from a folder dataset with VirtualCameraCapture."""
    temp_dir = tempfile.mkdtemp()
    try:
        gen = SyntheticIndustrialGenerator(img_size=(128, 128), seed=10)
        gen.generate_dataset(output_dir=temp_dir, samples_per_class=2, seed=10)

        cam = VirtualCameraCapture(source_dir=temp_dir, loop=False)
        assert cam.isOpened() is True

        frames_read = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            assert frame is not None
            assert frame.shape == (128, 128, 3)
            meta = cam.get_metadata()
            assert meta is not None
            assert "defect_type" in meta
            frames_read += 1

        assert frames_read == 10
        assert cam.isOpened() is True
        cam.release()
        assert cam.isOpened() is False
    finally:
        shutil.rmtree(temp_dir)


def test_virtual_camera_dynamic_stream():
    """Tests generating and streaming frames dynamically on the fly."""
    gen = SyntheticIndustrialGenerator(img_size=(128, 128), seed=77)
    cam = VirtualCameraCapture(generator=gen, fps=0.0)

    assert cam.isOpened() is True
    for _ in range(5):
        ret, frame = cam.read()
        assert ret is True
        assert frame is not None
        assert frame.shape == (128, 128, 3)
        meta = cam.get_metadata()
        assert meta is not None
        assert meta["defect_type"] in [d.value for d in DefectType]

    cam.release()
    assert cam.isOpened() is False
