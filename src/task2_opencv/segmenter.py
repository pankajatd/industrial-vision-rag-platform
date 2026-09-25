import cv2
import numpy as np

def segment(image: np.ndarray, *, threshold: int = 25, morph_kernel: int = 3, min_area: int = 20) -> np.ndarray:
    """Segment defects from a pre-processed BGR image.

    Neutralises horizontal brushed metal grain texture by subtracting row-wise mean intensity,
    isolating true localized surface anomalies (scratches, cracks, corrosion, dimensional flaws).

    Parameters
    ----------
    image: np.ndarray
        Pre-processed BGR image.
    threshold: int, default 25
        Anomaly threshold value above the background noise floor.
    morph_kernel: int, default 3
        Structuring element size for morphological closing.
    min_area: int, default 20
        Minimum area threshold to filter out tiny specks.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) where 255 marks defect pixels and 0 marks background.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("segment expects a 3-channel BGR image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. Neutralise horizontal brushed grain by subtracting row means
    row_means = np.mean(gray, axis=1, keepdims=True)
    grain_removed = np.abs(gray - row_means).astype(np.uint8)
    
    # 2. Threshold the grain-removed anomaly map
    _, thresh = cv2.threshold(grain_removed, threshold, 255, cv2.THRESH_BINARY)
    
    # 3. Morphological closing to connect defect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Remove small isolated noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
            
    return cleaned
