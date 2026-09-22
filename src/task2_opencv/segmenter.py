import cv2
import numpy as np

def segment(image: np.ndarray, *, canny_low: int = 50, canny_high: int = 150, morph_kernel: int = 3) -> np.ndarray:
    """Segment defects from a pre‑processed BGR image.

    The pipeline is:
    1. Convert to grayscale.
    2. Apply Canny edge detection.
    3. Morphological closing (kernel size ``morph_kernel``) to connect fragmented edges.
    4. Fill holes and remove tiny specks.

    Parameters
    ----------
    image: np.ndarray
        Pre‑processed BGR image.
    canny_low, canny_high: int
        Thresholds for Canny edge detector.
    morph_kernel: int
        Size of the square structuring element used for closing.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) where ``1`` marks defect pixels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("segment expects a 3‑channel BGR image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Fill holes: find contours, fill them
    mask = np.zeros_like(closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    # Remove small isolated blobs (area < 20 pixels)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            cleaned[labels == i] = 255
    return cleaned
