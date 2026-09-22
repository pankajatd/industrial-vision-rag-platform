import os
import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """Read an image from *path* and return a BGR ``np.ndarray``.
    Raises ``FileNotFoundError`` if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img

def preprocess(image: np.ndarray, *, clahe_clip: float = 2.0, denoise_h: int = 10) -> np.ndarray:
    """Apply standard preprocessing steps to a BGR image.

    Steps:
    1. Convert from BGR to Lab colour space.
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the L‑channel.
    3. Merge back to BGR.
    4. Optionally denoise with ``cv2.fastNlMeansD`` (applied per channel).

    Parameters
    ----------
    image: np.ndarray
        Input BGR image (uint8).
    clahe_clip: float, default 2.0
        Clip limit for the CLAHE algorithm.
    denoise_h: int, default 10
        ``h`` parameter for ``fastNlMeansD``; set to 0 to skip denoising.

    Returns
    -------
    np.ndarray
        Preprocessed BGR image (uint8, same spatial dimensions as input).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preprocess expects a 3‑channel BGR image")

    # 1. Lab conversion
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # 3. Merge and convert back to BGR
    lab_eq = cv2.merge((l_eq, a, b))
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 4. Denoise (optional)
    # 4. Denoise (optional)
    # If fastNlMeansD is available we use it; otherwise fall back to median blur.
    if denoise_h > 0:
        if hasattr(cv2, "fastNlMeansD"):
            denoised = np.empty_like(bgr_eq)
            for c in range(3):
                denoised[:, :, c] = cv2.fastNlMeansD(bgr_eq[:, :, c], h=denoise_h, templateWindowSize=7, searchWindowSize=21)
            return denoised
        else:
            # Median blur works on any cv2 build
            denoised = np.empty_like(bgr_eq)
            for c in range(3):
                denoised[:, :, c] = cv2.medianBlur(bgr_eq[:, :, c], ksize=3)
            return denoised
    else:
        return bgr_eq
