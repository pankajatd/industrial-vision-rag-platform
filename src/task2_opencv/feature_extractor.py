import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from skimage.feature import graycomatrix, graycoprops

def _contour_features(mask: np.ndarray) -> Dict[str, Any]:
    """Calculate geometric features from a binary mask.
    Returns a dictionary with contour count, total area, mean area,
    max aspect ratio and bounding‑box statistics.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    areas = []
    aspect_ratios = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area == 0:
            continue
        areas.append(area)
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect_ratios.append(w / h)
    total_area = sum(areas)
    mean_area = np.mean(areas) if areas else 0.0
    max_aspect = max(aspect_ratios) if aspect_ratios else 0.0
    return {
        "contour_count": contour_count,
        "total_area": float(total_area),
        "mean_contour_area": float(mean_area),
        "max_aspect_ratio": float(max_aspect),
    }

def _hu_moments(image: np.ndarray, mask: np.ndarray) -> List[float]:
    """Compute the 7 Hu moments on the masked region of *image*.
    The image is first converted to grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply mask
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    moments = cv2.moments(masked)
    hu = cv2.HuMoments(moments).flatten()
    # Log‑scale to improve numeric stability
    return [float(-np.sign(h) * np.log10(abs(h)) if h != 0 else 0.0) for h in hu]

def _intensity_stats(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Mean, std, min, max intensity values **inside** the mask (per channel).
    Returns flattened values (average across channels).
    """
    masked_pixels = image[mask == 255]
    if masked_pixels.size == 0:
        return {"intensity_mean": 0.0, "intensity_std": 0.0, "intensity_min": 0.0, "intensity_max": 0.0}
    # Compute per‑channel stats and then average
    mean = masked_pixels.mean(axis=0).mean()
    std = masked_pixels.std(axis=0).mean()
    min_val = masked_pixels.min()
    max_val = masked_pixels.max()
    return {
        "intensity_mean": float(mean),
        "intensity_std": float(std),
        "intensity_min": float(min_val),
        "intensity_max": float(max_val),
    }

def _texture_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Gray‑Level Co‑occurrence Matrix (GLCM) texture features on the masked region.
    The image is converted to 8‑bit grayscale and quantised to 16 levels.
    Returns contrast, dissimilarity, homogeneity, energy, correlation, ASM.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Quantise to 16 gray levels (0‑15)
    quantized = (gray // 16).astype(np.uint8)
    # Apply mask: set background to 0
    quantized[mask == 0] = 0
    # Compute GLCM with distances=[1] and angles=[0]
    glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    props = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        val = graycoprops(glcm, prop)[0, 0]
        props[f"texture_{prop}"] = float(val)
    return props

def extract_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """High‑level wrapper that aggregates all feature groups.
    Returns a flat dictionary that can be directly turned into a ``pandas`` row.
    """
    feats = {}
    feats.update(_contour_features(mask))
    feats.update({f"hu_{i+1}": v for i, v in enumerate(_hu_moments(image, mask))})
    feats.update(_intensity_stats(image, mask))
    feats.update(_texture_features(image, mask))
    return feats

def batch_extract(image_paths: List[str]) -> pd.DataFrame:
    """Run the full pipeline on a list of image file paths.
    For each path we:
    1. Load the image.
    2. Preprocess (CLAHE + optional denoise).
    3. Segment to obtain a binary mask.
    4. Extract features.
    The resulting ``DataFrame`` has one row per image and a column for every feature.
    """
    from .preprocessor import preprocess, load_image
    from .segmenter import segment

    rows = []
    for p in image_paths:
        img = load_image(p)
        pre = preprocess(img)
        mask = segment(pre)
        feat_dict = extract_features(pre, mask)
        feat_dict["image_path"] = p
        rows.append(feat_dict)
    return pd.DataFrame(rows)
