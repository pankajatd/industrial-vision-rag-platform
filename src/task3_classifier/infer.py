import os
import joblib
from pathlib import Path
import cv2
import pandas as pd

from src.task2_opencv import load_image, preprocess, segment, extract_features

def load_model(model_path: str):
    """Load a trained RandomForest model from ``model_path``.
    Returns the scikit‑learn estimator.
    """
    return joblib.load(model_path)

def predict_defect(image_path: str, model_path: str) -> str:
    """Predict the defect label for a single image.

    Steps:
    1. Load the image.
    2. Run Task 2 preprocessing, segmentation and feature extraction.
    3. Convert the feature dict to a one‑row ``pandas.DataFrame``.
    4. Load the trained model and call ``predict``.
    5. Return the predicted label as a string.
    """
    # 1. Load image using the same helper that the rest of the pipeline uses
    img = load_image(image_path)
    # 2. Pre‑process and segment
    pre = preprocess(img)
    mask = segment(pre)
    # 3. Extract features – this returns a flat dict
    feats = extract_features(pre, mask)
    # Convert to DataFrame (single row) – column order does not matter for prediction
    X = pd.DataFrame([feats])
    # 4. Load the model and predict
    clf = load_model(model_path)
    pred = clf.predict(X)
    # ``pred`` is an array-like, we return the first element as string
    return str(pred[0])
