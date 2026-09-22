# src/task3_classifier/api.py
"""
Simple public API for the Task 3 classifier.

- `load_model()` – loads the saved RandomForest model (cached).
- `predict(image_path)` – runs the full Task 2 pipeline on *image_path*
  and returns the predicted defect label.
"""

import pathlib
import joblib
from .infer import load_model, predict_defect

# Cache the model after the first load to avoid re‑reading the file each call
_model = None
_MODEL_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "models" / "rf_model.joblib"

def get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        _model = load_model(str(_MODEL_PATH))
    return _model

def predict(image_path: str) -> str:
    """
    Predict the defect type for a single image.

    Parameters
    ----------
    image_path: str
        Absolute or relative path to a PNG/JPG image.

    Returns
    -------
    str
        The predicted defect label (e.g. 'crack', 'normal', …).
    """
    model_path = str(_MODEL_PATH)      # predict_defect expects the model file path
    return predict_defect(str(image_path), model_path)