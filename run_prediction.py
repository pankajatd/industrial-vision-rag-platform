# run_prediction.py
"""
run_prediction.py

A tiny command‑line helper that runs the full Task 2 pipeline on every image in a folder,
loads the RandomForest model you trained (data/models/rf_model.joblib),
predicts the defect type, and writes the results to a CSV.

Usage::
    python run_prediction.py <input_folder> [--output <output_csv>]

If ``--output`` is omitted the script creates ``predictions.csv`` inside ``<input_folder>``.
"""

import argparse
from pathlib import Path
import pandas as pd

from src.task3_classifier import predict_defect


def main() -> None:
    parser = argparse.ArgumentParser(description="Run defect classification on a folder of images using the trained Random Forest model.")
    parser.add_argument("input_folder", type=str, help="Folder containing image files (PNG/JPG).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to CSV file for predictions (default: <input_folder>/predictions.csv).")
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    # Gather image files (common extensions)
    image_paths = list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg"))
    if not image_paths:
        print(f"No image files found in {input_dir}")
        return

    model_path = Path("data/models/rf_model.joblib")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Did you run the training step?")

    results = []
    for img_path in image_paths:
        try:
            label = predict_defect(str(img_path), str(model_path))
        except Exception as exc:
            label = f"ERROR: {exc}"  # keep the flow even if a single image fails
        results.append({"image_path": str(img_path), "predicted_defect": label})

    df = pd.DataFrame(results)
    output_csv = Path(args.output) if args.output else input_dir / "predictions.csv"
    df.to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}")


if __name__ == "__main__":
    main()
