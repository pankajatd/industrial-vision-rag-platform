# evaluate_model.py
"""
Utility script to evaluate the trained RandomForest model on the full validation set
contained in ``task2_demo_output/features.csv``.

It loads the CSV (which already has the true ``defect_type`` column), runs the
full Task 2 pipeline on each image, predicts the label, and prints a side‑by‑side
comparison plus the overall accuracy.

Usage::
    python evaluate_model.py
"""

import pandas as pd
from pathlib import Path

from src.task3_classifier import load_model, predict_defect

def main() -> None:
    # Load the CSV that already contains the true label
    csv_path = Path("task2_demo_output/features.csv")
    df = pd.read_csv(csv_path)

    # Load the trained model once
    model_path = Path("data/models/rf_model.joblib")
    model = load_model(str(model_path))  # model is loaded inside predict_defect, but we keep the call for symmetry

    # Predict for every image in the CSV
    preds = []
    for img_path in df["image_path"]:
        # Convert relative path (stored in CSV) to absolute path so loader can find the file.
        abs_path = Path(img_path).resolve()
        preds.append(predict_defect(str(abs_path), str(model_path)))

    df["predicted"] = preds

    # Show a quick comparison
    print(df[["image_path", "defect_type", "predicted"]].head(10))

    # Compute overall accuracy ourselves (just to double‑check)
    accuracy = (df["defect_type"] == df["predicted"]).mean()
    print(f"\nOverall accuracy on the full CSV: {accuracy:.2%}")
    # Save full predictions for future reference
    df.to_csv('task2_demo_output/predictions_full.csv', index=False)
    print('Predictions saved to task2_demo_output/predictions_full.csv')

if __name__ == "__main__":
    main()
