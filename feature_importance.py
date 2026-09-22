# feature_importance.py
"""
Utility script to load the trained RandomForest model and the full predictions
CSV, compute feature importances, and plot them as a sorted horizontal bar chart.
The plot is saved as `feature_importance.png` in the current directory.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Paths – adjust if you move files
MODEL_PATH = r"data/models/rf_model.joblib"
CSV_PATH = r"task2_demo_output/predictions_full.csv"

def main():
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load predictions CSV (contains all feature columns + labels)
    df = pd.read_csv(CSV_PATH)

    # Identify feature columns (everything except the three label columns)
    label_cols = {"image_path", "defect_type", "predicted"}
    feature_cols = [col for col in df.columns if col not in label_cols]

    # Get importances from the RandomForest classifier
    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        raise ValueError("Number of importances does not match number of features.")

    # Sort features by importance (descending)
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("RandomForest Feature Importances")
    plt.barh(range(len(sorted_features)), sorted_importances, align="center")
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importances plotted and saved to feature_importance.png")
    # Also print top‑5 to console for quick view
    print("Top 5 features:")
    for i in range(min(5, len(sorted_features))):
        print(f"{i+1}. {sorted_features[i]}: {sorted_importances[i]:.4f}")

if __name__ == "__main__":
    main()
