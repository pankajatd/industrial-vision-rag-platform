# plot_confusion.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CSV_PATH = r"task2_demo_output/predictions_full.csv"
df = pd.read_csv(CSV_PATH)

labels = sorted(df["defect_type"].unique())
cm = confusion_matrix(df["defect_type"], df["predicted"], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved heat‑map to confusion_matrix.png")