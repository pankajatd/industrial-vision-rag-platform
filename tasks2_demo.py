import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Task 2 imports
from src.task2_opencv import preprocess, segment, batch_extract, load_image

# Task 1 generator – we will create a tiny synthetic dataset on the fly
from src.task1_synthetic.generator import SyntheticIndustrialGenerator


def generate_small_dataset(output_dir: Path, samples_per_class: int = 2, seed: int = 42) -> Path:
    """Create a temporary synthetic dataset using Task 1.
    Returns the directory that contains the generated ``images/`` sub‑folder.
    """
    generator = SyntheticIndustrialGenerator()
    generator.generate_dataset(output_dir=output_dir,
                               samples_per_class=samples_per_class,
                               seed=seed)
    return output_dir


def main():
    # ------------------------------------------------------------------
    # 1️⃣ Create / locate a dataset
    # ------------------------------------------------------------------
    tmp_dir = Path("tmp_task2_demo")
    tmp_dir.mkdir(exist_ok=True)
    #dataset_dir = generate_small_dataset(tmp_dir, samples_per_class=2)
    dataset_dir = generate_small_dataset(tmp_dir, samples_per_class=12)
    print(f"Synthetic dataset generated at: {dataset_dir.resolve()}")

    # ------------------------------------------------------------------
    # 2️⃣ Run the full feature‑extraction pipeline on all images
    # ------------------------------------------------------------------
    image_paths = [str(p) for p in (dataset_dir / "images").glob("*.png")]
    df: pd.DataFrame = batch_extract(image_paths)
    print("\n--- Feature DataFrame (first 5 rows) ---")
    print(df.head())

    # ------------------------------------------------------------------
    # 2️⃣️⃣ Add ground‑truth defect label (defect_type) to the DataFrame
    # ------------------------------------------------------------------
    # The generator also writes a metadata.json file containing the label for each image.
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Align labels with the order of image_paths
    defect_labels = [meta[Path(p).name]["defect_type"] for p in image_paths]
    df["defect_type"] = defect_labels

    # ------------------------------------------------------------------
    # 3️⃣ Save the DataFrame as CSV (now includes defect_type column)
    # ------------------------------------------------------------------
    out_dir = Path("task2_demo_output")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "features.csv", index=False)
    print(f"Feature CSV written to: {out_dir / 'features.csv'}")

    # ------------------------------------------------------------------
    # 4️⃣ Show a visual example of preprocessing + segmentation
    # ------------------------------------------------------------------
    example_path = image_paths[0]
    img = load_image(example_path)
    pre = preprocess(img)
    mask = segment(pre)

    # Save the visualisation files
    cv2.imwrite(str(out_dir / "original.png"), img)
    cv2.imwrite(str(out_dir / "preprocessed.png"), pre)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / "mask.png"), mask_bgr)
    print(f"Demo images written to: {out_dir.resolve()}")

    # ------------------------------------------------------------------
    # 5️⃣ Clean‑up (optional) – comment out the line below if you want to keep the files
    # ------------------------------------------------------------------
    # shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()