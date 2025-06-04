import os
import shutil
import numpy as np
from pathlib import Path

# --- Define paths ---
image_dir = Path(r"D:\Ship_Detection_Data\SplittedDataset\images\train")
label_dir = Path(r"D:\Ship_Detection_Data\SplittedDataset\labels\train")

empty_label_dir = Path(r"D:\Ship_Detection_Data\SplittedDataset\labels\empty_labels")
empty_image_dir = Path(r"D:\Ship_Detection_Data\SplittedDataset\images\empty_images")

# --- Ensure output directories exist ---
empty_label_dir.mkdir(parents=True, exist_ok=True)
empty_image_dir.mkdir(parents=True, exist_ok=True)

print("🔎 Checking for empty label files...")

for label_file in label_dir.glob("*.txt"):
    # Get corresponding image name
    image_file = image_dir / (label_file.stem + ".png")  # Change .jpg to .png if needed

    # Check if image exists
    if not image_file.exists():
        print(f"❌ Image not found for {label_file.name}, skipping.")
        continue

    # Read label content
    with open(label_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # Check if label is empty (no non-empty lines)
    if not any(lines):
        print(f"📎 Empty label: {label_file.name}")

        # Copy label and image
        shutil.copy(label_file, empty_label_dir / label_file.name)
        shutil.copy(image_file, empty_image_dir / image_file.name)

print("✅ Done. All empty label-image pairs have been copied.")