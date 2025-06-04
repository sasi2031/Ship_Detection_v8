import os
import shutil
from pathlib import Path
import random
import numpy as np

# Define class indices and counts from your dataset
CLASS_COUNTS = {
    0: 361,   # Amphibious_Transport_Docks
    1: 262,   # Carrier_Vessels
    2: 4790,  # Commercial_Cargo
    3: 2164,  # Commercial_Tanker
    4: 535,   # Commercial_Transport
    5: 593,   # Fleet_Replenishment_Ships
    6: 4897,  # Naval_Auxiliary
    7: 1307,  # Naval_Tankers
    8: 1430,  # Small_Ships
    9: 2169,  # Submarines
    10: 8282, # Surface_War_Ships
    11: 4179  # Tugs
}
MINORITY_CLASSES = [0, 1, 4, 5]  # Classes with low instance counts
MAJORITY_CLASSES = [2, 3, 6, 7, 8, 9, 10, 11]

# Directories
IMAGE_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8\images"  # Update with your image directory
LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8\labels"  # Update with your label directory
OUTPUT_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted"  # Output directory for split dataset
TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DIR, "train/images")
TRAIN_LABEL_DIR = os.path.join(OUTPUT_DIR, "train/labels")
VAL_IMAGE_DIR = os.path.join(OUTPUT_DIR, "val/images")
VAL_LABEL_DIR = os.path.join(OUTPUT_DIR, "val/labels")

# Create output directories
for d in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR, VAL_LABEL_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Function to read labels and return class IDs
def read_labels(label_path):
    class_ids = []
    if not os.path.exists(label_path):
        return class_ids
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # Ensure valid YOLO OBB format
                class_id = int(parts[0])
                class_ids.append(class_id)
    return class_ids

# Function to check if image is solo-class and minority
def is_solo_minority(class_ids):
    unique_classes = set(class_ids)
    return len(unique_classes) == 1 and list(unique_classes)[0] in MINORITY_CLASSES

# Main splitting function
def split_dataset():
    # Initialize lists
    solo_minority_images = []
    other_images = []
    class_counts_train = {k: 0 for k in CLASS_COUNTS}
    class_counts_val = {k: 0 for k in CLASS_COUNTS}

    # Analyze images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    for image_file in image_files:
        label_path = os.path.join(LABEL_DIR, f"{Path(image_file).stem}.txt")
        class_ids = read_labels(label_path)
        if is_solo_minority(class_ids):
            solo_minority_images.append(image_file)
        else:
            other_images.append(image_file)

    # Split solo minority images (70:30 for minority classes)
    random.shuffle(solo_minority_images)
    train_solo_count = int(len(solo_minority_images) * 0.7)
    train_solo_images = solo_minority_images[:train_solo_count]
    val_solo_images = solo_minority_images[train_solo_count:]

    # Split other images (80:20)
    random.shuffle(other_images)
    train_other_count = int(len(other_images) * 0.8)
    train_other_images = other_images[:train_other_count]
    val_other_images = other_images[train_other_count:]

    # Combine splits
    train_images = train_solo_images + train_other_images
    val_images = val_solo_images + val_other_images

    # Copy files and count instances
    for image_file in train_images:
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, f"{Path(image_file).stem}.txt")
        shutil.copy(image_path, os.path.join(TRAIN_IMAGE_DIR, image_file))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(TRAIN_LABEL_DIR, f"{Path(image_file).stem}.txt"))
            class_ids = read_labels(label_path)
            for cid in class_ids:
                class_counts_train[cid] += 1

    for image_file in val_images:
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, f"{Path(image_file).stem}.txt")
        shutil.copy(image_path, os.path.join(VAL_IMAGE_DIR, image_file))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(VAL_LABEL_DIR, f"{Path(image_file).stem}.txt"))
            class_ids = read_labels(label_path)
            for cid in class_ids:
                class_counts_val[cid] += 1

    # Print instance counts
    print("Training set instance counts:")
    for cid, count in class_counts_train.items():
        print(f"Class {cid}: {count} instances ({count / CLASS_COUNTS[cid] * 100:.1f}% of original)")
    print("\nValidation set instance counts:")
    for cid, count in class_counts_val.items():
        print(f"Class {cid}: {count} instances ({count / CLASS_COUNTS[cid] * 100:.1f}% of original)")
    print(f"\nTotal images: Training = {len(train_images)}, Validation = {len(val_images)}")

if __name__ == "__main__":
    split_dataset()