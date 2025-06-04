import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import logging
import warnings
import shutil

# Suppress GaussNoise warning
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Class Info & Paths (same as provided)
CLASS_INFO = {
    0: {"name": "Amphibious_Transport_Docks", "current_count": 265, "target_count": 2000, "solo_images": 33, "solo_instances": 70},
    1: {"name": "Carrier_Vessels", "current_count": 206, "target_count": 2000, "solo_images": 36, "solo_instances": 36},
    2: {"name": "Commercial_Cargo", "current_count": 3759, "target_count": 3000, "solo_images": 609, "solo_instances": 1108},
    3: {"name": "Commercial_Tanker", "current_count": 1707, "target_count": 2000, "solo_images": 563, "solo_instances": 745},
    4: {"name": "Commercial_Transport", "current_count": 423, "target_count": 2000, "solo_images": 43, "solo_instances": 66},
    5: {"name": "Fleet_Replenishment_Ships", "current_count": 671, "target_count": 2000, "solo_images": 60, "solo_instances": 96},
    6: {"name": "Naval_Auxiliary", "current_count": 3950, "target_count": 3000, "solo_images": 230, "solo_instances": 492},
    7: {"name": "Naval_Tankers", "current_count": 1053, "target_count": 2000, "solo_images": 95, "solo_instances": 194},
    8: {"name": "Small_Ships", "current_count": 1163, "target_count": 2000, "solo_images": 61, "solo_instances": 103},
    9: {"name": "Submarines", "current_count": 1741, "target_count": 2000, "solo_images": 195, "solo_instances": 501},
    10: {"name": "Surface_War_Ships", "current_count": 6639, "target_count": 3000, "solo_images": 503, "solo_instances": 1188},
    11: {"name": "Tugs", "current_count": 3362, "target_count": 3000, "solo_images": 139, "solo_instances": 318}
}
MINORITY_CLASSES = [0, 1, 4, 5]
MAJORITY_CLASSES = [2, 6, 10, 11]
MODERATE_CLASSES = [3, 7, 8, 9]
EMPTY_IMAGE_TARGET = 2000

IMAGE_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\images"
LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\labels"
OUTPUT_IMAGE_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted_Aug\train\images"
OUTPUT_LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted_Aug\train\labels"

Path(OUTPUT_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# ----------------------
# ✅ Augmentations with CLAHE
# ----------------------
def get_augmentation_transform(strength='light'):
    base_transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ]
    if strength == 'strong':
        base_transforms = [
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.Affine(shear={'x': (-5, 5), 'y': (-5, 5)}, p=0.3),
            A.HorizontalFlip(p=0.2),
        ]
    elif strength == 'medium':
        base_transforms = [
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        ]
    
    return A.Compose(
        base_transforms,
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

# ----------------------
# 📂 Label Read/Write Functions
# ----------------------
def read_labels(label_path):
    obbs = []
    class_ids = []
    if not os.path.exists(label_path):
        return [], []

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = parts[1:]
            obbs.append([(coords[i], coords[i+1]) for i in range(0, 8, 2)])
            class_ids.append(class_id)
    return obbs, class_ids

def write_labels(label_path, obbs, class_ids, height, width):
    with open(label_path, 'w') as f:
        for class_id, corners in zip(class_ids, obbs):
            # Check if OBB is valid (at least one point inside image)
            valid = False
            for x, y in corners:
                x_pixel, y_pixel = x * width, y * height
                if 0 < x_pixel < width and 0 < y_pixel < height:
                    valid = True
                    break
            if not valid:
                continue  # Skip OBBs fully outside the image

            # Clip and normalize coordinates
            flat = []
            for x, y in corners:
                x_pixel = max(0, min(x * width, width))  # Clip to image boundaries
                y_pixel = max(0, min(y * height, height))
                x_norm = x_pixel / width  # Normalize to [0, 1]
                y_norm = y_pixel / height
                flat.append(f"{x_norm:.6f}")
                flat.append(f"{y_norm:.6f}")
            f.write(f"{class_id} {' '.join(flat)}\n")

# ----------------------
# 🔄 Main Augmentation Logic
# ----------------------
def get_augmentation_strategy(class_ids):
    if any(c in MINORITY_CLASSES for c in class_ids):
        return 'strong'
    elif any(c in MODERATE_CLASSES for c in class_ids):
        return 'medium'
    else:
        return 'light'

def apply_augmentation(image, obbs, transform, height, width):
    # Convert OBBs to keypoints (denormalized to pixel coordinates)
    flat_keypoints = []
    for corner in obbs:
        for x, y in corner:
            flat_keypoints.extend([x * width, y * height])
    kps = [(flat_keypoints[i], flat_keypoints[i+1]) for i in range(0, len(flat_keypoints), 2)]

    # Apply augmentation
    augmented = transform(image=image, keypoints=kps)
    aug_image = augmented['image']
    aug_kps = augmented['keypoints']

    # Reconstruct OBBs
    aug_obbs = []
    for i in range(0, len(aug_kps), 4):
        if i + 3 < len(aug_kps):
            aug_obbs.append([
                (aug_kps[i][0] / width, aug_kps[i][1] / height),  # Normalize back
                (aug_kps[i+1][0] / width, aug_kps[i+1][1] / height),
                (aug_kps[i+2][0] / width, aug_kps[i+2][1] / height),
                (aug_kps[i+3][0] / width, aug_kps[i+3][1] / height)
            ])
    return aug_image, aug_obbs

# ----------------------
# 🚀 Run Augmentation
# ----------------------
def augment_dataset():
    current_counts = {k: v["current_count"] for k, v in CLASS_INFO.items()}
    empty_count = 0

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, Path(image_file).stem + ".txt")
        output_base_img = os.path.join(OUTPUT_IMAGE_DIR, Path(image_file).stem)
        output_base_lbl = os.path.join(OUTPUT_LABEL_DIR, Path(image_file).stem)

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            continue

        height, width = image.shape[:2]
        obbs, class_ids = read_labels(label_path)

        if not class_ids:
            empty_count += 1
            shutil.copy(image_path, f"{output_base_img}_orig{Path(image_file).suffix}")
            with open(f"{output_base_lbl}_orig.txt", 'w') as f:
                pass
            continue

        strategy = get_augmentation_strategy(class_ids)
        transform = get_augmentation_transform(strategy)
        instances = len(class_ids)

        # Copy original
        shutil.copy(image_path, f"{output_base_img}_orig{Path(image_file).suffix}")
        write_labels(f"{output_base_lbl}_orig.txt", obbs, class_ids, height, width)

        # Determine number of augmentations
        augmentations_needed = 0
        for class_id in set(class_ids):
            target = CLASS_INFO[class_id]["target_count"]
            needed = max(0, (target - current_counts[class_id]) // instances)
            augmentations_needed = max(augmentations_needed, min(needed, 10))

        # Apply augmentations
        for i in range(max(1, augmentations_needed)):
            try:
                aug_image, aug_obbs = apply_augmentation(image, obbs, transform, height, width)
                aug_class_ids = class_ids[:]
                aug_label_path = f"{output_base_lbl}_aug_{i}.txt"
                aug_image_path = f"{output_base_img}_aug_{i}{Path(image_file).suffix}"

                # Save image and label
                cv2.imwrite(aug_image_path, aug_image)
                write_labels(aug_label_path, aug_obbs, aug_class_ids, height, width)

                # Update counts
                for class_id in aug_class_ids:
                    current_counts[class_id] += 1
            except Exception as e:
                logging.error(f"Error during augmentation of {image_file}: {e}")

    # Final Log
    logging.info("Final instance counts after augmentation:")
    for class_id, count in current_counts.items():
        name = CLASS_INFO[class_id]['name']
        target = CLASS_INFO[class_id]['target_count']
        logging.info(f"Class {class_id} ({name}): {count} (target: {target})")

if __name__ == "__main__":
    augment_dataset()