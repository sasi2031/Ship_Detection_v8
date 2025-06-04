import os
import random
from pathlib import Path
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Class Info (from your original dataset)
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
OVERREPRESENTED_CLASSES = [2, 6, 10, 11]

IMAGE_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\images"
LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\labels"
BACKUP_IMAGE_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\images_backup"
BACKUP_LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\labels_backup"

# Create backup directories
Path(BACKUP_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(BACKUP_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# ----------------------
# 📂 Label Reading Function
# ----------------------
def read_labels(label_path):
    class_ids = []
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            class_ids.append(class_id)
    return class_ids

# ----------------------
# 🗑️ Balance Original Dataset
# ----------------------
def balance_original_dataset():
    # Backup original dataset
    logging.info("Creating backup of original dataset...")
    for image_file in os.listdir(IMAGE_DIR):
        if image_file.endswith(('.jpg', '.png')):
            shutil.copy(os.path.join(IMAGE_DIR, image_file), os.path.join(BACKUP_IMAGE_DIR, image_file))
    for label_file in os.listdir(LABEL_DIR):
        if label_file.endswith('.txt'):
            shutil.copy(os.path.join(LABEL_DIR, label_file), os.path.join(BACKUP_LABEL_DIR, label_file))
    logging.info("Backup completed.")

    # Track images and their class contributions
    image_class_map = {}
    current_counts = {k: v["current_count"] for k, v in CLASS_INFO.items()}
    
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(LABEL_DIR, label_file)
        class_ids = read_labels(label_path)
        image_name = Path(label_file).stem
        image_class_map[image_name] = class_ids

    # Log initial counts
    logging.info("Initial instance counts in original dataset:")
    for class_id, count in current_counts.items():
        logging.info(f"Class {class_id} ({CLASS_INFO[class_id]['name']}): {count} (target: {CLASS_INFO[class_id]['target_count']})")

    # Determine images to keep
    images_to_keep = []
    temp_counts = {k: 0 for k in CLASS_INFO}
    target_counts = {k: v["target_count"] for k, v in CLASS_INFO.items()}
    
    # Shuffle images to ensure random selection
    image_names = list(image_class_map.keys())
    random.shuffle(image_names)
    
    for image_name in image_names:
        class_ids = image_class_map[image_name]
        # Keep image if it contains minority classes or doesn't push overrepresented classes too far
        should_keep = True
        for class_id in set(class_ids):
            if class_id in MINORITY_CLASSES:
                should_keep = True
                break
            if class_id in OVERREPRESENTED_CLASSES and temp_counts[class_id] >= target_counts[class_id] * 1.05:
                should_keep = False
                break
        if should_keep:
            images_to_keep.append(image_name)
            for class_id in class_ids:
                temp_counts[class_id] += 1

    # Remove excess images and labels
    for image_name in image_class_map:
        if image_name not in images_to_keep:
            img_path = os.path.join(IMAGE_DIR, image_name + '.jpg')  # Adjust extension if needed
            lbl_path = os.path.join(LABEL_DIR, image_name + '.txt')
            if os.path.exists(img_path):
                os.remove(img_path)
                logging.info(f"Removed image: {img_path}")
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
                logging.info(f"Removed label: {lbl_path}")

    # Update CLASS_INFO with new counts
    final_counts = {k: 0 for k in CLASS_INFO}
    for image_name in images_to_keep:
        for class_id in image_class_map[image_name]:
            final_counts[class_id] += 1

    for class_id in CLASS_INFO:
        CLASS_INFO[class_id]["current_count"] = final_counts[class_id]

    # Log final counts
    logging.info("Final instance counts after balancing original dataset:")
    for class_id, count in final_counts.items():
        logging.info(f"Class {class_id} ({CLASS_INFO[class_id]['name']}): {count} (target: {CLASS_INFO[class_id]['target_count']})")

    # Print updated CLASS_INFO for augmentation script
    logging.info("Updated CLASS_INFO for augmentation script:")
    print("\nUpdated CLASS_INFO:")
    print("CLASS_INFO = {")
    for class_id, info in CLASS_INFO.items():
        print(f"    {class_id}: {{'name': '{info['name']}', 'current_count': {info['current_count']}, 'target_count': {info['target_count']}, 'solo_images': {info['solo_images']}, 'solo_instances': {info['solo_instances']}}},")
    print("}")

    return CLASS_INFO

if __name__ == "__main__":
    balance_original_dataset()