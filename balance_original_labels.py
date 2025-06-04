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
OVERREPRESENTED_CLASSES = [2, 6, 10, 11]

LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\labels"
BACKUP_LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\labels_backup"

# Create backup directory
Path(BACKUP_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# ----------------------
# 📂 Label Reading/Writing Functions
# ----------------------
def read_labels(label_path):
    annotations = []
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            annotations.append((class_id, coords))
    return annotations

def write_labels(label_path, annotations):
    with open(label_path, 'w') as f:
        for class_id, coords in annotations:
            flat = [f"{x:.6f}" for x in coords]
            f.write(f"{class_id} {' '.join(flat)}\n")

# ----------------------
# 🗑️ Balance Original Dataset Labels
# ----------------------
def balance_original_labels():
    # Backup original label files
    logging.info("Creating backup of original label files...")
    for label_file in os.listdir(LABEL_DIR):
        if label_file.endswith('.txt'):
            shutil.copy(os.path.join(LABEL_DIR, label_file), os.path.join(BACKUP_LABEL_DIR, label_file))
    logging.info("Backup completed.")

    # Track instances to remove
    instance_list = []
    current_counts = {k: v["current_count"] for k, v in CLASS_INFO.items()}
    
    # Collect all instances of overrepresented classes
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(LABEL_DIR, label_file)
        annotations = read_labels(label_path)
        for idx, (class_id, coords) in enumerate(annotations):
            if class_id in OVERREPRESENTED_CLASSES:
                instance_list.append((label_file, idx, class_id))

    # Shuffle instances to ensure random removal
    random.shuffle(instance_list)
    
    # Determine how many instances to remove per class
    instances_to_remove = {}
    for class_id in OVERREPRESENTED_CLASSES:
        excess = current_counts[class_id] - CLASS_INFO[class_id]["target_count"]
        instances_to_remove[class_id] = max(0, excess)

    # Mark instances to remove
    remove_indices = {label_file: set() for label_file in os.listdir(LABEL_DIR) if label_file.endswith('.txt')}
    for label_file, idx, class_id in instance_list:
        if instances_to_remove.get(class_id, 0) > 0:
            remove_indices[label_file].add(idx)
            instances_to_remove[class_id] -= 1

    # Update label files
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(LABEL_DIR, label_file)
        annotations = read_labels(label_path)
        # Keep only annotations not marked for removal
        new_annotations = [
            (class_id, coords) for idx, (class_id, coords) in enumerate(annotations)
            if idx not in remove_indices.get(label_file, set())
        ]
        write_labels(label_path, new_annotations)
        logging.info(f"Updated label file: {label_path}")

    # Recalculate counts
    final_counts = {k: 0 for k in CLASS_INFO}
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith('.txt'):
            continue
        annotations = read_labels(os.path.join(LABEL_DIR, label_file))
        for class_id, _ in annotations:
            final_counts[class_id] += 1

    # Update CLASS_INFO
    for class_id in CLASS_INFO:
        CLASS_INFO[class_id]["current_count"] = final_counts[class_id]

    # Log final counts
    logging.info("Final instance counts after balancing original labels:")
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
    balance_original_labels()