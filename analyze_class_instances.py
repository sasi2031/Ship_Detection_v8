import os
from pathlib import Path

# Define class names and original counts for reference
CLASS_INFO = {
    0: {"name": "Amphibious_Transport_Docks", "total_count": 361},
    1: {"name": "Carrier_Vessels", "total_count": 262},
    2: {"name": "Commercial_Cargo", "total_count": 4790},
    3: {"name": "Commercial_Tanker", "total_count": 2164},
    4: {"name": "Commercial_Transport", "total_count": 535},
    5: {"name": "Fleet_Replenishment_Ships", "total_count": 593},
    6: {"name": "Naval_Auxiliary", "total_count": 4897},
    7: {"name": "Naval_Tankers", "total_count": 1307},
    8: {"name": "Small_Ships", "total_count": 1430},
    9: {"name": "Submarines", "total_count": 2169},
    10: {"name": "Surface_War_Ships", "total_count": 8282},
    11: {"name": "Tugs", "total_count": 4179}
}

# Directory with label files
LABEL_DIR = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\train\labels"  # Update with your label directory

# Function to read class IDs from a label file
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

# Main analysis function
def analyze_class_instances():
    # Initialize counters
    total_instances = {k: 0 for k in CLASS_INFO}
    solo_images = {k: 0 for k in CLASS_INFO}  # Count of solo-class images
    solo_instances = {k: 0 for k in CLASS_INFO}  # Instances in solo-class images
    total_images = 0

    # Process label files
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    for label_file in label_files:
        label_path = os.path.join(LABEL_DIR, label_file)
        class_ids = read_labels(label_path)
        
        if not class_ids:
            continue  # Skip empty label files
        
        total_images += 1
        
        # Update total instance counts
        for class_id in class_ids:
            if class_id in total_instances:
                total_instances[class_id] += 1
        
        # Check if solo-class image
        unique_classes = set(class_ids)
        if len(unique_classes) == 1:
            class_id = list(unique_classes)[0]
            if class_id in solo_images:
                solo_images[class_id] += 1
                solo_instances[class_id] += len(class_ids)

    # Print results
    print("Total Class Instance Counts:")
    print("----------------------------")
    for class_id, count in total_instances.items():
        class_name = CLASS_INFO[class_id]["name"]
        original_count = CLASS_INFO[class_id]["total_count"]
        print(f"Class {class_id} ({class_name}): {count} instances (matches original: {count == original_count})")
    
    print("\nSolo-Class Image Counts and Instances:")
    print("-------------------------------------")
    for class_id, image_count in solo_images.items():
        class_name = CLASS_INFO[class_id]["name"]
        instance_count = solo_instances[class_id]
        print(f"Class {class_id} ({class_name}): {image_count} solo images, {instance_count} instances "
              f"({instance_count / CLASS_INFO[class_id]['total_count'] * 100:.1f}% of total class instances)")

    print(f"\nTotal images processed: {total_images}")

if __name__ == "__main__":
    analyze_class_instances()