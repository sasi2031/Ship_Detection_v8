import os
from collections import Counter

def count_class_instances(labels_dir, classes_file):
    # Load class names from classes.txt
    class_names = []
    try:
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    if not class_names:
        print(f"Error: classes.txt at {classes_file} is empty")
        return
    
    # Initialize counter for class instances
    class_counts = Counter()
    
    # Process each .txt file in labels directory
    total_files = 0
    for label_file in os.listdir(labels_dir):
        if label_file.lower().endswith('.txt'):
            total_files += 1
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Assume YOLOv8-OBB format: class_idx x1 y1 x2 y2 x3 y3 x4 y4
                        parts = line.split()
                        if len(parts)< 9:  # Expect class_idx + 8 coordinates
                            print(f"Warning: Malformed line in {label_file}: {line}")
                            continue
                        try:
                            class_idx = int(parts[0])
                            if 0 <= class_idx < len(class_names):
                                class_counts[class_idx] += 1
                            else:
                                print(f"Warning: Invalid class index {class_idx} in {label_file}")
                        except ValueError:
                            print(f"Warning: Invalid class index in {label_file}: {parts[-2]}")
            except Exception as e:
                print(f"Error reading {label_file}: {str(e)}")
    
    # Print results
    print("\nClass Instance Counts:")
    total_instances = 0
    for class_idx in range(len(class_names)):
        count = class_counts.get(class_idx, 0)
        print(f"  {class_names[class_idx]} (index {class_idx}): {count}")
        total_instances += count
    
    print(f"\nSummary:")
    print(f"Total label files processed: {total_files}")
    print(f"Total class instances: {total_instances}")
    if total_instances == 0:
        print("Note: No valid class instances found. Check label files and classes.txt.")

def main():
    # Configuration
    print("Starting class instance counting... for train")
    labels_dir = r"D:\ArmyVehicles_Data_3_Classes\train" # Update to your labels folder
    classes_file = r"D:\ArmyVehicles_Data_3_Classes\classes.txt"  # Update to your classes.txt path
    
    # Verify labels directory exists
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory {labels_dir} does not exist")
        return
    
    # Count class instances
    count_class_instances(labels_dir, classes_file)

if __name__ == "__main__":
    main()