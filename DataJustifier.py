import os
import random
import shutil
from collections import Counter

def balance_training_set(train_images_dir, train_labels_dir, excess_dir, classes_file, commercial_limit=2500, frigate_limit=2500):
    # Create excess directory
    os.makedirs(os.path.join(excess_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(excess_dir, 'labels'), exist_ok=True)
    
    # Load class names
    class_names = []
    try:
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        if not class_names:
            print(f"Error: classes.txt at {classes_file} is empty")
            return
    except FileNotFoundError:
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')]
    if not image_files:
        print(f"Error: No .png files found in {train_images_dir}")
        return
    
    # Categorize images by classes and count instances per image
    image_class_counts = {}
    carrier_files = []
    submarine_files = []
    commercial_files = []
    frigate_files = []
    other_files = []
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(train_labels_dir, f"{base_name}.txt")
        classes = set()
        class_counts = Counter()
        if os.path.exists(lbl_path):
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                class_idx = int(line.split()[0])
                                classes.add(class_idx)
                                class_counts[class_idx] += 1
                            except (ValueError, IndexError):
                                print(f"Warning: Invalid line in {lbl_path}: {line.strip()}")
            except Exception as e:
                print(f"Error reading {lbl_path}: {str(e)}")
        
        image_class_counts[img_file] = class_counts
        if 2 in classes:  # Carrier
            carrier_files.append(img_file)
        elif 0 in classes:  # Submarine (no Carrier)
            submarine_files.append(img_file)
        elif 1 in classes:  # Commercial (no Carrier or Submarine)
            commercial_files.append(img_file)
        elif 3 in classes:  # Frigate (no Carrier, Submarine, or Commercial)
            frigate_files.append(img_file)
        else:
            other_files.append(img_file)
    
    print(f"\nInitial categorization:")
    print(f"Carrier images: {len(carrier_files)}")
    print(f"Submarine images: {len(submarine_files)}")
    print(f"Commercial images: {len(commercial_files)}")
    print(f"Frigate images: {len(frigate_files)}")
    print(f"Other (no-label/empty) images: {len(other_files)}")
    
    # Initialize kept files and track total instances
    kept_files = set(carrier_files + submarine_files + other_files)
    total_counts = Counter()
    for img_file in kept_files:
        total_counts.update(image_class_counts[img_file])
    
    # Select Commercial files to reach ~commercial_limit
    commercial_candidates = sorted(commercial_files, key=lambda f: sum(1 for idx in [0, 2] if idx in image_class_counts[f]), reverse=True)
    for img_file in commercial_candidates:
        if total_counts[1] < commercial_limit:
            kept_files.add(img_file)
            total_counts.update(image_class_counts[img_file])
        else:
            break
    
    # Select Frigate files to reach ~frigate_limit
    frigate_candidates = sorted(frigate_files, key=lambda f: sum(1 for idx in [0, 1, 2] if idx in image_class_counts[f]), reverse=True)
    for img_file in frigate_candidates:
        if total_counts[3] < frigate_limit:
            kept_files.add(img_file)
            total_counts.update(image_class_counts[img_file])
        else:
            break
    
    # Move excess files to excess_dir
    excess_files = [f for f in image_files if f not in kept_files]
    for img_file in excess_files:
        base_name = os.path.splitext(img_file)[0]
        img_src = os.path.join(train_images_dir, img_file)
        lbl_src = os.path.join(train_labels_dir, f"{base_name}.txt")
        img_dst = os.path.join(excess_dir, 'images', img_file)
        lbl_dst = os.path.join(excess_dir, 'labels', f"{base_name}.txt")
        try:
            shutil.move(img_src, img_dst)
            if os.path.exists(lbl_src):
                shutil.move(lbl_src, lbl_dst)
            else:
                print(f"Note: No label file for {img_file}, moved image only")
            print(f"Moved to excess: {img_file}")
        except Exception as e:
            print(f"Error moving {img_file}: {str(e)}")
    
    # Recount class instances
    class_counts = Counter()
    total_files = 0
    for img_file in os.listdir(train_images_dir):
        if img_file.lower().endswith('.png'):
            total_files += 1
            lbl_path = os.path.join(train_labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
            if os.path.exists(lbl_path):
                try:
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    class_idx = int(line.split()[0])
                                    class_counts[class_idx] += 1
                                except (ValueError, IndexError):
                                    print(f"Warning: Invalid line in {lbl_path}: {line.strip()}")
                except Exception as e:
                    print(f"Error reading {lbl_path}: {str(e)}")
    
    # Print new class distribution
    print("\nNew Training Set Class Distribution:")
    total_instances = 0
    for class_idx in range(len(class_names)):
        count = class_counts.get(class_idx, 0)
        print(f"  {class_names[class_idx]} (index {class_idx}): {count}")
        total_instances += count
    
    print(f"\nSummary:")
    print(f"Total images in training set: {total_files}")
    print(f"Total class instances: {total_instances}")
    print(f"Excess images moved to: {excess_dir}")

def main():
    # Configuration
    train_images_dir = "D:/ShipDetection/Dataset/train/images"
    train_labels_dir = "D:/ShipDetection/Dataset/train/labels"
    excess_dir = "D:/ShipDetection/Dataset/Excess"
    classes_file = "D:/ShipDetection/classes.txt"
    
    # Verify directories
    if not os.path.exists(train_images_dir):
        print(f"Error: Training images directory {train_images_dir} does not exist")
        return
    if not os.path.exists(train_labels_dir):
        print(f"Error: Training labels directory {train_labels_dir} does not exist")
        return
    if not os.path.exists(classes_file):
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    # Balance the training set
    balance_training_set(train_images_dir, train_labels_dir, excess_dir, classes_file)

if __name__ == "__main__":
    main()