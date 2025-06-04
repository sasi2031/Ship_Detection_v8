import os
import random
import shutil
from collections import Counter, defaultdict

def stratified_split_dataset(images_dir, labels_dir, dataset_dir, classes_file, train_ratio=0.8, val_ratio=0.1):
    # Create dataset directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)
    
    # Load class names
    class_names = []
    try:
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
    
    # Group images by classes (based on labels)
    class_groups = defaultdict(list)
    no_label_files = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(lbl_path):
            try:
                with open(lbl_path, 'r') as f:
                    classes = {int(line.split()[0]) for line in f if line.strip()}
                if classes:
                    # Add image to the group of each class it contains
                    for class_idx in classes:
                        class_groups[class_idx].append(img_file)
                else:
                    # Empty label file
                    no_label_files.append(img_file)
            except (ValueError, IndexError):
                print(f"Warning: Malformed label file {lbl_path}, treating as no-label")
                no_label_files.append(img_file)
        else:
            # No label file
            no_label_files.append(img_file)
    
    # Split each class group proportionally
    splits = {'train': [], 'val': [], 'test': []}
    for class_idx, files in class_groups.items():
        random.shuffle(files)
        total = len(files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        splits['train'].extend(files[:train_count])
        splits['val'].extend(files[train_count:train_count + val_count])
        splits['test'].extend(files[train_count + val_count:])
    
    # Split no-label files
    random.shuffle(no_label_files)
    total_no_label = len(no_label_files)
    train_count = int(total_no_label * train_ratio)
    val_count = int(total_no_label * val_ratio)
    splits['train'].extend(no_label_files[:train_count])
    splits['val'].extend(no_label_files[train_count:train_count + val_count])
    splits['test'].extend(no_label_files[train_count + val_count:])
    
    # Remove duplicates (images may appear in multiple class groups)
    for split in splits:
        splits[split] = list(set(splits[split]))
    
    # Copy files to respective directories and track class distribution
    split_class_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    for split, files in splits.items():
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            img_src = os.path.join(images_dir, img_file)
            lbl_src = os.path.join(labels_dir, f"{base_name}.txt")
            img_dst = os.path.join(dataset_dir, split, 'images', img_file)
            lbl_dst = os.path.join(dataset_dir, split, 'labels', f"{base_name}.txt")
            
            try:
                shutil.copy(img_src, img_dst)
                if os.path.exists(lbl_src):
                    shutil.copy(lbl_src, lbl_dst)
                    # Count classes in this label file
                    with open(lbl_src, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    class_idx = int(line.split()[0])
                                    split_class_counts[split][class_idx] += 1
                                except (ValueError, IndexError):
                                    print(f"Warning: Invalid line in {lbl_src}")
                else:
                    print(f"Warning: No label for {img_file}, creating empty label")
                    with open(lbl_dst, 'w') as f:
                        pass
                print(f"Copied {img_file} and {base_name}.txt to {split}")
            except Exception as e:
                print(f"Error copying {img_file}: {str(e)}")
    
    # Print split summary and class distribution
    print(f"\nDataset split complete:")
    print(f"Train: {len(splits['train'])} images")
    print(f"Val: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")
    
    print("\nClass distribution per split:")
    for split in splits:
        print(f"{split.capitalize()}:")
        if split_class_counts[split]:
            for class_idx in sorted(split_class_counts[split]):
                print(f"  {class_names[class_idx]} (index {class_idx}): {split_class_counts[split][class_idx]}")
        else:
            print("  No labeled instances")

def main():
    # Configuration
    images_dir = r"D:\ShipDetection\Combined4ClassDataset\images"
    labels_dir = r"D:\ShipDetection\Combined4ClassDataset\labels"
    dataset_dir = r"D:\ShipDetection\Dataset"
    classes_file = r"D:\ShipDetection\classes.txt"
    
    # Verify directories
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} does not exist")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory {labels_dir} does not exist")
        return
    if not os.path.exists(classes_file):
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    # Split dataset
    stratified_split_dataset(images_dir, labels_dir, dataset_dir, classes_file)

if __name__ == "__main__":
    main()
