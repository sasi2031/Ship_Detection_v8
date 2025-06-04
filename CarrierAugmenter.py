import os
import shutil
import cv2
import numpy as np
import albumentations as A
from collections import Counter

def validate_and_fix_labels(label_path, image_size=1024):
    """Validate and fix labels in format: class_id x1 y1 x2 y2 x3 y3 x4 y4."""
    fixed_lines = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    print(f"Warning: Skipping invalid line in {label_path}: {line.strip()}")
                    continue
                class_idx = int(float(parts[0]))  # Handle float or int class ID
                coords = [float(x) for x in parts[1:9]]
                # Clip coordinates to [0.0, 1.0]
                for i in range(8):
                    coords[i] = min(max(coords[i], 0.0), 1.0)
                # Validate non-degenerate polygon
                points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                if max(x_coords) - min(x_coords) < 1.0 / image_size or max(y_coords) - min(y_coords) < 1.0 / image_size:
                    print(f"Warning: Degenerate polygon in {label_path}, adjusting")
                    x_mean = sum(x_coords) / 4
                    y_mean = sum(y_coords) / 4
                    delta = 1.0 / image_size
                    coords = [x_mean - delta, y_mean - delta, x_mean + delta, y_mean - delta,
                              x_mean + delta, y_mean + delta, x_mean - delta, y_mean + delta]
                    for i in range(8):
                        coords[i] = min(max(coords[i], 0.0), 1.0)
                fixed_lines.append(f"{class_idx} {' '.join(map(str, coords))}")
        if fixed_lines:
            with open(label_path, 'w') as f:
                for line in fixed_lines:
                    f.write(f"{line}\n")
            return True
        print(f"Warning: No valid labels in {label_path}, skipping")
        return False
    except Exception as e:
        print(f"Error processing {label_path}: {str(e)}")
        return False

def obb_to_aabb(coords, image_size=1024):
    """Convert OBB vertices to axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
    x_coords = [coords[i] for i in range(0, 8, 2)]
    y_coords = [coords[i+1] for i in range(0, 8, 2)]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    if x_max <= x_min:
        x_max = x_min + 1.0 / image_size
    if y_max <= y_min:
        y_max = y_min + 1.0 / image_size
    return [x_min, y_min, x_max, y_max]

def aabb_to_obb(aabb, original_coords, image_size=1024):
    """Reconstruct OBB vertices from AABB, preserving original orientation."""
    x_min, y_min, x_max, y_max = aabb
    original_points = [(original_coords[i], original_coords[i+1]) for i in range(0, 8, 2)]
    original_x_min, original_x_max = min(p[0] for p in original_points), max(p[0] for p in original_points)
    original_y_min, original_y_max = min(p[1] for p in original_points), max(p[1] for p in original_points)
    
    if original_x_max - original_x_min < 1.0 / image_size or original_y_max - original_y_min < 1.0 / image_size:
        delta = (x_max - x_min) / 2
        return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
    
    x_scale = (x_max - x_min) / (original_x_max - original_x_min)
    y_scale = (y_max - y_min) / (original_y_max - original_y_min)
    new_points = []
    for x, y in original_points:
        x_new = x_min + (x - original_x_min) * x_scale
        y_new = y_min + (y - original_y_min) * y_scale
        x_new = min(max(x_new, 0.0), 1.0)
        y_new = min(max(y_new, 0.0), 1.0)
        new_points.extend([x_new, y_new])
    return new_points

def augment_carrier(train_images_dir, train_labels_dir, excess_dir, classes_file, augment_count=7, commercial_limit=2500, frigate_limit=2750, image_size=1024):
    os.makedirs(os.path.join(excess_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(excess_dir, 'labels'), exist_ok=True)
    
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
    
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.RandomScale(scale_limit=0.05, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_area=0.0, min_visibility=0.0))
    
    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')]
    carrier_files = []
    for img_file in image_files:
        lbl_path = os.path.join(train_labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                if any(line.strip().startswith('2 ') for line in f):
                    if validate_and_fix_labels(lbl_path, image_size):
                        carrier_files.append(img_file)
                    else:
                        print(f"Skipping {img_file} due to invalid label file")
    
    print(f"\nFound {len(carrier_files)} valid images containing Carrier")
    
    for img_file in carrier_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(train_images_dir, img_file)
        lbl_path = os.path.join(train_labels_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image {img_path}")
            continue
        
        bboxes = []
        class_labels = []
        original_coords = []
        try:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:
                        class_labels.append(int(float(parts[0])))  # Ensure integer class ID
                        coords = [float(x) for x in parts[1:9]]
                        bboxes.append(obb_to_aabb(coords, image_size))
                        original_coords.append(coords)
        except Exception as e:
            print(f"Error reading {lbl_path}: {str(e)}")
            continue
        
        successful_augs = 0
        attempts = 0
        max_attempts = augment_count * 2
        while successful_augs < augment_count and attempts < max_attempts:
            try:
                augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = [int(float(cls)) for cls in augmented['class_labels']]  # Convert to int
                aug_orig_coords = [orig for orig, cls in zip(original_coords, class_labels) if cls in aug_labels]
                
                if not aug_bboxes:
                    print(f"Warning: No valid bboxes after augmentation for {img_file}, attempt {attempts}, skipping")
                    attempts += 1
                    continue
                
                aug_img_path = os.path.join(train_images_dir, f"{base_name}_carrier_aug{successful_augs}.png")
                aug_lbl_path = os.path.join(train_labels_dir, f"{base_name}_carrier_aug{successful_augs}.txt")
                cv2.imwrite(aug_img_path, aug_img)
                with open(aug_lbl_path, 'w') as f:
                    for cls, bbox, orig in zip(aug_labels, aug_bboxes, aug_orig_coords):
                        obb_coords = aabb_to_obb(bbox, orig, image_size)
                        f.write(f"{cls} {' '.join(map(str, obb_coords))}\n")  # cls is already int
                print(f"Generated augmented image: {aug_img_path}")
                successful_augs += 1
                attempts += 1
            except Exception as e:
                print(f"Error during augmentation of {img_file}, attempt {attempts}: {str(e)}")
                attempts += 1
                continue
        
        if successful_augs < augment_count:
            print(f"Warning: Only {successful_augs} successful augmentations for {img_file} after {max_attempts} attempts")
    
    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')]
    image_class_counts = {}
    for img_file in image_files:
        lbl_path = os.path.join(train_labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        class_counts = Counter()
        if os.path.exists(lbl_path):
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                class_idx = int(float(line.split()[0]))  # Handle float or int class ID
                                class_counts[class_idx] += 1
                            except (ValueError, IndexError):
                                print(f"Warning: Invalid line in {lbl_path}: {line.strip()}")
            except Exception as e:
                print(f"Error reading {lbl_path}: {str(e)}")
        image_class_counts[img_file] = class_counts
    
    total_counts = Counter()
    kept_files = set(image_files)
    for img_file in image_files:
        total_counts.update(image_class_counts[img_file])
    
    if total_counts[1] > commercial_limit or total_counts[3] > frigate_limit:
        priority_files = [f for f in image_files if 2 in image_class_counts[f] or 0 in image_class_counts[f]]
        other_files = [f for f in image_files if f not in priority_files]
        other_files.sort(key=lambda f: sum(image_class_counts[f][idx] for idx in [0, 2]))
        kept_files = set(priority_files)
        temp_counts = Counter()
        for f in priority_files:
            temp_counts.update(image_class_counts[f])
        
        for img_file in other_files:
            new_counts = temp_counts + image_class_counts[img_file]
            if new_counts[1] <= commercial_limit and new_counts[3] <= frigate_limit:
                kept_files.add(img_file)
                temp_counts.update(image_class_counts[img_file])
            else:
                break
        
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
                                    class_idx = int(float(line.split()[0]))  # Handle float or int class ID
                                    class_counts[class_idx] += 1
                                except (ValueError, IndexError):
                                    print(f"Warning: Invalid line in {lbl_path}: {line.strip()}")
                except Exception as e:
                    print(f"Error reading {lbl_path}: {str(e)}")
    
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
    train_images_dir = "D:/ShipDetection/Dataset/train/images"
    train_labels_dir = "D:/ShipDetection/Dataset/train/labels"
    excess_dir = "D:/ShipDetection/Dataset/Excess"
    classes_file = "D:/ShipDetection/classes.txt"
    
    if not os.path.exists(train_images_dir):
        print(f"Error: Training images directory {train_images_dir} does not exist")
        return
    if not os.path.exists(train_labels_dir):
        print(f"Error: Training labels directory {train_labels_dir} does not exist")
        return
    if not os.path.exists(classes_file):
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    augment_carrier(train_images_dir, train_labels_dir, excess_dir, classes_file)

if __name__ == "__main__":
    main()