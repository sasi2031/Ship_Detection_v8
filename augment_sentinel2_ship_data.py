import os
import cv2
import numpy as np
import albumentations as A

# Define paths
train_image_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split\train\images" # Replace with your train images folder
train_label_dir =r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split\train\labels" # Replace with your train labels folder
aug_image_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split_aug\train\images" # Output folder for augmented images
aug_label_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split_aug\train\labels"  # Output folder for augmented labels

debug_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split_aug\Debug"
target_size = (512, 512)
num_augmentations = 11  # For 8,784 total images (732 * (1 + 11))

# Create output directories
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# Define augmentation pipeline with keypoints
augmentations = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.VerticalFlip(p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
], keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'], remove_invisible=False))

def reorder_coordinates(points):
    """Reorder clockwise points (top-left start) to counterclockwise (bottom-left start)."""
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    # Counterclockwise from bottom-left: (x4, y4) -> (x3, y3) -> (x2, y2) -> (x1, y1)
    return [x4, y4, x3, y3, x2, y2, x1, y1]

def read_labels(label_path, img_w, img_h):
    """Read pixel-coordinate OBB labels and prepare keypoints."""
    labels = []
    keypoints = []
    keypoint_labels = []
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}, treating as empty")
        return labels, keypoints, keypoint_labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                print(f"Skipping invalid line in {label_path}: {line.strip()}")
                continue
            class_id = int(parts[0])
            points = [float(x) for x in parts[1:]]
            # Reorder to counterclockwise starting from bottom-left
            reordered_points = reorder_coordinates(points)
            # Convert to keypoints (x, y pairs)
            kp = [(reordered_points[i], reordered_points[i + 1]) for i in range(0, 8, 2)]
            labels.append({
                'class_id': class_id,
                'points': reordered_points
            })
            keypoints.extend(kp)
            keypoint_labels.extend([class_id] * 4)  # One class ID per keypoint
    return labels, keypoints, keypoint_labels

def write_labels(labels, output_path):
    """Write pixel-coordinate OBB labels."""
    with open(output_path, 'w') as f:
        for label in labels:
            class_id = label['class_id']
            points = label['points']
            formatted_points = [f"{x:.6f}" for x in points]
            f.write(f"{class_id} {' '.join(formatted_points)}\n")

def draw_obb(image, labels, color=(0, 255, 0), thickness=2):
    """Draw OBBs on image for visualization."""
    img = image.copy()
    for label in labels:
        points = label['points']
        pts = [(int(points[i]), int(points[i + 1])) for i in range(0, 8, 2)]
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img

def visualize_images(orig_image, orig_labels, aug_image, aug_labels, filename, aug_idx=None):
    """Save original and augmented images with OBBs for debugging."""
    orig_viz = draw_obb(orig_image, orig_labels)
    orig_viz_path = os.path.join(debug_dir, f"orig_{filename}")
    cv2.imwrite(orig_viz_path, cv2.cvtColor(orig_viz, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization: {orig_viz_path}")
    if aug_image is not None and aug_labels is not None:
        aug_viz = draw_obb(aug_image, aug_labels)
        aug_viz_filename = f"{os.path.splitext(filename)[0]}_aug{aug_idx}{os.path.splitext(filename)[1]}"
        aug_viz_path = os.path.join(debug_dir, aug_viz_filename)
        cv2.imwrite(aug_viz_path, cv2.cvtColor(aug_viz, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization: {aug_viz_path}")

# Process training images and labels
image_extensions = ('.jpg', '.jpeg', '.png')
for filename in os.listdir(train_image_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(train_image_dir, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(train_label_dir, label_filename)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        
        # Read labels
        labels, keypoints, keypoint_labels = read_labels(label_path, img_w, img_h)
        
        # Visualize original image
        visualize_images(image, labels, None, None, filename)
        
        # Save original image and label
        orig_image_path = os.path.join(aug_image_dir, f"orig_{filename}")
        orig_label_path = os.path.join(aug_label_dir, f"orig_{label_filename}")
        cv2.imwrite(orig_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        write_labels(labels, orig_label_path)
        print(f"Copied original image and label: orig_{filename}, orig_{label_filename}")
        
        # Generate augmentations
        for aug_idx in range(num_augmentations):
            # Apply augmentations with keypoints
            augmented = augmentations(
                image=image,
                keypoints=keypoints,
                keypoint_labels=keypoint_labels
            )
            aug_image = augmented['image']
            aug_keypoints = augmented['keypoints']
            
            # Reconstruct labels from keypoints
            aug_labels = []
            kp_idx = 0
            for _ in range(len(labels)):
                if kp_idx + 4 > len(aug_keypoints):
                    print(f"Skipping bbox due to insufficient keypoints in {filename}_aug{aug_idx}")
                    continue
                kp = aug_keypoints[kp_idx:kp_idx + 4]
                points = []
                for x, y in kp:
                    # Clip to image bounds (pixel coordinates)
                    x = np.clip(x, 0, target_size[0] - 1)
                    y = np.clip(y, 0, target_size[1] - 1)
                    points.extend([x, y])
                kp_idx += 4
                aug_labels.append({
                    'class_id': labels[kp_idx//4 - 1]['class_id'],
                    'points': points
                })
            
            # Save augmented image
            aug_image_filename = f"{os.path.splitext(filename)[0]}_aug{aug_idx}{os.path.splitext(filename)[1]}"
            aug_image_path = os.path.join(aug_image_dir, aug_image_filename)
            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            print(f"Saved augmented image: {aug_image_path}")
            
            # Save augmented labels
            aug_label_filename = f"{os.path.splitext(label_filename)[0]}_aug{aug_idx}.txt"
            aug_label_path = os.path.join(aug_label_dir, aug_label_filename)
            write_labels(aug_labels, aug_label_path)
            print(f"Saved augmented label: {aug_label_path}")
            
            # Visualize augmented image
            visualize_images(image, labels, aug_image, aug_labels, filename, aug_idx)