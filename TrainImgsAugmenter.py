import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

# Define class counts from your dataset
class_counts = {
    "Small_Ship": 884,
    "Tugs": 5653,
    "Commercial_Tanker": 1098,
    "Commercial_Cargo": 6833,
    "Fleet_Replenishment_Ship": 82,
    "Tariq_Class_Frigate": 731,
    "Naval_Auxiliary": 224,
    "Agosta_70_Submarine": 480,
    "Oliver_Hazard_Perry_Class_Frigate": 140,
    "Munsif_Class_Mine_Hunter": 705,
    "Naval_Tanker": 764,
    "Moawin_Replenishment_oiler": 79,
    "Poolster_Class": 76,
    "Rah_Naward_Sail_Training_Ship": 177,
    "Utility_Ship": 293,
    "Jalalat_Class_Missile_Boat": 765,
    "MRTP_Class_Fast_Patrol_Craft": 450,
    "Azmat_Class_Fast_Attack_Craft": 483,
    "Coastal_Tanker": 70,
    "Zulfiquar_Class_Frigate": 650,
    "Fuqing_Class_Replenishment_Oiler": 181,
    "Behar_Paima_Hydrographic_Survey_Ship": 170,
    "Behar_Masah_Hydrographic_Survey_Ship": 90,
    "PNS_Rizwan": 5,
    "Tughril_Class_Frigate": 64,
    "Yarmook_Class_Corvette": 137,
    "Babur_Class_Corvette": 172,
    "Larkana_Class": 2,
    "Hingol_Class_vessel": 550,
    "Barkat_Class": 429,
    "Patrol_Forces": 13,
    "Survey_Ship": 4,
    "Small_Transport": 9,
    "Agosta_90B_Submarine": 4
}

# Define augmentation factors based on class counts
def get_augmentation_factor(class_name):
    count = class_counts.get(class_name, 0)
    if count < 50:
        return 50
    elif count < 100:
        return 20
    elif count < 500:
        return 5
    else:
        return 0

# Define individual augmentation pipelines with keypoint_params for OBBs
augmentation_pipelines = [
    ("rotate", A.Compose([A.Rotate(limit=30, p=1.0, border_mode=cv2.BORDER_CONSTANT)], 
                        keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("hflip", A.Compose([A.HorizontalFlip(p=1.0)], 
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("vflip", A.Compose([A.VerticalFlip(p=1.0)], 
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("brightness_contrast", A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)], 
                                    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("noise", A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)], 
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("scale", A.Compose([A.RandomScale(scale_limit=0.2, p=1.0)], 
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("shear", A.Compose([A.Affine(shear=(-10, 10), p=1.0)], 
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("hue_saturation", A.Compose([A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=1.0)], 
                                keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
    ("blur", A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)], 
                      keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))),
]

# Function to validate and clip keypoints
def validate_and_clip_keypoints(bboxes, image_shape, class_labels):
    height, width = image_shape[:2]
    valid_bboxes = []
    valid_class_labels = []
    
    for bbox, cls in zip(bboxes, class_labels):
        coords = np.array(bbox)
        if not np.all(np.isfinite(coords)):
            print(f"Invalid coordinates (non-finite) in bbox for {cls}: {bbox}")
            continue
        
        # Clip coordinates to image bounds
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)  # x
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)  # y
        
        # Relaxed validation: allow small bboxes if they form a polygon
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
        if x_range > 1e-3 and y_range > 1e-3:  # Allow small but non-zero area
            valid_bboxes.append(coords.tolist())
            valid_class_labels.append(cls)
        else:
            print(f"Invalid bbox (near-collapsed, x_range={x_range}, y_range={y_range}) for {cls}: {bbox}")
    
    return valid_bboxes, valid_class_labels

# Function to read OBB labels
def read_obb_label(label_path):
    bboxes = []
    class_labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    print(f"Invalid label format in {label_path}: {line.strip()}")
                    continue
                try:
                    coords = list(map(float, parts[:8]))
                    # Pre-clip coordinates
                    coords = np.array(coords).reshape(4, 2)
                    coords[:, 0] = np.clip(coords[:, 0], 0, 1023)  # x
                    coords[:, 1] = np.clip(coords[:, 1], 0, 1023)  # y
                    poly = coords.tolist()
                    bboxes.append(poly)
                    class_labels.append(parts[8])
                except ValueError:
                    print(f"Invalid coordinate values in {label_path}: {line.strip()}")
                    continue
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
    return bboxes, class_labels

# Function to write OBB labels
def write_obb_label(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            coords = [coord for point in bbox for coord in point]
            f.write(f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f} "
                    f"{coords[4]:.6f} {coords[5]:.6f} {coords[6]:.6f} {coords[7]:.6f} {cls}\n")

# Main augmentation function
def augment_dataset(image_dir, label_dir, aug_image_dir, aug_label_dir):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    aug_image_dir = Path(aug_image_dir)
    aug_label_dir = Path(aug_label_dir)
    
    aug_image_dir.mkdir(parents=True, exist_ok=True)
    aug_label_dir.mkdir(parents=True, exist_ok=True)
    
    for label_file in label_dir.glob('*.txt'):
        image_name = label_file.stem + '.png'
        image_path = image_dir / image_name
        
        if not image_path.exists():
            print(f"Image {image_path} not found, skipping.")
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image {image_path}, skipping.")
            continue
        
        bboxes, class_labels = read_obb_label(label_file)
        if not bboxes:
            print(f"No valid labels in {label_file}, skipping.")
            continue
        
        # Validate and clip keypoints
        bboxes, class_labels = validate_and_clip_keypoints(bboxes, image.shape, class_labels)
        if not bboxes:
            print(f"No valid bboxes after validation in {label_file}, skipping.")
            continue
        
        max_aug_factor = max(get_augmentation_factor(cls) for cls in class_labels)
        if max_aug_factor == 0:
            continue
        
        num_techniques = len(augmentation_pipelines)
        cycles = max(1, max_aug_factor // num_techniques)
        
        for cycle in range(cycles):
            for aug_name, pipeline in augmentation_pipelines:
                # Adjust parameters for variation across cycles
                if aug_name == "rotate":
                    angle = 15 if cycle % 2 == 0 else -15
                    pipeline = A.Compose([A.Rotate(limit=angle, p=1.0, border_mode=cv2.BORDER_CONSTANT)], 
                                        keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))
                elif aug_name == "scale":
                    scale = 0.1 if cycle % 2 == 0 else -0.1
                    pipeline = A.Compose([A.RandomScale(scale_limit=scale, p=1.0)], 
                                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))
                
                try:
                    augmented = pipeline(image=image, keypoints=sum(bboxes, []), class_labels=sum([[cls] * 4 for cls in class_labels], []))
                    aug_image = augmented['image']
                    aug_bboxes = augmented['keypoints']
                    aug_class_labels = augmented['class_labels']
                    
                    # Reconstruct bboxes, ensuring alignment
                    if len(aug_bboxes) % 4 != 0 or len(aug_bboxes) // 4 != len(aug_class_labels) // 4:
                        print(f"Mismatch in keypoints and labels for {image_name} with {aug_name} (cycle {cycle})")
                        continue
                    
                    reconstructed_bboxes = [aug_bboxes[j:j+4] for j in range(0, len(aug_bboxes), 4)]
                    reconstructed_class_labels = [aug_class_labels[j*4] for j in range(len(reconstructed_bboxes))]
                    
                    # Clip and validate augmented keypoints
                    reconstructed_bboxes, reconstructed_class_labels = validate_and_clip_keypoints(
                        reconstructed_bboxes, aug_image.shape, reconstructed_class_labels)
                    if not reconstructed_bboxes:
                        print(f"No valid bboxes after augmentation for {image_name} with {aug_name} (cycle {cycle})")
                        continue
                    
                    aug_image_name = f"{image_name.split('.')[0]}_{aug_name}_cycle{cycle}.png"
                    aug_image_path = aug_image_dir / aug_image_name
                    cv2.imwrite(str(aug_image_path), aug_image)
                    
                    aug_label_name = f"{label_file.stem}_{aug_name}_cycle{cycle}.txt"
                    aug_label_path = aug_label_dir / aug_label_name
                    write_obb_label(aug_label_path, reconstructed_bboxes, reconstructed_class_labels)
                    print(f"Saved augmented image {aug_image_path} and label {aug_label_path}")
                except Exception as e:
                    print(f"Error augmenting {image_name} with {aug_name} (cycle {cycle}): {e}")
                    continue


# Run augmentation
if __name__ == "__main__":
    image_dir = r"D:\Ship_Detection_Data\SplittedDataset\images\train"  # Path to original images
    label_dir = r"D:\Ship_Detection_Data\SplittedDataset\labels\train"  # Path to original labels
    aug_image_dir = r"D:\Ship_Detection_Data\SplittedDataset\images\trainImgs_aug"  # Path to save augmented images
    aug_label_dir = r"D:\Ship_Detection_Data\SplittedDataset\labels\trainLbls_aug"  # Path to save augmented labels
    augment_dataset(image_dir, label_dir, aug_image_dir, aug_label_dir)