import os
import cv2
from pathlib import Path

# Define paths
label_dir =r"D:\Ship_Detection_Data_Sentinel1\Dataset_v5\labels" # YOLOv5 OBB labels
image_dir =r"D:\Ship_Detection_Data_Sentinel1\Dataset_v5\images"# Corresponding images
output_dir = r"D:\Ship_Detection_Data_Sentinel1\Dataset_v5\labels_v8"  # YOLOv8 OBB labels

classes_file = r"D:\Ship_Detection_Data_Sentinel1\classes.txt"  # Classes file

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Read classes.txt to create class mapping
class_mapping = {}
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]
    for idx, class_name in enumerate(class_names):
        class_mapping[class_name] = idx

def reorder_coordinates(points):
    """Reorder clockwise points (top-left start) to counterclockwise (bottom-left start)."""
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    # Counterclockwise from bottom-left: (x4, y4) -> (x3, y3) -> (x2, y2) -> (x1, y1)
    reordered = [x4, y4, x3, y3, x2, y2, x1, y1]
    return reordered

def normalize_coordinates(points, img_width, img_height):
    """Normalize pixel coordinates to [0, 1]."""
    norm_points = []
    for i in range(0, 8, 2):
        x = points[i] / img_width
        y = points[i + 1] / img_height
        norm_points.extend([x, y])
    return norm_points

# Process each label file
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    
    label_path = os.path.join(label_dir, label_file)
    image_path = os.path.join(image_dir, f"{Path(label_file).stem}.png")  # Adjust extension if needed
    output_path = os.path.join(output_dir, label_file)
    
    # Load image to get dimensions
    if not os.path.exists(image_path):
        print(f"Image not found for {label_file}, skipping normalization (assuming pixel coordinates)")
        continue
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}, skipping")
        continue
    img_height, img_width = image.shape[:2]
    
    # Read and convert labels
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    v8_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 10:
            print(f"Skipping invalid line in {label_file}: {line.strip()}")
            continue
        
        # Extract coordinates, class name, and difficulty
        points = [float(x) for x in parts[:8]]
        class_name = parts[8]
        # difficulty = parts[9]  # Ignored for YOLOv8
        
        # Map class name to class ID
        if class_name not in class_mapping:
            print(f"Class {class_name} not in class_mapping, skipping")
            continue
        class_id = class_mapping[class_name]
        
        # Reorder coordinates to counterclockwise starting from bottom-left
        reordered_points = reorder_coordinates(points)
        
        # Normalize coordinates
        norm_points = normalize_coordinates(reordered_points, img_width, img_height)
        formatted_points = [f"{x:.6f}" for x in norm_points]
        
        # Write in YOLOv8 OBB format
        v8_lines.append(f"{class_id} {' '.join(formatted_points)}\n")
    
    # Save YOLOv8 labels
    with open(output_path, "w") as f:
        f.writelines(v8_lines)
    print(f"Converted {label_file} to YOLOv8 OBB format")

print("Conversion complete.")