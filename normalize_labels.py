import os
import numpy as np

# Define paths
label_input_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split_aug\val\labels"  # Update with your labels folder path
label_output_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_split_aug\val\normalzied_labels"  # Update with your destination folder path

# Image dimensions (for normalization)
image_width = 512
image_height = 512

# Create output directory
os.makedirs(label_output_dir, exist_ok=True)

def normalize_coordinates(points, width, height):
    """Normalize pixel coordinates to [0, 1] and ensure they stay within bounds."""
    norm_points = []
    for i in range(0, 8, 2):
        x = points[i] / width
        y = points[i + 1] / height
        # Explicitly clip to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        norm_points.extend([x, y])
    return norm_points

# Process each label file
for label_file in os.listdir(label_input_dir):
    if not label_file.endswith(".txt"):
        continue
    
    label_path = os.path.join(label_input_dir, label_file)
    output_path = os.path.join(label_output_dir, label_file)
    
    # Read and normalize labels
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    norm_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            print(f"Skipping invalid line in {label_file}: {line.strip()}")
            continue
        class_id = parts[0]
        points = [float(x) for x in parts[1:]]
        norm_points = normalize_coordinates(points, image_width, image_height)
        formatted_points = [f"{x:.6f}" for x in norm_points]
        norm_line = f"{class_id} {' '.join(formatted_points)}\n"
        norm_lines.append(norm_line)
    
    # Save normalized labels
    with open(output_path, "w") as f:
        f.writelines(norm_lines)
    print(f"Normalized and saved: {output_path}")

print("Normalization complete.")