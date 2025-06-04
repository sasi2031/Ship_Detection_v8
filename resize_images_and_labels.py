import os
import cv2
import numpy as np

# Define paths
image_input_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8\images"
label_input_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8\labels"
image_output_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_Corrected\images"
label_output_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v8_Corrected\labels"

# Target size
target_size = (512, 512)  # (width, height)

# Create output directories
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

def resize_image(image_path, output_path, target_size=(512, 512)):
    """Resize image to target_size with padding to preserve aspect ratio."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None, None, None
    orig_h, orig_w = img.shape[:2]
    
    # Calculate scaling factor to fit within target size while preserving aspect ratio
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Choose interpolation method
    if scale < 1:  # Downscaling
        interpolation = cv2.INTER_AREA
    else:  # Upscaling
        interpolation = cv2.INTER_LINEAR
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Add padding to reach target size
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # Save resized image
    cv2.imwrite(output_path, padded)
    print(f"Resized and saved: {output_path}")
    
    return scale, x_offset, y_offset, (orig_w, orig_h)

def adjust_label_file(label_path, output_path, scale, x_offset, y_offset, orig_dims, target_size):
    """Adjust label coordinates based on resizing and padding (pixel coordinates)."""
    orig_w, orig_h = orig_dims
    target_w, target_h = target_size
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:  # class_id + 8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
            print(f"Skipping invalid line in {label_path}: {line.strip()}")
            continue
        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        # Adjust coordinates: scale and apply padding offset
        new_coords = []
        for i in range(0, 8, 2):
            x = coords[i] * scale + x_offset
            y = coords[i + 1] * scale + y_offset
            # Clip to ensure coordinates are within image bounds
            x = np.clip(x, 0, target_w - 1)
            y = np.clip(y, 0, target_h - 1)
            new_coords.extend([x, y])
        
        # Format coordinates to 6 decimal places
        formatted_coords = [f"{x:.6f}" for x in new_coords]
        modified_line = f"{class_id} {' '.join(formatted_coords)}\n"
        modified_lines.append(modified_line)
    
    with open(output_path, 'w') as f:
        f.writelines(modified_lines)
    print(f"Adjusted and saved label: {output_path}")

# Process images
image_extensions = ('.jpg', '.jpeg', '.png')
for filename in os.listdir(image_input_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(image_input_dir, filename)
        output_image_path = os.path.join(image_output_dir, filename)
        
        # Load image to check dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}, skipping")
            continue
        orig_h, orig_w = img.shape[:2]
        
        # Check if resizing is needed
        if orig_w != target_size[0] or orig_h != target_size[1]:
            # Resize image and get scaling factors
            scale, x_offset, y_offset, orig_dims = resize_image(image_path, output_image_path, target_size)
            if scale is None:
                continue  # Skip if image loading failed
            
            # Adjust corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_input_dir, label_filename)
            output_label_path = os.path.join(label_output_dir, label_filename)
            
            if os.path.exists(label_path):
                adjust_label_file(label_path, output_label_path, scale, x_offset, y_offset, orig_dims, target_size)
            else:
                print(f"Label file not found for {filename}: {label_path}")
        else:
            # Copy unchanged if already 512x512
            cv2.imwrite(output_image_path, img)
            print(f"Copied unchanged image: {output_image_path}")
            
            # Copy label file unchanged
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_input_dir, label_filename)
            output_label_path = os.path.join(label_output_dir, label_filename)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read()
                with open(output_label_path, 'w') as f:
                    f.write(content)
                print(f"Copied unchanged label: {output_label_path}")
            else:
                print(f"Label file not found for {filename}: {label_path}")