import os
import cv2

# Set paths
label_folder = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\labels_v8_0"    # Folder containing normalized .txt files
image_folder = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\images"  # Folder containing corresponding .png/.jpg images
output_folder = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\lables_v8_Denorm"      # Folder to save denormalized labels
overwrite = False                                 # Set True to overwrite original files

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Loop through all label files
for label_file in os.listdir(label_folder):
    if not label_file.endswith(".txt"):
        continue

    image_file = os.path.splitext(label_file)[0] + ".png"
    image_path = os.path.join(image_folder, image_file)

    # Load image to get its size
    if not os.path.exists(image_path):
        print(f"Image not found for {label_file}, skipping...")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    height, width = image.shape[:2]

    # Read label file
    label_path = os.path.join(label_folder, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    denormalized_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            print(f"Skipping invalid line in {label_file}: {line}")
            continue

        class_id = parts[0]
        coords = list(map(float, parts[1:]))  # 8 normalized coordinates

        # Denormalize
        denorm_coords = []
        for i in range(0, 8, 2):
            x = int(coords[i] * width)
            y = int(coords[i+1] * height)
            denorm_coords.extend([x, y])

        # Build new line
        new_line = f"{class_id} {' '.join(map(str, denorm_coords))}\n"
        denormalized_lines.append(new_line)

    # Save to output folder
    output_path = os.path.join(output_folder, label_file)
    with open(output_path, 'w') as f:
        f.writelines(denormalized_lines)

    print(f"Processed: {label_file}")

print("✅ Denormalization complete.")