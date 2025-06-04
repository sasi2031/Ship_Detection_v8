import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
image_input_dir = r"D:\Ship_Detection_Data_Sentinel1\Dataset_v5\images" # Replace with your image folder path
label_input_dir = r"D:\Ship_Detection_Data_Sentinel1\Dataset_v5\labels_v8_Denorm"  # Replace with your label folder path
output_dir = r"D:\Ship_Detection_Data_Sentinel1\Dataset_v8_split" # Replace with output directory for train/val folders

# Create output directories
train_image_dir = os.path.join(output_dir, "train", "images")
train_label_dir = os.path.join(output_dir, "train", "labels")
val_image_dir = os.path.join(output_dir, "val", "images")
val_label_dir = os.path.join(output_dir, "val", "labels")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get list of image files
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(image_input_dir) if f.lower().endswith(image_extensions)]

# Split dataset into train and validation sets (80:20)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

def copy_files(file_list, src_image_dir, src_label_dir, dst_image_dir, dst_label_dir):
    """Copy images and their corresponding labels to destination folders."""
    for filename in file_list:
        # Copy image
        src_image_path = os.path.join(src_image_dir, filename)
        dst_image_path = os.path.join(dst_image_dir, filename)
        shutil.copy2(src_image_path, dst_image_path)
        print(f"Copied image: {dst_image_path}")
        
        # Copy corresponding label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        src_label_path = os.path.join(src_label_dir, label_filename)
        dst_label_path = os.path.join(dst_label_dir, label_filename)
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
            print(f"Copied label: {dst_label_path}")
        else:
            print(f"Label file not found for {filename}: {src_label_path}")

# Copy training files
copy_files(train_files, image_input_dir, label_input_dir, train_image_dir, train_label_dir)

# Copy validation files
copy_files(val_files, image_input_dir, label_input_dir, val_image_dir, val_label_dir)

print(f"Dataset split complete: {len(train_files)} training files, {len(val_files)} validation files")