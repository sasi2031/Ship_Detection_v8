import os
import random
import shutil
from collections import defaultdict

def split_dataset(image_folder, label_folder, output_dir, image_ext='.png'):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    # Step 1: Group label files by class
    class_to_files = defaultdict(list)

    for label_file in os.listdir(label_folder):
        if not label_file.endswith('.txt'):
            continue
        base_name = os.path.splitext(label_file)[0]
        image_file = base_name + image_ext
        label_path = os.path.join(label_folder, label_file)

        classes_in_file = set()
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:
                    cls = parts[-2]
                    classes_in_file.add(cls)

        # Use primary class (first one found), or fallback to "Unknown"
        primary_class = next(iter(classes_in_file), "Unknown")
        class_to_files[primary_class].append(base_name)

    print("Class distribution:")
    for cls, files in class_to_files.items():
        print(f" - {cls}: {len(files)} files")

    # Step 2: Decide train/val split per class
    train_files = set()
    val_files = set()

    for cls, files in class_to_files.items():
        if len(files) <= 4:
            # All in train
            train_files.update(files)
        else:
            # 80/20 split
            split_idx = int(len(files) * 0.8)
            random.shuffle(files)
            train_files.update(files[:split_idx])
            val_files.update(files[split_idx:])

    print(f"\nTotal train files: {len(train_files)}")
    print(f"Total val files: {len(val_files)}")

    # Step 3: Copy files
    def copy_files(file_set, src_img, src_lbl, dst_img, dst_lbl):
        for base_name in file_set:
            img_src = os.path.join(src_img, base_name + image_ext)
            lbl_src = os.path.join(src_lbl, base_name + '.txt')
            img_dst = os.path.join(dst_img, base_name + image_ext)
            lbl_dst = os.path.join(dst_lbl, base_name + '.txt')

            if os.path.exists(img_src) and os.path.exists(lbl_src):
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)

    print("\nCopying files...")
    copy_files(
        train_files,
        image_folder, label_folder,
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'labels', 'train')
    )

    copy_files(
        val_files,
        image_folder, label_folder,
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'val')
    )

    print("✅ Dataset split complete.")

# Example usage
image_folder = r"D:\Ship_Detection_Data\Dataset_v5\images"
label_folder = r"D:\Ship_Detection_Data\Dataset_v5\labels"
output_dir = r"D:\Ship_Detection_Data\SplittedDataset"

split_dataset(image_folder, label_folder, output_dir)