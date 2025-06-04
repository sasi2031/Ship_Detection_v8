import os
import shutil

def collect_images_and_labels(source_dir, dest_dir):
    
    dest_imgs= os.path.join(dest_dir, 'images')
    dest_lbls= os.path.join(dest_dir, 'labels')

    # Create destination directories if they don't exist
    os.makedirs(dest_imgs, exist_ok=True)
    os.makedirs(dest_lbls, exist_ok=True)

    # Iterate through all files in the source directory
    for root, dirs, files in os.walk(source_dir):
        if os.path.basename(root).lower() == 'images':
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_imgs, file)
                shutil.copy2(src_file, dest_file)
                print(f"Copied image: {src_file} to {dest_file}")
        elif os.path.basename(root).lower() == 'labels':
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_lbls, file)
                shutil.copy2(src_file, dest_file)
                print(f"Copied label: {src_file} to {dest_file}")


source_dir = r"D:\Ship_Detection_Data\Final_Pakistan\Normal"
dest_dir = r"D:\Ship_Detection_Data\Dataset_v5"

if __name__ == "__main__":
    collect_images_and_labels(source_dir, dest_dir)