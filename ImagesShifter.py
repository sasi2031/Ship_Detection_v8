import os
import shutil

def copy_png_images(source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Counter for copied files
    copied_count = 0
    
    # Iterate through files in source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.png'):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                # Check if file already exists in destination
                if os.path.exists(dest_path):
                    print(f"Skipping: {filename} already exists in {dest_dir}")
                    continue
                
                # Copy the file
                shutil.copy(source_path, dest_path)
                print(f"Copied: {filename}")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
    
    print(f"\nCompleted: Copied {copied_count} .png files from {source_dir} to {dest_dir}")

def main():
    # Configuration
    source_dir = r"D:\ShipDetection\GE_4class_15_06_2023\Images"  # Update to your source folder
    dest_dir = r"D:\ShipDetection\Dataset\images"  # Update to your destination folder
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    # Copy the images
    copy_png_images(source_dir, dest_dir)

if __name__ == "__main__":
    main()
