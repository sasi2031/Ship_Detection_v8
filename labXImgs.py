import os

def remove_unmatched_labels(images_dir, labels_dir):
    # Get list of .png files in images folder (without extension)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.png')}
    
    # Get list of .txt files in labels folder
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.xml')]
    
    # Counter for deleted files
    deleted_count = 0
    
    # Check each label file
    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        if base_name not in image_files:
            # label_path = os.path.join(labels_dir, label_file)
            try:
                # os.remove(label_path)
                print(f"Deleted: {label_file}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {label_file}: {str(e)}")
    
    print(f"\nCompleted: Deleted {deleted_count} unmatched .txt files from {labels_dir}")

def main():
    # Configuration
    images_dir = r"D:\Ship_Detection_Data_12Classes\GE_Highres\images" # Update to your images folder
    labels_dir = r"D:\Ship_Detection_Data_12Classes\GE_Highres\labels"  # Update to your labels folder
    
    # Verify directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} does not exist")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory {labels_dir} does not exist")
        return
    
    # Remove unmatched label files
    remove_unmatched_labels(images_dir, labels_dir)

if __name__ == "__main__":
    main()