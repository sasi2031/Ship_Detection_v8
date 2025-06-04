import os

def create_empty_labels(images_dir, labels_dir):
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get list of .png files in images folder (without extension)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.png')}
    
    # Get list of .txt files in labels folder (without extension)
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')}
    
    # Find images without corresponding labels
    images_without_labels = image_files - label_files
    
    # Counter for created files
    created_count = 0
    
    # Create empty .txt files for unmatched images
    for base_name in images_without_labels:
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        try:
            # Create empty file
            with open(label_path, 'w') as f:
                pass  # Empty file
            print(f"Created: {base_name}.txt")
            created_count += 1
        except Exception as e:
            print(f"Error creating {base_name}.txt: {str(e)}")
    
    print(f"\nCompleted: Created {created_count} empty .txt files in {labels_dir}")

def main():
    # Configuration
    images_dir = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\val\images" # Update to your images folder
    labels_dir = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\val\labels"  # Update to your labels folder
    
    # Verify directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} does not exist")
        return
    
    # Create empty label files
    create_empty_labels(images_dir, labels_dir)

if __name__ == "__main__":
    main()