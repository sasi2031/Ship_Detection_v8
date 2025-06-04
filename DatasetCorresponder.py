import os

def check_image_label_pairs(images_dir, labels_dir):
    # Get list of .png files in images folder (without extension)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.png')}
    
    # Get list of .txt files in labels folder (without extension)
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')}
    
    # Find images without corresponding labels
    images_without_labels = image_files - label_files
    
    # Find labels without corresponding images (optional, for completeness)
    labels_without_images = label_files - image_files
    
    # Print results
    print("\nChecking image-label pairs...")
    print(f"Total images (.png): {len(image_files)}")
    print(f"Total labels (.txt): {len(label_files)}")
    
    if images_without_labels:
        print("\nImages missing corresponding label files:")
        for img in sorted(images_without_labels):
            print(f"  {img}.png")
    else:
        print("\nAll images have corresponding label files.")
    
    if labels_without_images:
        print("\nLabel files missing corresponding images:")
        for lbl in sorted(labels_without_images):
            print(f"  {lbl}.txt")
    else:
        print("All label files have corresponding images.")
    
    # Summary
    print(f"\nSummary:")
    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Labels without images: {len(labels_without_images)}")

def main():
    # Configuration
    images_dir = r"D:\ShipDetection\Dataset\images"  # Update to your images folder
    labels_dir = r"D:\ShipDetection\Dataset\labels"  # Update to your labels folder
    
    # Verify directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} does not exist")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory {labels_dir} does not exist")
        return
    
    # Check image-label pairs
    check_image_label_pairs(images_dir, labels_dir)

if __name__ == "__main__":
    main()