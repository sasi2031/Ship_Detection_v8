import os

def extract_class_names_from_labels(labels_dir, output_file='classes.txt'):
    """
    Extracts unique class names from label files in the specified directory.
    
    Args:
        labels_dir (str): Path to the directory containing label files.
        
    Returns:
        set: A set of unique class names.
    """
    class_names = set()
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 10:
                            class_name = parts[-2]
                            class_names.add(class_name)
    with open(output_file, 'w') as f:
        for class_name in sorted(class_names):
            f.write(f"{class_name}\n")
    print(f"Class names extracted and saved to {output_file}")


if __name__ == "__main__":
    labels_dir = r"D:\Ship_Detection_Data_Sentinel1\Dataset_v5\labels"
    output_file = r"D:\Ship_Detection_Data_Sentinel1\classes.txt"
    extract_class_names_from_labels(labels_dir, output_file)