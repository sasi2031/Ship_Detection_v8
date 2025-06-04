import os
from collections import Counter

# def count_class_instances(folder_path):
#     # Dictionary to store class counts
#     class_counts = Counter()

#     # Iterate through all files in the folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'):  # Assuming label files are .txt
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'r') as file:
#                 # Read each line in the file
#                 for line in file:
#                     # Split the line and extract the class name (second-to-last element)
#                     parts = line.strip().split()
#                     if len(parts) >= 2:  # Ensure there are enough parts
#                         class_name = parts[-1]  # Class name is second-to-last
#                         class_counts[class_name] += 1

#     # Print the counts for each class
#     print("Class instance counts:")
#     for class_name, count in class_counts.items():
#         print(f"{class_name}: {count}")

#     return class_counts

# count_class_instances(folder_path)

import os

def find_label_files_with_class(folder_path, class_name="Survey_Ship"):
    # List to store paths of files containing the class
    matching_files = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Assuming label files are .txt
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                # Check each line for the class name
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[-2] == class_name:
                        matching_files.append(file_path)
                        break  # Stop checking this file once the class is found

    # Print the results
    if matching_files:
        print(f"Label files containing '{class_name}':")
        for path in matching_files:
            print(path)
    else:
        print(f"No label files found containing '{class_name}'.")

    return matching_files

# Example usage
# folder_path = "path/to/your/labels/folder"  # Replace with your folder path
folder_path = r"D:\Ship_Detection_Data\Dataset_v5\labels"  # Replace with your folder path
find_label_files_with_class(folder_path, "Survey_Ship")

# # Example usage