import os

# Class to remove
CLASS_TO_REMOVE = "Fishing_Boat"

def remove_class_from_labels(label_folder, output_folder=None):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    class_set = set()

    for root, dirs, files in os.walk(label_folder):
        for file in files:
            if file.endswith('.txt'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder if output_folder else root, file)

                with open(input_path, 'r') as infile:
                    lines = infile.readlines()

                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        class_name = parts[-2]
                        if class_name == CLASS_TO_REMOVE:
                            # Skip this line (i.e. delete it)
                            continue
                        # Otherwise keep the line and collect class
                        class_set.add(class_name)
                        updated_lines.append(line)
                    else:
                        # Keep malformed lines as-is
                        updated_lines.append(line)

                with open(output_path, 'w') as outfile:
                    outfile.writelines(updated_lines)

    # Generate updated classes.txt
    with open("classes.txt", "w") as f:
        for cls in sorted(class_set):
            f.write(cls + "\n")

    print(f"All instances of class '{CLASS_TO_REMOVE}' removed.")
    print(f"Unique remaining classes saved to 'classes.txt'.")

# Example usage
label_folder = r"D:\Ship_Detection_Data\Dataset_v5\labels"
output_folder = None  # Optional: leave None to overwrite originals

remove_class_from_labels(label_folder, output_folder)