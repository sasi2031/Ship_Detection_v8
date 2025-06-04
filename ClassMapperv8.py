import os, shutil

def create_class_mapping(classes_file):
    class_to_id = {}
    with open(classes_file, 'r') as f:
        for idx, line in enumerate(f):
            cls_name = line.strip()
            if cls_name:
                class_to_id[cls_name] = idx
    return class_to_id


def convert_label_files(label_folder, output_folder, class_mapping):
    os.makedirs(output_folder, exist_ok=True)

    for label_file in os.listdir(label_folder):
        if not label_file.endswith('.txt'):
            continue

        input_path = os.path.join(label_folder, label_file)
        output_path = os.path.join(output_folder, label_file)

        # Read the file
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        # If file is empty, copy as-is and continue
        if not lines or all(line.strip() == '' for line in lines):
            shutil.copy2(input_path, output_path)
            print(f"📎 Copied empty file: {label_file}")
            continue

        # Process non-empty files
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # skip malformed lines

            coords = parts[:8]
            class_name = parts[8]

            if class_name in class_mapping:
                class_id = class_mapping[class_name]
                converted_line = ' '.join([str(class_id)] + coords) + '\n'
                converted_lines.append(converted_line)
            else:
                print(f"⚠️ Class '{class_name}' not found in classes.txt. Skipping line:\n{line}")

        # Write converted content
        with open(output_path, 'w') as outfile:
            outfile.writelines(converted_lines)

    print("✅ Label conversion completed.")

# --- Example Usage ---
classes_file = r"D:\Ship_Detection_Data_Sentinel2\classes.txt"
label_folder = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\labels"
output_folder = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\labels_v8"

class_map = create_class_mapping(classes_file)
convert_label_files(label_folder, output_folder, class_map)
