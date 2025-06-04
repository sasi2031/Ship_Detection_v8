import os

def clean_label_files(label_folder, output_folder=None, overwrite=True):
    if not overwrite and output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for label_file in os.listdir(label_folder):
        if not label_file.endswith('.txt'):
            continue

        input_path = os.path.join(label_folder, label_file)
        output_path = os.path.join(output_folder if output_folder else label_folder, label_file)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9:
                # Remove the last part (e.g., '0')
                cleaned_line = ' '.join(parts[:-1]) + '\n'
                cleaned_lines.append(cleaned_line)
            else:
                # Keep malformed or short lines as-is
                cleaned_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(cleaned_lines)

    print(f"✅ Cleaned label files in: {label_folder}")

# --- Example Usage ---
label_folder = r"D:\Ship_Detection_Data\SplittedDataset\labels\val"
# If you want to save to a new folder instead of overwriting:
# output_folder = "path/to/cleaned_labels"
# clean_label_files(label_folder, output_folder=output_folder, overwrite=False)

clean_label_files(label_folder)