import os

def normalize_labels(label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for label_file in os.listdir(label_folder):
        if not label_file.endswith('.txt'):
            continue

        input_path = os.path.join(label_folder, label_file)
        output_path = os.path.join(output_folder, label_file)

        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in fin:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue  # skip malformed lines

                class_id = parts[0]
                coords = list(map(float, parts[1:]))

                # Normalize all coordinates by 1024
                normalized_coords = [f"{coord / 1024:.6f}" for coord in coords]

                # Write new line
                fout.write(f"{class_id} {' '.join(normalized_coords)}\n")

        print(f"✅ Normalized: {label_file}")

# --- Example Usage ---
label_folder = r"D:\Ship_Detection_Data\SplittedDataset\labels\val_v8"
output_folder = r"D:\Ship_Detection_Data\SplittedDataset\labels\val_v8_normalized"

normalize_labels(label_folder, output_folder)