import os

def fix_normalized_labels(label_folder, output_folder):
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
                    fout.write(line)  # write malformed lines back as-is
                    continue

                class_id = parts[0]
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    fout.write(line)
                    continue

                # Clip to [0.0, 1.0]
                fixed_coords = [min(max(c, 0.0), 1.0) for c in coords]

                # Format output
                fixed_line = [class_id] + [f"{c:.6f}" for c in fixed_coords]
                fout.write(" ".join(fixed_line) + "\n")

        print(f"✅ Fixed: {label_file}")

# --- Example Usage ---
label_folder = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\val\labels"
output_folder = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8_Splitted\val\labelsTruth"

fix_normalized_labels(label_folder, output_folder)