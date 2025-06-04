import os

# Mapping of incorrect class names to correct one
class_replacement_map = {
    # Fleet Replenishment variants
    "Fleet*Replenishment*Ship": "Fleet_Replenishment_Ship",
    "Fleet*Replenishment*Ships": "Fleet_Replenishment_Ship",
    "Fleet*_Replenishment_Ships": "Fleet_Replenishment_Ship",
    "Fleet_Replenishment_Ships": "Fleet_Replenishment_Ship",

    # Munsif Mine Hunter variants
    "Munsif_Class_Mine_Hunte": "Munsif_Class_Mine_Hunter",

    # Tughril/Turghil Frigate
    "Turghil_Class_Frigate": "Tughril_Class_Frigate",

    # Utility Ship variants
    "UTILITY_SHIPS": "Utility_Ship",
    "Utility_Ships": "Utility_Ship",
    "Utiliy_Ship": "Utility_Ship",

    # Small Ship variants
    "Small_Ships": "Small_Ship",
    "Small_ship": "Small_Ship",

    # Naval Auxiliary variants
    "Naval_auxiliary": "Naval_Auxiliary",

    # Agosta Submarine variants
    "Agosta_70B_Submarine": "Agosta_70_Submarine",

    # Fishing Boat variants
    "Fishing_Boats": "Fishing_Boat",

    # Naval Tanker variants
    "Naval_Tankers": "Naval_Tanker",
    "Navy_Tanker": "Naval_Tanker",

    # Tariq Class variants
    "Tariq_Class": "Tariq_Class_Frigate"
}

def normalize_labels(label_folder, output_folder=None):
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
                        original_class = parts[-2]
                        corrected_class = class_replacement_map.get(original_class, original_class)
                        parts[-2] = corrected_class
                        updated_line = ' '.join(parts) + '\n'
                        updated_lines.append(updated_line)
                        class_set.add(corrected_class)
                    else:
                        updated_lines.append(line)  # Keep malformed lines as-is

                with open(output_path, 'w') as outfile:
                    outfile.writelines(updated_lines)

    # Write updated classes.txt
    with open("classes.txt", "w") as f:
        for cls in sorted(class_set):
            f.write(cls + "\n")

    print(f"Normalization complete. Unique classes saved to 'classes.txt'.")

# Example usage
label_folder = r"D:\Ship_Detection_Data\Dataset_v5\labels"
output_folder = None  # Optional: leave None to overwrite originals

normalize_labels(label_folder, output_folder)