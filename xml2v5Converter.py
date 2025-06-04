
import xml.etree.ElementTree as ET
import os
import math

# --- Configuration: YOU SET THESE PATHS ---
XML_FOLDER = r"D:\Ship_Detection_Data_12Classes\CombinedData\labels"    # Folder containing XML files
OUTPUT_FOLDER = r"D:\Ship_Detection_Data_12Classes\CombinedData_v5\labels" # Output folder for TXT files
CLASSES_FILE = r"D:\Ship_Detection_Data_12Classes\CombinedData\classes.txt"     # Path to save/load classes.txt

def extract_unique_classes(xml_folder):
    """Extracts unique class names from all XML files in the given folder."""
    class_names = set()
    for filename in os.listdir(xml_folder):
        if filename.lower().endswith('.xml'):
            xml_filepath = os.path.join(xml_folder, filename)
            try:
                tree = ET.parse(xml_filepath)
                root = tree.getroot()
                for obj in root.findall('object'):
                    class_name = obj.find('name').text if obj.find('name') is not None else None
                    if class_name:
                        class_names.add(class_name.strip())
            except Exception as e:
                print(f"Error parsing {xml_filepath}: {e}")
    return sorted(class_names)

def generate_classes_file(classes_file, xml_folder):
    """Generates classes.txt by extracting unique class names from XML files."""
    print("Generating classes.txt...")
    class_names = extract_unique_classes(xml_folder)
    if not class_names:
        raise ValueError("No classes found in XML files. Cannot generate classes.txt.")

    with open(classes_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Generated {classes_file} with {len(class_names)} classes: {class_names}")
    return {name: idx for idx, name in enumerate(class_names)}

def load_class_mapping(classes_file):
    """Loads class mapping from an existing classes.txt file."""
    if not os.path.isfile(classes_file):
        raise FileNotFoundError(f"classes.txt file not found: {classes_file}")

    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]

    return {name: idx for idx, name in enumerate(class_names)}

def convert_xml_to_yolo_obb(xml_file, output_folder, class_mapping):
    """
    Converts a single XML annotation file to YOLOv5 OBB format.
    Always creates a corresponding .txt file, even if empty.
    """
    yolo_labels = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            img_width = 1.0
            img_height = 1.0
            print(f"Warning: No <size> tag found in {xml_file}. Assuming dummy dimensions.")
        else:
            img_width = float(size.find('width').text) if size.find('width') is not None else 1.0
            img_height = float(size.find('height').text) if size.find('height') is not None else 1.0

        if img_width <= 0 or img_height <= 0:
            print(f"Warning: Invalid image dimensions in {xml_file}. Skipping annotations.")
        else:
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    print(f"Warning: Missing <name> in object of {xml_file}. Skipping.")
                    continue

                class_name = class_name_elem.text
                if not class_name or class_name not in class_mapping:
                    print(f"Warning: Class '{class_name}' not found in class mapping. Skipping object in {xml_file}.")
                    continue

                class_index = class_mapping[class_name]

                robndbox = obj.find('robndbox')
                if robndbox is None:
                    print(f"Warning: Missing <robndbox> in object of {xml_file}. Skipping.")
                    continue

                cx_elem = robndbox.find('cx')
                cy_elem = robndbox.find('cy')
                w_elem = robndbox.find('w')
                h_elem = robndbox.find('h')
                angle_elem = robndbox.find('angle')

                if None in [cx_elem, cy_elem, w_elem, h_elem, angle_elem]:
                    print(f"Warning: Missing one or more bounding box fields in {xml_file}. Skipping.")
                    continue

                try:
                    cx = float(cx_elem.text)
                    cy = float(cy_elem.text)
                    w = float(w_elem.text)
                    h = float(h_elem.text)
                    angle_rad = float(angle_elem.text)
                except (ValueError, TypeError) as ve:
                    print(f"Warning: Could not convert numeric values in {xml_file}: {ve}")
                    continue

                angle_deg = math.degrees(angle_rad)

                x_center_norm = cx / img_width
                y_center_norm = cy / img_height
                width_norm = w / img_width
                height_norm = h / img_height

                yolo_labels.append(
                    f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} "
                    f"{width_norm:.6f} {height_norm:.6f} {angle_deg:.6f}"
                )

    except ET.ParseError as e:
        print(f"Error: Failed to parse XML file {xml_file}. It may be malformed. Error: {e}")
    except Exception as e:
        print(f"Unexpected error processing {xml_file}: {e}")

    # --- ALWAYS CREATE OUTPUT FILE ---
    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    output_txt_file = os.path.join(output_folder, f"{base_filename}.txt")

    with open(output_txt_file, 'w') as f:
        if yolo_labels:
            f.write("\n".join(yolo_labels))
        # Else leave it empty — this is what we want!

    return

# --- Main Execution ---
if __name__ == "__main__":
    # Generate or load classes.txt
    if not os.path.isfile(CLASSES_FILE):
        CLASS_MAPPING = generate_classes_file(CLASSES_FILE, XML_FOLDER)
    else:
        CLASS_MAPPING = load_class_mapping(CLASSES_FILE)
        print(f"Loaded existing classes from {CLASSES_FILE}")

    # Create output folder if it doesn't exist
    if not os.path.isdir(OUTPUT_FOLDER):
        print(f"Output folder not found, creating it: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    # Convert XML files
    processed_files = 0
    print(f"Starting conversion from {XML_FOLDER} to {OUTPUT_FOLDER}...")
    for filename in os.listdir(XML_FOLDER):
        if filename.lower().endswith('.xml'):
            xml_filepath = os.path.join(XML_FOLDER, filename)
            convert_xml_to_yolo_obb(xml_filepath, OUTPUT_FOLDER, CLASS_MAPPING)
            processed_files += 1
    print(f"Conversion complete. Processed {processed_files} XML files.")