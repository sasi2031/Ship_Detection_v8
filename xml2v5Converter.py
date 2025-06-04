import xml.etree.ElementTree as ET
import os
import math

# --- Configuration ---
# Define the mapping from your class names to YOLO class indices
CLASS_MAPPING = {
        "Submarine": 0,
        "Commercial": 1,
        "Carrier": 2,
        "Frigate": 3
}

# Define the path to the folder containing your XML files
XML_FOLDER = r"D:\ShipDetection\Combined4ClassDataset\labels"  # <-- CHANGE THIS

# Define the path where you want to save the output TXT files
OUTPUT_FOLDER = r"D:\ShipDetection\Combined4ClassDataset\labs" # <-- CHANGE THIS
# --- End Configuration ---

def convert_xml_to_yolo_obb(xml_file, output_folder):
    """
    Converts a single XML annotation file to YOLOv5 OBB format.

    Args:
        xml_file (str): Path to the input XML file.
        output_folder (str): Path to the folder where the TXT file will be saved.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image dimensions
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)

        if img_width <= 0 or img_height <= 0:
             print(f"Warning: Invalid image dimensions (width={img_width}, height={img_height}) in {xml_file}. Skipping file.")
             return

        yolo_labels = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_MAPPING:
                print(f"Warning: Class '{class_name}' not found in CLASS_MAPPING. Skipping object in {xml_file}.")
                continue

            class_index = CLASS_MAPPING[class_name]

            robndbox = obj.find('robndbox')
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle_rad = float(robndbox.find('angle').text) # Angle is already in radians in the XML
            angle_deg= math.degrees(angle_rad)  # Convert to degrees if needed
            # Normalize coordinates and dimensions
            x_center_norm = cx / img_width
            y_center_norm = cy / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            # Ensure angle is within a common range like [0, pi) if necessary
            # YOLOv5 OBB usually expects angles in radians.
            # The provided angle 2.446373 is ~140 degrees.
            # Let's assume the XML angle is already in the desired range [0, pi) or similar
            # If YOLOv5 requires a different range (e.g. [-pi/2, pi/2)), you might need adjustments here.
            # Example adjustment to put angle in [0, pi):
            # angle_rad = angle_rad % math.pi
            # if angle_rad < 0:
            #     angle_rad += math.pi


            # Format for YOLOv5 OBB: class_id cx_norm cy_norm w_norm h_norm angle_rad
            yolo_labels.append(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f} {angle_deg:.6f}")

        # Create output filename
        base_filename = os.path.splitext(os.path.basename(xml_file))[0]
        output_txt_file = os.path.join(output_folder, f"{base_filename}.txt")

        # Write to output file
        if yolo_labels:
            with open(output_txt_file, 'w') as f:
                f.write("\n".join(yolo_labels))
            # print(f"Successfully converted {xml_file} to {output_txt_file}")
        # else:
            # print(f"No valid objects found or written for {xml_file}")


    except ET.ParseError:
        print(f"Error parsing XML file: {xml_file}")
    except AttributeError as e:
         print(f"Error processing file {xml_file}: Missing expected XML element. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred processing {xml_file}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(XML_FOLDER):
        print(f"Error: XML input folder not found: {XML_FOLDER}")
    elif not os.path.isdir(OUTPUT_FOLDER):
        print(f"Output folder not found, creating it: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    if os.path.isdir(XML_FOLDER) and os.path.isdir(OUTPUT_FOLDER) :
      processed_files = 0
      print(f"Starting conversion from {XML_FOLDER} to {OUTPUT_FOLDER}...")
      for filename in os.listdir(XML_FOLDER):
          if filename.lower().endswith('.xml'):
              xml_filepath = os.path.join(XML_FOLDER, filename)
              convert_xml_to_yolo_obb(xml_filepath, OUTPUT_FOLDER)
              processed_files += 1
      print(f"Conversion complete. Processed {processed_files} XML files.")