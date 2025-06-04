import os
import xml.etree.ElementTree as ET
import math
import numpy as np

def get_obb_corners(cx, cy, w, h, angle, img_width, img_height):
    # Define the rectangle corners in local coordinates (before rotation)
    corners = [
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ]
    
    # Rotation matrix
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Rotate and translate corners to image coordinates
    rotated_corners = []
    for x, y in corners:
        # Rotate
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        # Translate to center
        x_img = x_rot + cx
        y_img = y_rot + cy
        # Normalize
        x_norm = x_img / img_width
        y_norm = y_img / img_height
        rotated_corners.append([x_norm, y_norm])
    
    return rotated_corners

def convert_xml_to_yolo_obb(xml_file, output_dir, class_map):
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            print(f"Error: Missing <size> tag in {xml_file}, skipping file")
            return
        
        width_tag = size.find('width')
        height_tag = size.find('height')
        
        if width_tag is None or height_tag is None:
            print(f"Error: Missing <width> or <height> tag in {xml_file}, skipping file")
            return
        
        img_width = int(width_tag.text or 0)
        img_height = int(height_tag.text or 0)
        
        # Check for valid dimensions
        if img_width <= 0 or img_height <= 0:
            print(f"Error: Invalid dimensions (width={img_width}, height={img_height}) in {xml_file}, skipping file")
            return
        
        # Use XML filename instead of <filename> tag content
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        
        # Output .txt file path
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Open output file
        with open(txt_path, 'w') as f:
            # Process each object
            for obj in root.findall('object'):
                # Get class name and map to index
                class_name = obj.find('name').text
                if class_name not in class_map:
                    print(f"Warning: Class '{class_name}' not in class map, skipping object in {xml_file}")
                    continue
                class_idx = class_map[class_name]
                
                # Get rotated bounding box parameters
                robndbox = obj.find('robndbox')
                if robndbox is None:
                    print(f"Warning: Missing <robndbox> tag for object in {xml_file}, skipping object")
                    continue
                
                cx = float(robndbox.find('cx').text or 0)
                cy = float(robndbox.find('cy').text or 0)
                w = float(robndbox.find('w').text or 0)
                h = float(robndbox.find('h').text or 0)
                angle = float(robndbox.find('angle').text or 0)
                
                # Convert to four corners
                corners = get_obb_corners(cx, cy, w, h, angle, img_width, img_height)
                
                # Flatten corners to [x1, y1, x2, y2, x3, y3, x4, y4]
                corners_flat = [coord for point in corners for coord in point]
                
                # Write to file: class_idx x1 y1 x2 y2 x3 y3 x4 y4
                line = [str(class_idx)] + [f"{c:.6f}" for c in corners_flat]
                f.write(' '.join(line) + '\n')
    
    except ET.ParseError:
        print(f"Error: Failed to parse XML file {xml_file}, skipping file")
    except Exception as e:
        print(f"Error: Unexpected issue processing {xml_file}: {str(e)}, skipping file")

def main():
    # Configuration
    xml_dir = r"D:\Ship_Detection_Data_12Classes\Trail\labels_xml"  # Update to your XML folder
    output_dir = r"D:\Ship_Detection_Data_12Classes\Trail\labels00"# Update to your output folder
    classes_file = 'D:\\Ship_Detection_Data_12Classes\\Ship_12Class_Data_v8\\classes.txt' # Update to your classes.txt path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class map
    class_map = {}
    try:
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                class_map[class_name] = idx
    except FileNotFoundError:
        print(f"Error: classes.txt not found at {classes_file}")
        return
    
    # Process each XML file
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            convert_xml_to_yolo_obb(xml_path, output_dir, class_map)
            print(f"Processed {xml_file}")

if __name__ == "__main__":
    main()
 