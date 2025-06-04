import xml.etree.ElementTree as ET
import os
import glob
from collections import Counter

def count_solo_class_appearances(xml_dir, output_file):
    # Initialize counter for solo appearances
    solo_counts = Counter()
    total_solo_images = 0
    
    # Get all XML files
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return
    
    print(f"Processing {len(xml_files)} XML files...")
    
    # Process each XML file
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get all class names in this file
            class_names = []
            for obj in root.findall('object'):
                name = obj.find('name')
                if name is None or name.text is None:
                    print(f"Warning: Object with no <name> tag in {xml_file}, skipping object")
                    continue
                class_names.append(name.text.strip())
            
            # Check if the file has objects and only one unique class
            if class_names and len(set(class_names)) == 1:
                solo_class = class_names[0]
                solo_counts[solo_class] += 1
                total_solo_images += 1
            elif not class_names:
                print(f"Warning: No valid objects found in {xml_file}, skipping")
            
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file}: {e}, skipping")
            continue
    
    # Print results
    print("\nSolo Class Appearance Counts (Images with only one class):")
    print("-" * 50)
    for class_name, count in sorted(solo_counts.items()):
        print(f"{class_name}: {count}")
    print("-" * 50)
    print(f"Total Solo Images: {total_solo_images}")
    print(f"Classes with Solo Appearances: {len(solo_counts)}")
    print(f"Total Images Processed: {len(xml_files)}")
    
    # Save results to output file
    with open(output_file, 'w') as f:
        f.write("Solo Class Appearance Counts (Images with only one class):\n")
        f.write("-" * 50 + "\n")
        for class_name, count in sorted(solo_counts.items()):
            f.write(f"{class_name}: {count}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Solo Images: {total_solo_images}\n")
        f.write(f"Classes with Solo Appearances: {len(solo_counts)}\n")
        f.write(f"Total Images Processed: {len(xml_files)}\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    xml_dir = 'D:\\Ship_Detection_Data_12Classes\\GE_Highres\\labels'
    output_file = 'D:\\Ship_Detection_Data_12Classes\\Ship_12Class_Data_v8\\solo_class_counts.txt'
    count_solo_class_appearances(xml_dir, output_file)