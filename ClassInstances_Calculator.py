import xml.etree.ElementTree as ET
import os
import glob
from collections import Counter

def count_class_instances(xml_dir, output_file):
    # Initialize counter for class instances
    class_counts = Counter()
    total_instances = 0
    
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
            
            # Count objects in this file
            for obj in root.findall('object'):
                name = obj.find('name')
                if name is None or name.text is None:
                    print(f"Warning: Object with no <name> tag in {xml_file}, skipping")
                    continue
                class_name = name.text.strip()
                class_counts[class_name] += 1
                total_instances += 1
                
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file}: {e}, skipping")
            continue
    
    # Print results
    print("\nClass Instance Counts:")
    print("-" * 30)
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")
    print("-" * 30)
    print(f"Total Instances: {total_instances}")
    print(f"Total Classes: {len(class_counts)}")
    
    # Save results to output file
    with open(output_file, 'w') as f:
        f.write("Class Instance Counts:\n")
        f.write("-" * 30 + "\n")
        for class_name, count in sorted(class_counts.items()):
            f.write(f"{class_name}: {count}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Instances: {total_instances}\n")
        f.write(f"Total Classes: {len(class_counts)}\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    xml_dir = 'D:\\Ship_Detection_Data_12Classes\\GE_Highres\\labels'
    output_file = 'D:\\Ship_Detection_Data_12Classes\\Ship_12Class_Data_v8\\class_counts.txt'
    count_class_instances(xml_dir, output_file)