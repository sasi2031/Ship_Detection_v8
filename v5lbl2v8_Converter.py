import math
import os

def yolov5obb_to_yolov8obb(line):
    """
    Converts a single YOLOv5-OBB label line (normalized center, size, angle degrees)
    to a YOLOv8-OBB label line (normalized 8 corner points).

    Args:
        line (str): A string in the format "class_id center_x center_y width height angle_degrees"
                    Coordinates, size, and angle are normalized [0, 1] and [-180, 180] (or similar range),
                    but specifically angle is in degrees.

    Returns:
        str: A string in the format "class_id x1 y1 x2 y2 x3 y3 x4 y4" with normalized coordinates.
             Returns None if the input line is invalid or empty after stripping.
    """
    line = line.strip() # Remove leading/trailing whitespace
    if not line or line.startswith('#'): # Skip empty lines or comments
        return None # Indicate that this line should be skipped

    try:
        # 1. Parse the line
        parts = line.split()
        if len(parts) != 6:
            print(f"Warning: Skipping invalid line format: {line}")
            return None

        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        angle_degrees = float(parts[5])

        # 2. Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)

        # 3. Pre-calculate sin and cos of the angle
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)

        # 4. Define half dimensions
        half_width = width / 2.0
        half_height = height / 2.0

        # 5. Define unrotated corner coordinates relative to the center (0,0)
        # Order: TL, TR, BR, BL (Counter-clockwise after rotation)
        unrotated_corners = [
            (-half_width, half_height),  # Top-Left
            (half_width, half_height),   # Top-Right
            (half_width, -half_height),  # Bottom-Right
            (-half_width, -half_height)  # Bottom-Left
        ]

        rotated_translated_corners = []
        for x_rel, y_rel in unrotated_corners:
            # 6. Rotate the corner point
            x_rotated = x_rel * cos_a - y_rel * sin_a
            y_rotated = x_rel * sin_a + y_rel * cos_a

            # 7. Translate the rotated corner point
            x_final = center_x + x_rotated
            y_final = center_y + y_rotated

            rotated_translated_corners.append((x_final, y_final))

        # 8. Format the output string
        # The order is important for YOLOv8 training, the calculation above
        # produces corners in a specific order (relative TL, TR, BR, BL rotated)
        # which is a valid counter-clockwise representation of the polygon.
        x1, y1 = rotated_translated_corners[0]
        x2, y2 = rotated_translated_corners[1]
        x3, y3 = rotated_translated_corners[2]
        x4, y4 = rotated_translated_corners[3]

        # Format with sufficient precision, e.g., 6 decimal places
        output_line = f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"

        return output_line

    except ValueError as e:
        print(f"Error processing line '{line}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing line '{line}': {e}")
        return None

# --- Batch Conversion Script ---

input_labels_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\labels" # <--- CHANGE THIS
output_labels_dir = r"D:\Ship_Detection_Data_Sentinel2\Dataset_v5\labels_v8_0" # <--- CHANGE THIS

# Create the output directory if it doesn't exist
os.makedirs(output_labels_dir, exist_ok=True)
print(f"Output directory ensured: {output_labels_dir}")

# Get list of files in the input directory
try:
    label_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} .txt files in {input_labels_dir}")

    if not label_files:
        print("No .txt files found. Exiting.")

    for filename in label_files:
        input_filepath = os.path.join(input_labels_dir, filename)
        output_filepath = os.path.join(output_labels_dir, filename)

        print(f"Processing {filename}...")

        try:
            with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
                for line in infile:
                    yolov8_line = yolov5obb_to_yolov8obb(line)
                    if yolov8_line: # Only write if the conversion was successful and line is valid
                        outfile.write(yolov8_line + '\n')

        except FileNotFoundError:
            print(f"Error: Input file not found during processing: {input_filepath}")
        except Exception as e:
            print(f"An error occurred processing file {filename}: {e}")

    print("\nBatch conversion complete.")

except FileNotFoundError:
    print(f"Error: Input directory not found at {input_labels_dir}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")