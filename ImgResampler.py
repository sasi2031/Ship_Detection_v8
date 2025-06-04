import os
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

# --- Configuration ---
INPUT_IMAGE_FOLDER = r"D:\Ship_Detection_Data_12Classes\CombinedData\images"
OUTPUT_IMAGE_FOLDER = r"D:\Ship_Detection_Data_12Classes\CombinedData_v8\images"
TARGET_SIZE = (1024, 1024)
MAX_WORKERS = os.cpu_count()  # Use all available CPU cores
# -----------------------

class ImageResizer:
    def __init__(self, target_size, input_folder, output_folder):
        self.target_size = target_size
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.success_count = 0
        self.error_count = 0
        self.lock = threading.Lock()
        
    def resize_image_with_average_pooling(self, src_path, dest_path):
        """
        Resizes image using INTER_AREA interpolation (acts like average pooling).
        Thread-safe version with error handling.
        """
        try:
            img = cv2.imread(src_path)
            if img is None:
                with self.lock:
                    print(f"Error: Could not read image {src_path}")
                    self.error_count += 1
                return False
            
            resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Ensure output directory exists (thread-safe)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            if cv2.imwrite(dest_path, resized_img):
                with self.lock:
                    self.success_count += 1
                return True
            else:
                with self.lock:
                    print(f"Error: Could not write image {dest_path}")
                    self.error_count += 1
                return False
                
        except Exception as e:
            with self.lock:
                print(f"Error processing {src_path}: {str(e)}")
                self.error_count += 1
            return False
    
    def process_image(self, filename):
        """Process a single image file."""
        src_path = os.path.join(self.input_folder, filename)
        dest_path = os.path.join(self.output_folder, filename)
        return self.resize_image_with_average_pooling(src_path, dest_path)

def get_image_files(folder_path):
    """Get all image files from the input folder."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(folder_path):
        if Path(filename).suffix.lower() in image_extensions:
            image_files.append(filename)
    
    return image_files

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
    
    # Get all image files
    image_files = get_image_files(INPUT_IMAGE_FOLDER)
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"No image files found in '{INPUT_IMAGE_FOLDER}'")
        return
    
    print(f"Found {total_images} images to resize")
    print(f"Resizing images from '{INPUT_IMAGE_FOLDER}' to '{OUTPUT_IMAGE_FOLDER}' ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
    print(f"Using {MAX_WORKERS} worker threads")
    
    # Initialize resizer
    resizer = ImageResizer(TARGET_SIZE, INPUT_IMAGE_FOLDER, OUTPUT_IMAGE_FOLDER)
    
    # Start timing
    start_time = time.time()
    
    # Process images with multithreading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_filename = {executor.submit(resizer.process_image, filename): filename 
                            for filename in image_files}
        
        # Process completed tasks and show progress
        completed = 0
        for future in as_completed(future_to_filename):
            completed += 1
            filename = future_to_filename[future]
            
            try:
                result = future.result()
                # Print progress every 10% or every 100 images, whichever is more frequent
                progress_interval = max(1, min(100, total_images // 10))
                if completed % progress_interval == 0 or completed == total_images:
                    progress_percent = (completed / total_images) * 100
                    print(f"Progress: {completed}/{total_images} ({progress_percent:.1f}%)")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                with resizer.lock:
                    resizer.error_count += 1
    
    # Calculate and display results
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total images processed: {total_images}")
    print(f"Successfully resized: {resizer.success_count}")
    print(f"Errors encountered: {resizer.error_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/total_images:.3f} seconds")
    print(f"Images per second: {total_images/processing_time:.2f}")

if __name__ == "__main__":
    main()