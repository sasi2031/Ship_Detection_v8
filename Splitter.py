import os
import random
import shutil

def split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create train, val, test subfolders for images and labels
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_folder, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, subset, 'labels'), exist_ok=True)

    # Get the list of all images and labels
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    labels = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    # Shuffle the images and labels
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    # Calculate the number of files for each subset
    total_files = len(images)
    train_size = int(train_ratio * total_files)
    val_size = int(val_ratio * total_files)
    test_size = total_files - train_size - val_size

    # Split the files
    train_images, val_images, test_images = images[:train_size], images[train_size:train_size + val_size], images[train_size + val_size:]
    train_labels, val_labels, test_labels = labels[:train_size], labels[train_size:train_size + val_size], labels[train_size + val_size:]

    # Copy files to the respective folders
    for subset, subset_images, subset_labels in zip(
            ['train', 'val', 'test'],
            [train_images, val_images, test_images],
            [train_labels, val_labels, test_labels]
    ):
        for image, label in zip(subset_images, subset_labels):
            shutil.copy(os.path.join(image_folder, image), os.path.join(output_folder, subset, 'images', image))
            shutil.copy(os.path.join(label_folder, label), os.path.join(output_folder, subset, 'labels', label))

    print(f"Dataset split complete. Files are saved in '{output_folder}'.")

# Example usage
split_dataset(r"D:\ShipDetection\Combined4ClassDataset\images", r"D:\ShipDetection\Combined4ClassDataset\labels", r"D:\ShipDetection\Combined4ClassDataset\V8LABELS")
