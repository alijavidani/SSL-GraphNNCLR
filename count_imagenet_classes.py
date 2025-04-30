import os
import argparse
import json
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm

def count_images_per_class(dataset_path, split='train'):
    """
    Count the number of images in each class folder in the ImageNet-1K dataset.
    
    Args:
        dataset_path (str): Path to the ImageNet dataset root
        split (str): 'train' or 'val' split
    
    Returns:
        OrderedDict: Mapping from class folder name to image count
        list: Sorted class folder names
        int: Total number of images
    """
    # Construct the path to the specified split
    split_path = os.path.join(dataset_path, split)
    
    if not os.path.exists(split_path):
        raise ValueError(f"Path does not exist: {split_path}")
    
    # Get all class folders
    class_folders = sorted([d for d in os.listdir(split_path) 
                           if os.path.isdir(os.path.join(split_path, d))])
    
    # Check if this looks like ImageNet data
    if not class_folders or not class_folders[0].startswith('n'):
        print(f"Warning: Folders in {split_path} don't match expected ImageNet format (e.g., 'n01440764')")
    
    # Count images in each class folder
    class_counts = OrderedDict()
    total_images = 0
    
    print(f"Counting images in {len(class_folders)} class folders...")
    for folder in tqdm(class_folders):
        folder_path = os.path.join(split_path, folder)
        # Count image files (jpg, jpeg, png)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        count = len(image_files)
        class_counts[folder] = count
        total_images += count
    
    print(f"Found {total_images} total images in {len(class_counts)} classes")
    
    # Calculate statistics
    counts = np.array(list(class_counts.values()))
    print(f"Images per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}, median={np.median(counts)}")
    
    return class_counts, class_folders, total_images


def generate_class_mappings(class_counts, output_file=None):
    """
    Generate mappings from class folder names to indices, and cumulative counts.
    
    Args:
        class_counts (OrderedDict): Mapping from class folder name to image count
        output_file (str, optional): Path to save the mappings as JSON
    
    Returns:
        tuple: (class_to_idx, cumulative_counts)
    """
    # Create class index mapping
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_counts.keys())}
    
    # Create cumulative counts for efficient indexing
    counts = list(class_counts.values())
    cumulative_counts = [0]
    running_sum = 0
    for count in counts:
        running_sum += count
        cumulative_counts.append(running_sum)
    
    # Save mappings if requested
    if output_file:
        mappings = {
            'class_to_idx': class_to_idx,
            'class_counts': class_counts,
            'cumulative_counts': cumulative_counts,
            'total_images': cumulative_counts[-1]
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"Saved class mappings to {output_file}")
    
    return class_to_idx, cumulative_counts


def create_labels_tensor(class_counts, cumulative_counts, train_split=True):
    """
    Create a labels tensor for ImageNet based on actual image counts.
    
    Args:
        class_counts (OrderedDict): Mapping from class folder name to image count
        cumulative_counts (list): Cumulative counts for efficient indexing
        train_split (bool): Whether this is for the training split (True) or validation (False)
    
    Returns:
        torch.Tensor: Labels tensor with correct class assignments
    """
    total_images = cumulative_counts[-1]
    labels = torch.zeros(total_images, dtype=torch.long)
    
    for class_idx, (_, count) in enumerate(zip(class_counts.keys(), class_counts.values())):
        start_idx = cumulative_counts[class_idx]
        end_idx = cumulative_counts[class_idx + 1]
        labels[start_idx:end_idx] = class_idx
    
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count images per class in ImageNet dataset")
    parser.add_argument("--dataset_path", type=str, default="/amin/imagenet/imagenet",
                       help="Path to the ImageNet dataset root directory")
    parser.add_argument("--train_dir", type=str, default="train",
                       help="Name of the training directory")
    parser.add_argument("--val_dir", type=str, default="val", 
                       help="Name of the validation directory")
    parser.add_argument("--output_file", type=str, default="imagenet_class_info.json",
                       help="Output JSON file to save class information")
    
    args = parser.parse_args()
    
    # Count training images
    print(f"Counting training images...")
    train_counts, train_classes, train_total = count_images_per_class(
        args.dataset_path, args.train_dir)
    
    # Count validation images
    print(f"\nCounting validation images...")
    val_counts, val_classes, val_total = count_images_per_class(
        args.dataset_path, args.val_dir)
    
    # Generate and save mappings
    train_class_to_idx, train_cumulative_counts = generate_class_mappings(
        train_counts, output_file=os.path.join(os.path.dirname(args.output_file), 
                                              "train_" + os.path.basename(args.output_file)))
    
    val_class_to_idx, val_cumulative_counts = generate_class_mappings(
        val_counts, output_file=os.path.join(os.path.dirname(args.output_file), 
                                            "val_" + os.path.basename(args.output_file)))
    
    # Combined information
    combined_info = {
        'train': {
            'class_to_idx': train_class_to_idx,
            'class_counts': train_counts,
            'cumulative_counts': train_cumulative_counts,
            'total_images': train_total
        },
        'val': {
            'class_to_idx': val_class_to_idx,
            'class_counts': val_counts,
            'cumulative_counts': val_cumulative_counts,
            'total_images': val_total
        }
    }
    
    # Save combined information
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(combined_info, f, indent=2)
    
    print(f"\nSaved combined information to {args.output_file}")
    
    # Demo: Create label tensors
    train_labels = create_labels_tensor(train_counts, train_cumulative_counts, train_split=True)
    val_labels = create_labels_tensor(val_counts, val_cumulative_counts, train_split=False)
    
    print(f"\nCreated labels tensors:")
    print(f"  Training: {train_labels.shape}, unique classes: {len(torch.unique(train_labels))}")
    print(f"  Validation: {val_labels.shape}, unique classes: {len(torch.unique(val_labels))}") 