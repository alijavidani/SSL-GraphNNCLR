import torch
import os
import argparse
import json
from collections import OrderedDict
from count_imagenet_classes import count_images_per_class, create_labels_tensor, generate_class_mappings


def save_graph(args):
    """
    Save teacher and student graphs with train/val masks based on dataset name.
    
    Args:
        args: Command line arguments containing dataset_name and other parameters
    """
    # Dataset-specific constants
    dataset_configs = {
        "cifar10": {
            "num_classes": 10,
            "num_train": 50000,
            "num_val": 10000
        },
        "imagenet100": {
            "num_classes": 100,
            "num_train": 130000,
            "num_val": 5000
        },
        "imagenet1k": {
            "num_classes": 1000,
            "num_train": 1281167,  # ~1.28M training images
            "num_val": 50000       # ~50K validation images
        }
    }
    
    # Check if dataset is supported
    if args.dataset_name.lower() not in dataset_configs:
        raise ValueError(f"Dataset {args.dataset_name} not supported. Choose from: {list(dataset_configs.keys())}")
    
    print(f"Processing dataset: {args.dataset_name}")
    
    # Special handling for ImageNet-1K with actual class counts
    if args.dataset_name.lower() == "imagenet1k" and args.dataset_path:
        print(f"Using actual class counts from ImageNet-1K at {args.dataset_path}")
        
        # Use cached class info if available, otherwise count images
        cache_file = os.path.join(os.path.dirname(args.output_path), "imagenet_class_info.json")
        
        if os.path.exists(cache_file) and not args.recount_classes:
            print(f"Loading cached class information from {cache_file}")
            with open(cache_file, 'r') as f:
                class_info = json.load(f)
                
            # Extract necessary information
            train_total = class_info['train']['total_images']
            val_total = class_info['val']['total_images']
            
            # Create label tensors directly from cached information
            train_counts = OrderedDict(class_info['train']['class_counts'])
            val_counts = OrderedDict(class_info['val']['class_counts'])
            
            train_cumulative_counts = class_info['train']['cumulative_counts']
            val_cumulative_counts = class_info['val']['cumulative_counts']
            
        else:
            print("Counting actual images per class (this may take a while)...")
            # Count training images
            train_counts, _, train_total = count_images_per_class(
                args.dataset_path, args.train_dir)
            
            # Count validation images
            val_counts, _, val_total = count_images_per_class(
                args.dataset_path, args.val_dir)
            
            # Generate mappings
            _, train_cumulative_counts = generate_class_mappings(train_counts)
            _, val_cumulative_counts = generate_class_mappings(val_counts)
            
            # Save information for future use
            combined_info = {
                'train': {
                    'class_counts': dict(train_counts),
                    'cumulative_counts': train_cumulative_counts,
                    'total_images': train_total
                },
                'val': {
                    'class_counts': dict(val_counts),
                    'cumulative_counts': val_cumulative_counts,
                    'total_images': val_total
                }
            }
            
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(combined_info, f, indent=2)
        
        # Create label tensors
        y_train = create_labels_tensor(train_counts, train_cumulative_counts, train_split=True)
        y_val = create_labels_tensor(val_counts, val_cumulative_counts, train_split=False)
        
        # Update dataset configuration with actual counts
        num_train = train_total
        num_val = val_total
        num_total = num_train + num_val
        
    else:
        # Get dataset configuration for other datasets or when dataset_path is not provided
        config = dataset_configs[args.dataset_name.lower()]
        num_classes = config["num_classes"]
        num_train = config["num_train"]
        num_val = config["num_val"]
        num_total = num_train + num_val
        
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        
        # Calculate samples per class (assuming equal distribution)
        num_samples_per_class_train = num_train // num_classes
        num_samples_per_class_val = num_val // num_classes
        
        # Create labels (assuming equal distribution)
        y_train = torch.zeros(num_train, dtype=torch.long)
        y_val = torch.zeros(num_val, dtype=torch.long)
        
        for i in range(num_classes):
            train_indices = torch.arange(i*num_samples_per_class_train, (i+1)*num_samples_per_class_train)
            val_indices = torch.arange(i*num_samples_per_class_val, (i+1)*num_samples_per_class_val)
            
            # Handle edge cases where the division isn't perfect
            train_indices = train_indices[train_indices < num_train]
            val_indices = val_indices[val_indices < num_val]
            
            y_train[train_indices] = i
            y_val[val_indices] = i
    
    # Create masks for train/val split
    train_mask = torch.zeros(num_total, dtype=torch.bool)
    train_mask[:num_train] = True
    val_mask = ~train_mask
    
    # Combine train and val labels
    y = torch.cat([y_train, y_val], dim=0)
    print(f"Created labels tensor with shape: {y.shape}")
    
    # Process teacher graph
    teacher_graph = args.teacher_graph
    teacher_graph_test = args.teacher_graph_test
    
    teacher_graph.num_nodes = num_total
    teacher_graph.edge_index = torch.cat([teacher_graph.edge_index, teacher_graph_test.edge_index], dim=1)
    teacher_graph.x = torch.cat([teacher_graph.x, teacher_graph_test.x], dim=0)
    teacher_graph.train_mask = train_mask
    teacher_graph.val_mask = val_mask
    teacher_graph.y = y
    
    # Process student graph
    student_graph = args.student_graph
    student_graph_test = args.student_graph_test
    
    student_graph.num_nodes = num_total
    student_graph.edge_index = torch.cat([student_graph.edge_index, student_graph_test.edge_index], dim=1)
    student_graph.x = torch.cat([student_graph.x, student_graph_test.x], dim=0)
    student_graph.train_mask = train_mask
    student_graph.val_mask = val_mask
    student_graph.y = y
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save the graphs
    torch.save(
        {'teacher_graph': teacher_graph, 'student_graph': student_graph}, 
        args.output_path
    )
    
    print(f"Graphs successfully saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save graph data with dataset-specific parameters")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["cifar10", "imagenet100", "imagenet1k"],
                        help="Name of the dataset (cifar10, imagenet100, or imagenet1k)")
    parser.add_argument("--edges_per_node", type=int, default=1, 
                        help="Number of edges per node used in training")
    parser.add_argument("--model_name", type=str, default="vit_base", 
                        help="Model architecture name (e.g., vit_base, vit_small)")
    parser.add_argument("--output_dir", type=str, default="./saved_graphs", 
                        help="Directory to save the output graph file")
    parser.add_argument("--dataset_path", type=str, default="/amin/imagenet/imagenet",
                        help="Path to the dataset (needed for ImageNet-1K to count class images)")
    parser.add_argument("--train_dir", type=str, default="train",
                        help="Name of the training directory (for ImageNet-1K)")
    parser.add_argument("--val_dir", type=str, default="val",
                        help="Name of the validation directory (for ImageNet-1K)")
    parser.add_argument("--recount_classes", action="store_true",
                        help="Force recount of class images even if cached information exists")
    
    args = parser.parse_args()
    
    # These would be set from outside when calling the script
    # In a real scenario, these would be passed from the calling code
    # For now, we'll assume they are set before this script is called
    
    # Create the output path
    patch_size = "16"  # Default patch size, could be added as an argument
    if "vit" in args.model_name:
        if args.model_name == "vit_base":
            model_short = "vitb"
        elif args.model_name == "vit_small":
            model_short = "vits"
        elif args.model_name == "vit_tiny":
            model_short = "vitt"
        else:
            model_short = args.model_name
    else:
        model_short = args.model_name
    
    filename = f"{args.dataset_name}_{model_short}{patch_size}_epn{args.edges_per_node}.pth"
    args.output_path = os.path.join(args.output_dir, args.dataset_name, filename)
    
    # In a real scenario, you would set teacher_graph, teacher_graph_test, etc. here
    # For this example, we assume these are set before this script is called
    
    save_graph(args) 