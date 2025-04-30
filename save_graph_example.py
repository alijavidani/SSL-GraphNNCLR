import torch
import os
import argparse
import numpy as np
import json
from collections import OrderedDict
from save_graph_wrapper import save_graphs_from_training
from torch_geometric.data import Data
import torch.nn.functional as F

# This would be in your main training script after training is complete

def save_graphs_after_training(args, checkpoint_path=None):
    """
    Example function showing how to save graphs after training is complete.
    
    Args:
        args: Command line arguments from training
        checkpoint_path: Path to a checkpoint file to load features and nearest neighbors
    """
    # Determine the dataset name based on your data path or args
    if 'cifar' in args.data_path.lower():
        dataset_name = 'cifar10'
    elif 'imagenet1k' in args.data_path.lower() or 'ilsvrc2012' in args.data_path.lower():
        dataset_name = 'imagenet1k'
    elif 'imagenet100' in args.data_path.lower():
        dataset_name = 'imagenet100'
    else:
        # Default to imagenet1k or use the dataset_name from args if provided
        dataset_name = getattr(args, 'dataset_name', 'imagenet1k')
        print(f"Using dataset: {dataset_name}")
    
    # Extract model name and patch size from args
    model_name = args.arch  # e.g., 'vit_base', 'vit_small', etc.
    patch_size = args.patch_size  # e.g., 8, 16, etc.
    
    # Use the same output directory as the training results by default
    output_dir = os.path.join(args.output_dir, 'saved_graphs')
    
    # For ImageNet-1K, we need to pass the dataset path to count actual class images
    dataset_path = None
    train_dir = "train"
    val_dir = "val"
    
    if dataset_name.lower() == 'imagenet1k':
        # Use the base directory (containing train and val directories)
        dataset_path = os.path.dirname(args.data_path) if 'train' in args.data_path else args.data_path
        
        # Check and adjust train/val directory names if needed
        if os.path.exists(os.path.join(dataset_path, 'ILSVRC2012_img_train')):
            train_dir = 'ILSVRC2012_img_train'
        
        if os.path.exists(os.path.join(dataset_path, 'ILSVRC2012_img_val')):
            val_dir = 'ILSVRC2012_img_val'
            
        print(f"Using ImageNet-1K from {dataset_path}")
        print(f"Train directory: {train_dir}")
        print(f"Validation directory: {val_dir}")
    
    # Get dataset configuration based on dataset name
    dataset_configs = {
        "cifar10": {"num_classes": 10, "num_train": 50000, "num_val": 10000, "feature_dim": 768},
        "imagenet100": {"num_classes": 100, "num_train": 130000, "num_val": 5000, "feature_dim": 768},
        "imagenet1k": {"num_classes": 1000, "num_train": 1281167, "num_val": 50000, "feature_dim": 384}
    }
    
    config = dataset_configs[dataset_name.lower()]
    num_train = config["num_train"]
    num_val = config["num_val"]
    feature_dim = config["feature_dim"]
    num_classes = config["num_classes"]
    
    # Load or generate feature embeddings and nearest neighbors
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading embeddings and nearest neighbors from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load teacher/student features and nearest neighbors from checkpoint
        teacher_features = checkpoint.get('teacher_features_cpu', None)
        teacher_nn_tensor = checkpoint.get('teacher_nn_tensor_cpu', None)
        teacher_sim_tensor = checkpoint.get('teacher_sim_tensor_cpu', None)
        
        student_features = checkpoint.get('student_features_cpu', None)
        student_nn_tensor = checkpoint.get('student_nn_tensor_cpu', None)
        student_sim_tensor = checkpoint.get('student_sim_tensor_cpu', None)
        
        if teacher_features is None or student_features is None:
            print("Warning: Features not found in checkpoint, generating random features")
            teacher_features = torch.randn(num_train, feature_dim)
            student_features = torch.randn(num_train, feature_dim)
        
        if teacher_nn_tensor is None or student_nn_tensor is None:
            print("Warning: Nearest neighbors not found in checkpoint, generating from features")
            # Normalize features
            teacher_features_norm = F.normalize(teacher_features, p=2, dim=1)
            student_features_norm = F.normalize(student_features, p=2, dim=1)
            
            # Compute similarities and get top-k nearest neighbors
            teacher_sim = teacher_features_norm @ teacher_features_norm.t()
            student_sim = student_features_norm @ student_features_norm.t()
            
            # Mask self-connections
            mask = torch.eye(teacher_sim.size(0)).bool()
            teacher_sim.masked_fill_(mask, -1)
            student_sim.masked_fill_(mask, -1)
            
            topk = args.topk if hasattr(args, 'topk') else 3
            teacher_sims, teacher_indices = teacher_sim.topk(k=topk, dim=1)
            student_sims, student_indices = student_sim.topk(k=topk, dim=1)
            
            teacher_nn_tensor = teacher_indices
            teacher_sim_tensor = teacher_sims
            student_nn_tensor = student_indices
            student_sim_tensor = student_sims
    else:
        print("No checkpoint provided, generating random features and nearest neighbors")
        # Generate random features (normalized)
        teacher_features = F.normalize(torch.randn(num_train, feature_dim), p=2, dim=1)
        student_features = F.normalize(torch.randn(num_train, feature_dim), p=2, dim=1)
        
        # Compute similarities and get top-k nearest neighbors
        teacher_sim = teacher_features @ teacher_features.t()
        student_sim = student_features @ student_features.t()
        
        # Mask self-connections
        mask = torch.eye(teacher_sim.size(0)).bool()
        teacher_sim.masked_fill_(mask, -1)
        student_sim.masked_fill_(mask, -1)
        
        topk = args.topk if hasattr(args, 'topk') else 3
        teacher_sims, teacher_indices = teacher_sim.topk(k=topk, dim=1)
        student_sims, student_indices = student_sim.topk(k=topk, dim=1)
        
        teacher_nn_tensor = teacher_indices
        teacher_sim_tensor = teacher_sims
        student_nn_tensor = student_indices
        student_sim_tensor = student_sims
    
    # Generate test features
    teacher_features_test = F.normalize(torch.randn(num_val, feature_dim), p=2, dim=1)
    student_features_test = F.normalize(torch.randn(num_val, feature_dim), p=2, dim=1)
    
    # Create teacher graph
    teacher_graph = Data(num_nodes=num_train)
    teacher_graph.x = teacher_features
    
    # Create student graph
    student_graph = Data(num_nodes=num_train)
    student_graph.x = student_features
    
    # Create test graphs
    teacher_graph_test = Data(num_nodes=num_val)
    teacher_graph_test.x = teacher_features_test
    
    student_graph_test = Data(num_nodes=num_val)
    student_graph_test.x = student_features_test
    
    # Build edge_index from nearest neighbors (using a subset of neighbors)
    edges_per_node = args.edges_per_node if hasattr(args, 'edges_per_node') else 1
    
    # Create edge_index tensors based on nearest neighbors
    def build_edge_index(nn_tensor, num_nodes, edges_per_node):
        # Use the top-k nearest neighbors to build edges
        k = min(edges_per_node, nn_tensor.size(1))
        
        # Source nodes (repeated for each neighbor)
        source_nodes = torch.arange(num_nodes).repeat_interleave(k)
        
        # Target nodes (k nearest neighbors for each node)
        target_nodes = nn_tensor[:, :k].reshape(-1)
        
        # Stack to form edge_index
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        
        return edge_index
    
    # Create edge indices
    teacher_edge_index = build_edge_index(teacher_nn_tensor, num_train, edges_per_node)
    student_edge_index = build_edge_index(student_nn_tensor, num_train, edges_per_node)
    
    # Create test edge indices (connect test nodes to similar training nodes)
    # For simplicity, we'll create random connections in this example
    teacher_edge_index_test = torch.stack([
        torch.randint(0, num_val, (num_val * edges_per_node,)),
        torch.randint(0, num_val, (num_val * edges_per_node,))
    ], dim=0)
    
    student_edge_index_test = torch.stack([
        torch.randint(0, num_val, (num_val * edges_per_node,)),
        torch.randint(0, num_val, (num_val * edges_per_node,))
    ], dim=0)
    
    # Assign edge indices to graphs
    teacher_graph.edge_index = teacher_edge_index
    student_graph.edge_index = student_edge_index
    teacher_graph_test.edge_index = teacher_edge_index_test
    student_graph_test.edge_index = student_edge_index_test
    
    print(f"Graph sizes:")
    print(f"Teacher graph: {teacher_graph.x.shape}, edges: {teacher_graph.edge_index.shape}")
    print(f"Student graph: {student_graph.x.shape}, edges: {student_graph.edge_index.shape}")
    print(f"Teacher test graph: {teacher_graph_test.x.shape}, edges: {teacher_graph_test.edge_index.shape}")
    print(f"Student test graph: {student_graph_test.x.shape}, edges: {student_graph_test.edge_index.shape}")
    
    # Concatenate train and test graphs before saving
    print("\nConcatenating train and test graphs...")
    
    # Calculate total number of nodes
    num_total = num_train + num_val
    
    # Create train/val masks
    train_mask = torch.zeros(num_total, dtype=torch.bool)
    train_mask[:num_train] = True
    val_mask = ~train_mask
    
    # Prepare teacher graph
    combined_teacher_graph = Data(num_nodes=num_total)
    combined_teacher_graph.x = torch.cat([teacher_graph.x, teacher_graph_test.x], dim=0)
    
    # Adjust test edge indices to account for concatenation
    test_edge_indices_teacher = teacher_graph_test.edge_index.clone()
    if test_edge_indices_teacher.numel() > 0 and test_edge_indices_teacher.max() >= 0:
        mask = test_edge_indices_teacher >= 0  # Filter out negative placeholder indices
        test_edge_indices_teacher[mask] += num_train
        
    combined_teacher_graph.edge_index = torch.cat([teacher_graph.edge_index, test_edge_indices_teacher], dim=1)
    combined_teacher_graph.train_mask = train_mask
    combined_teacher_graph.val_mask = val_mask
    
    # Prepare student graph
    combined_student_graph = Data(num_nodes=num_total)
    combined_student_graph.x = torch.cat([student_graph.x, student_graph_test.x], dim=0)
    
    # Adjust test edge indices for student graph
    test_edge_indices_student = student_graph_test.edge_index.clone()
    if test_edge_indices_student.numel() > 0 and test_edge_indices_student.max() >= 0:
        mask = test_edge_indices_student >= 0  # Filter out negative placeholder indices
        test_edge_indices_student[mask] += num_train
        
    combined_student_graph.edge_index = torch.cat([student_graph.edge_index, test_edge_indices_student], dim=1)
    combined_student_graph.train_mask = train_mask
    combined_student_graph.val_mask = val_mask
    
    # Create class labels based on dataset
    if dataset_name.lower() == 'imagenet1k':
        # For ImageNet-1K, read the actual class distribution from the cached file
        print("Using actual class distribution for ImageNet-1K")
        
        # Look for the cached imagenet_class_info.json
        json_files = [
            os.path.join(os.path.dirname(args.output_dir), "imagenet_class_info.json"),
            "/home/alij/SSL-GraphNNCLR/imagenet_class_info.json",
            os.path.join(output_dir, "imagenet_class_info.json")
        ]
        
        class_info_file = None
        for json_file in json_files:
            if os.path.exists(json_file):
                class_info_file = json_file
                break
                
        if class_info_file:
            print(f"Loading class distribution from: {class_info_file}")
            with open(class_info_file, 'r') as f:
                class_info = json.load(f)
                
            # Extract necessary information
            train_counts = OrderedDict(class_info['train']['class_counts'])
            val_counts = OrderedDict(class_info['val']['class_counts'])
            
            train_cumulative_counts = class_info['train']['cumulative_counts']
            val_cumulative_counts = class_info['val']['cumulative_counts']
            
            # Create the label tensors using actual class distributions
            y_train = torch.zeros(num_train, dtype=torch.long)
            y_val = torch.zeros(num_val, dtype=torch.long)
            
            # Assign class labels based on cumulative counts for training set
            for class_idx, (start, end) in enumerate(zip(train_cumulative_counts[:-1], train_cumulative_counts[1:])):
                y_train[start:end] = class_idx
                
            # Assign class labels based on cumulative counts for validation set
            for class_idx, (start, end) in enumerate(zip(val_cumulative_counts[:-1], val_cumulative_counts[1:])):
                y_val[start:end] = class_idx
                
            print(f"Created labels using actual class distributions")
        else:
            print("Warning: imagenet_class_info.json not found. Using equal distribution for ImageNet-1K as fallback.")
            # Fall back to equal distribution if the file is not found
            num_samples_per_class_train = num_train // num_classes
            num_samples_per_class_val = num_val // num_classes
            
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
    else:
        # For other datasets (CIFAR-10, ImageNet-100), assume equal distribution
        print(f"Using equal class distribution for {dataset_name}")
        
        num_samples_per_class_train = num_train // num_classes
        num_samples_per_class_val = num_val // num_classes
        
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
    
    # Combine train and val labels
    y = torch.cat([y_train, y_val], dim=0)
    
    # Add class labels to graphs
    combined_teacher_graph.y = y
    combined_student_graph.y = y
    
    print(f"Combined teacher graph: {combined_teacher_graph.x.shape}, edges: {combined_teacher_graph.edge_index.shape}")
    print(f"Combined student graph: {combined_student_graph.x.shape}, edges: {combined_student_graph.edge_index.shape}")
    
    # Save the graphs with only the combined versions
    output_path = save_graphs_from_training(
        teacher_graph=combined_teacher_graph,
        student_graph=combined_student_graph,
        teacher_graph_test=None,  # Pass None since we've already concatenated the graphs
        student_graph_test=None,  # Pass None since we've already concatenated the graphs
        dataset_name=dataset_name,
        model_name=model_name,
        edges_per_node=edges_per_node,
        patch_size=str(patch_size),
        output_dir=output_dir,
        dataset_path=dataset_path,
        train_dir=train_dir,
        val_dir=val_dir
    )
    
    print(f"Graphs saved to: {output_path}")
    return output_path


# Example of how this would be called from your main training loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save graphs from training")
    parser.add_argument('--data_path', default='/amin/imagenet/imagenet/train', type=str,
                        help='Path to the ImageNet training data.')
    parser.add_argument('--dataset_name', default='imagenet1k', type=str,
                        choices=['cifar10', 'imagenet100', 'imagenet1k'],
                        help='Dataset name (default: imagenet1k)')
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'],
                        help='Model architecture name')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for ViT models')
    parser.add_argument('--edges_per_node', default=5, type=int,
                        help='Number of edges per node in the graph')
    parser.add_argument('--topk', default=10, type=int,
                        help='Number of nearest neighbors to compute')
    parser.add_argument('--output_dir', default='./', type=str,
                        help='Output directory')
    parser.add_argument('--checkpoint_path', default='/amin/alij_cache/RESULTS/ImageNet/SSL-GraphNNCLR/Adasim-ImageNet1K/vits16/checkpoint.pth', type=str,
                        help='Path to a checkpoint to load features and neighbors')
    
    args = parser.parse_args()
    
    # Run the graph saving pipeline
    save_graphs_after_training(args, args.checkpoint_path)
    
    print("Example complete! In your real training script, call save_graphs_after_training() after training is done.") 