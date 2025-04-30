#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from memmap_utils import MemmapTensor, create_memmap_tensor, torch_to_memmap
from graph_memmap import MemmapGraph
from torch_geometric.data import Data


def parse_args():
    parser = argparse.ArgumentParser(description='Convert AdaSim checkpoint tensors to memory maps')
    parser.add_argument('--checkpoint', type=str, default='/home/alij/SSL-GraphNNCLR/Results/Adasim-ImageNet1K/vitb8/checkpoint.pth', required=True,\
                        help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, default='/amin/alij_cache/RESULTS/ImageNet/SSL-GraphNNCLR/memmap_storage',
                        help='Directory to save memory maps')
    parser.add_argument('--keep_in_memory', action='store_true',
                        help='Keep the original tensors in the checkpoint')
    return parser.parse_args()


def convert_tensor_to_memmap(tensor, name, output_dir):
    """Convert a PyTorch tensor to a memory-mapped array."""
    print(f"Converting {name} with shape {tensor.shape} to memory map...")
    
    # Create memmap and copy data
    memmap_array = torch_to_memmap(tensor, f"{name}.npy", output_dir)
    
    print(f"  Saved to {os.path.join(output_dir, name + '.npy')}")
    
    return memmap_array


def convert_graph_to_memmap(graph, prefix, output_dir):
    """Convert a PyG graph to a memory-mapped graph."""
    print(f"Converting {prefix} graph to memory map...")
    
    graph_dir = os.path.join(output_dir, 'graphs')
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a memmap graph
    num_nodes = graph.num_nodes
    feature_dim = graph.x.shape[1] if hasattr(graph, 'x') else 0
    max_edges = graph.reserved_edge_index.shape[1] if hasattr(graph, 'reserved_edge_index') else graph.edge_index.shape[1]
    
    memmap_graph = MemmapGraph(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        max_edges=max_edges,
        dir_path=graph_dir,
        prefix=prefix
    )
    
    # Update memmap graph from PyG data
    memmap_graph.update_from_pyg_data(graph)
    
    print(f"  Saved to {graph_dir}/{prefix}_*.npy")
    
    return memmap_graph


def convert_checkpoint(checkpoint_path, output_dir, keep_in_memory=False):
    """Convert tensors in a checkpoint to memory maps."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Track converted tensors and their memory map info
    memmap_info = {}
    
    # Convert large tensors to memory maps
    tensors_to_convert = [
        'teacher_features_cpu',
        'teacher_nn_tensor_cpu',
        'teacher_sim_tensor_cpu',
        'teacher_nn_matrix_cpu',
        'teacher_sim_matrix_cpu',
        'student_features_cpu',
        'student_nn_tensor_cpu',
        'student_sim_tensor_cpu',
        'student_nn_matrix_cpu',
        'student_sim_matrix_cpu'
    ]
    
    for tensor_name in tensors_to_convert:
        if tensor_name in checkpoint:
            # Convert tensor to memmap
            memmap_array = convert_tensor_to_memmap(
                checkpoint[tensor_name], tensor_name, output_dir
            )
            
            # Store memmap info
            memmap_info[tensor_name] = {
                'dir_path': output_dir,
                'filename': f"{tensor_name}.npy",
                'shape': memmap_array.shape,
                'dtype': str(memmap_array.dtype)
            }
            
            # Optionally remove tensor from checkpoint to save memory
            if not keep_in_memory:
                checkpoint[tensor_name] = None
    
    # Convert graphs to memory maps
    graphs_to_convert = [
        'teacher_graph',
        'student_graph',
        'teacher_graph_test',
        'student_graph_test'
    ]
    
    for graph_name in graphs_to_convert:
        if graph_name in checkpoint:
            # Convert graph to memmap
            memmap_graph = convert_graph_to_memmap(
                checkpoint[graph_name], graph_name, output_dir
            )
            
            # Store memmap info
            memmap_info[graph_name] = {
                'dir_path': os.path.join(output_dir, 'graphs'),
                'prefix': graph_name,
                'num_nodes': memmap_graph.num_nodes,
                'feature_dim': memmap_graph.feature_dim,
                'max_edges': memmap_graph.max_edges
            }
            
            # Optionally remove graph from checkpoint to save memory
            if not keep_in_memory:
                checkpoint[graph_name] = None
    
    # Add memmap info to checkpoint
    checkpoint['memmap_info'] = memmap_info
    
    # Save updated checkpoint
    new_checkpoint_path = checkpoint_path + '.memmap'
    print(f"Saving updated checkpoint to {new_checkpoint_path}...")
    torch.save(checkpoint, new_checkpoint_path)
    
    print("Conversion complete!")
    print(f"Memory-mapped files are in: {output_dir}")
    print(f"Updated checkpoint is: {new_checkpoint_path}")
    
    return new_checkpoint_path


if __name__ == '__main__':
    args = parse_args()
    convert_checkpoint(args.checkpoint, args.output_dir, args.keep_in_memory) 