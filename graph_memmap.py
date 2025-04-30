import os
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data
from memmap_utils import create_memmap_tensor, load_memmap_tensor, memmap_to_torch, update_memmap_from_torch


class MemmapGraph:
    """
    A class that represents a PyG graph using memory-mapped arrays for nodes, edges, and features.
    """
    def __init__(self, num_nodes, feature_dim, max_edges, dir_path='./memmap_graphs', 
                 prefix='graph', existing=False, device=None):
        """
        Initialize a memory-mapped graph.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            feature_dim (int): Feature dimension for node embeddings
            max_edges (int): Maximum number of edges to allocate space for
            dir_path (str): Directory to store the memmap files
            prefix (str): Prefix for memmap filenames
            existing (bool): Whether to load existing memmap files
            device (str): Device to put tensors on when needed
        """
        self._num_nodes = num_nodes
        self._feature_dim = feature_dim
        self._max_edges = max_edges
        self.dir_path = dir_path
        self.prefix = prefix
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directory if it doesn't exist
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Define memmap files
        mode = 'r+' if existing else 'w+'
        
        if existing:
            # Load existing memmap arrays
            self.node_features = load_memmap_tensor(
                f"{prefix}_node_features.npy",
                shape=(num_nodes, feature_dim),
                dtype=np.float32,
                dir_path=dir_path,
                mode=mode
            )
            
            self.edge_index = load_memmap_tensor(
                f"{prefix}_edge_index.npy",
                shape=(2, max_edges),
                dtype=np.int64,
                dir_path=dir_path,
                mode=mode
            )
            
            self.edge_counts = load_memmap_tensor(
                f"{prefix}_edge_counts.npy",
                shape=(num_nodes,),
                dtype=np.int64,
                dir_path=dir_path,
                mode=mode
            )
            
            # Load auxiliary information
            edge_info_path = os.path.join(dir_path, f"{prefix}_edge_info.npz")
            if os.path.exists(edge_info_path):
                edge_info = np.load(edge_info_path)
                self.current_edge_count = int(edge_info['current_edge_count'])
            else:
                self.current_edge_count = 0
                
        else:
            # Create new memmap arrays
            self.node_features = create_memmap_tensor(
                shape=(num_nodes, feature_dim),
                dtype=np.float32,
                filename=f"{prefix}_node_features.npy",
                dir_path=dir_path
            )
            
            self.edge_index = create_memmap_tensor(
                shape=(2, max_edges),
                dtype=np.int64,
                filename=f"{prefix}_edge_index.npy",
                dir_path=dir_path
            )
            
            self.edge_counts = create_memmap_tensor(
                shape=(num_nodes,),
                dtype=np.int64,
                filename=f"{prefix}_edge_counts.npy",
                dir_path=dir_path
            )
            
            self.current_edge_count = 0
    
    @property
    def num_nodes(self):
        return self._num_nodes
    
    @property
    def feature_dim(self):
        return self._feature_dim
    
    @property
    def max_edges(self):
        return self._max_edges
    
    def save_info(self):
        """Save auxiliary information about the graph"""
        info_path = os.path.join(self.dir_path, f"{self.prefix}_edge_info.npz")
        np.savez(info_path, current_edge_count=self.current_edge_count)
    
    def to_pyg_data(self, device=None):
        """
        Convert to PyG Data object.
        
        Args:
            device (str): Device to put the tensor on
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        if device is None:
            device = self.device
            
        # Convert memmaps to torch tensors
        x = torch.from_numpy(np.array(self.node_features)).to(device)
        edge_index = torch.from_numpy(np.array(self.edge_index[:, :self.current_edge_count])).to(device)
        edge_counts = torch.from_numpy(np.array(self.edge_counts)).to(device)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_counts=edge_counts,
            num_nodes=self.num_nodes
        )
        
        # Add reserved edge index for compatibility
        data.reserved_edge_index = torch.from_numpy(np.array(self.edge_index)).to(device)
        
        return data
    
    def update_from_pyg_data(self, data):
        """
        Update memmap arrays from PyG Data object.
        
        Args:
            data (Data): PyTorch Geometric Data object
        """
        # Update node features
        update_memmap_from_torch(self.node_features, data.x.cpu())
        
        # Update edge index
        if hasattr(data, 'reserved_edge_index'):
            update_memmap_from_torch(self.edge_index, data.reserved_edge_index.cpu())
        else:
            # If no reserved_edge_index, use edge_index and pad with zeros
            edge_index_np = data.edge_index.cpu().numpy()
            self.edge_index[:, :edge_index_np.shape[1]] = edge_index_np
            self.edge_index.flush()
        
        # Update edge counts
        update_memmap_from_torch(self.edge_counts, data.edge_counts.cpu())
        
        # Update edge count
        self.current_edge_count = data.edge_index.shape[1]
        self.save_info()
    
    def update_node_features(self, indices, features):
        """
        Update node features for specific indices.
        
        Args:
            indices (torch.Tensor): Node indices to update
            features (torch.Tensor): New features for these nodes
        """
        indices_np = indices.cpu().numpy()
        features_np = features.cpu().numpy()
        
        self.node_features[indices_np] = features_np
        self.node_features.flush()
    
    def add_edges(self, src_nodes, dst_nodes):
        """
        Add edges to the graph.
        
        Args:
            src_nodes (torch.Tensor): Source node indices
            dst_nodes (torch.Tensor): Destination node indices
        """
        src_np = src_nodes.cpu().numpy()
        dst_np = dst_nodes.cpu().numpy()
        
        num_new_edges = len(src_np)
        
        if self.current_edge_count + num_new_edges > self.max_edges:
            raise ValueError(f"Cannot add {num_new_edges} more edges. Max edge limit reached.")
        
        # Add new edges
        self.edge_index[0, self.current_edge_count:self.current_edge_count+num_new_edges] = src_np
        self.edge_index[1, self.current_edge_count:self.current_edge_count+num_new_edges] = dst_np
        
        # Update edge counts for each source node
        for src in src_np:
            self.edge_counts[src] += 1
        
        self.current_edge_count += num_new_edges
        
        # Flush changes
        self.edge_index.flush()
        self.edge_counts.flush()
        self.save_info()
    
    def get_edges_for_node(self, node_idx):
        """
        Get all edges for a specific node.
        
        Args:
            node_idx (int): Node index
            
        Returns:
            np.ndarray: Array of destination node indices
        """
        # Find all edges with this node as source
        mask = self.edge_index[0, :self.current_edge_count] == node_idx
        dst_nodes = self.edge_index[1, :self.current_edge_count][mask]
        
        return dst_nodes 