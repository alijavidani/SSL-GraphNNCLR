import os
import numpy as np
import torch
from pathlib import Path


def create_memmap_tensor(shape, dtype, filename, dir_path='./memmap_storage'):
    """
    Create a memory-mapped array for a tensor.
    
    Args:
        shape (tuple): Shape of the tensor
        dtype (np.dtype): Data type of the tensor
        filename (str): Name of the memmap file
        dir_path (str): Directory to store the memmap files
        
    Returns:
        np.memmap: Memory-mapped array
    """
    # Create directory if it doesn't exist
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Full path to memmap file
    file_path = os.path.join(dir_path, filename)
    
    # Create memmap array
    memmap_array = np.memmap(
        file_path,
        dtype=dtype,
        mode='w+',  # Create or overwrite
        shape=shape
    )
    
    return memmap_array


def load_memmap_tensor(filename, shape, dtype, dir_path='./memmap_storage', mode='r+'):
    """
    Load an existing memory-mapped array.
    
    Args:
        filename (str): Name of the memmap file
        shape (tuple): Shape of the tensor
        dtype (np.dtype): Data type of the tensor
        dir_path (str): Directory where memmap files are stored
        mode (str): Mode to open the file ('r+', 'r', 'w+', etc.)
        
    Returns:
        np.memmap: Memory-mapped array
    """
    file_path = os.path.join(dir_path, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Memmap file {file_path} not found")
    
    # Load memmap array
    memmap_array = np.memmap(
        file_path,
        dtype=dtype,
        mode=mode,
        shape=shape
    )
    
    return memmap_array


def torch_to_memmap(tensor, filename, dir_path='./memmap_storage'):
    """
    Convert a torch tensor to a memory-mapped array.
    
    Args:
        tensor (torch.Tensor): PyTorch tensor to convert
        filename (str): Name of the memmap file
        dir_path (str): Directory to store the memmap files
        
    Returns:
        np.memmap: Memory-mapped array
    """
    # Convert tensor to numpy
    numpy_array = tensor.cpu().numpy()
    
    # Create memmap array
    memmap_array = create_memmap_tensor(
        shape=numpy_array.shape,
        dtype=numpy_array.dtype,
        filename=filename,
        dir_path=dir_path
    )
    
    # Copy data to memmap array
    memmap_array[:] = numpy_array[:]
    memmap_array.flush()
    
    return memmap_array


def memmap_to_torch(memmap_array, device='cpu'):
    """
    Convert a memory-mapped array to a torch tensor.
    
    Args:
        memmap_array (np.memmap): Memory-mapped array
        device (str): Device to put the tensor on
        
    Returns:
        torch.Tensor: PyTorch tensor
    """
    # Convert memmap to tensor
    tensor = torch.from_numpy(np.array(memmap_array)).to(device)
    return tensor


def update_memmap_from_torch(memmap_array, tensor, indices=None):
    """
    Update a memory-mapped array from a torch tensor.
    
    Args:
        memmap_array (np.memmap): Memory-mapped array to update
        tensor (torch.Tensor): PyTorch tensor with updated values
        indices (torch.Tensor, optional): Indices to update. If None, update all.
    """
    if indices is None:
        # Update entire array
        memmap_array[:] = tensor.cpu().numpy()[:]
    else:
        # Update specific indices
        cpu_indices = indices.cpu().numpy()
        memmap_array[cpu_indices] = tensor.cpu().numpy()[cpu_indices]
    
    memmap_array.flush()


class MemmapTensor:
    """
    A class that wraps a memory-mapped array to provide a tensor-like interface.
    """
    def __init__(self, shape, dtype, filename, dir_path='./memmap_storage', existing=False, mode='r+'):
        # Store shape in a private attribute
        self._shape = shape
        self.dtype = dtype
        self.filename = filename
        self.dir_path = dir_path
        
        if existing:
            self.memmap = load_memmap_tensor(filename, shape, dtype, dir_path, mode)
        else:
            self.memmap = create_memmap_tensor(shape, dtype, filename, dir_path)
    
    def to_torch(self, device='cpu'):
        """Convert to torch tensor on specified device"""
        return memmap_to_torch(self.memmap, device)
    
    def update_from_torch(self, tensor, indices=None):
        """Update memmap from torch tensor"""
        update_memmap_from_torch(self.memmap, tensor, indices)
    
    def __getitem__(self, idx):
        """Get item by index, returns a numpy array"""
        return self.memmap[idx]
    
    def __setitem__(self, idx, value):
        """Set item by index"""
        self.memmap[idx] = value
        self.memmap.flush()
    
    @property
    def shape(self):
        """Get shape of memmap tensor"""
        # Use the memmap shape if available, otherwise fall back to _shape
        if hasattr(self, 'memmap') and self.memmap is not None:
            return self.memmap.shape
        return self._shape
    
    def flush(self):
        """Force flush changes to disk"""
        self.memmap.flush() 