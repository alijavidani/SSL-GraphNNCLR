# Memory Mapping Implementation Guide for AdaSim

This guide explains how to modify the AdaSim code to use memory mapping for large data structures, allowing you to run experiments with ImageNet-1K without exhausting RAM.

## Overview

The following variables need to be memory-mapped:
1. `teacher_features_cpu` / `student_features_cpu` - Feature vectors
2. `teacher_nn_tensor_cpu` / `student_nn_tensor_cpu` - Nearest neighbor indices
3. `teacher_sim_tensor_cpu` / `student_sim_tensor_cpu` - Similarity scores
4. `teacher_nn_matrix_cpu` / `student_nn_matrix_cpu` - Nearest neighbor history 
5. `teacher_sim_matrix_cpu` / `student_sim_matrix_cpu` - Similarity history
6. `teacher_graph` / `student_graph` / `teacher_graph_test` / `student_graph_test` - Graph structures

## Step 1: Add Imports

Add these imports at the top of `main_adasim.py`:

```python
from memmap_utils import MemmapTensor, create_memmap_tensor, load_memmap_tensor
from graph_memmap import MemmapGraph
```

## Step 2: Modify Initialization 

Replace tensor initializations with memory-mapped versions. Look for these lines:

```python
teacher_features_cpu = torch.zeros(len(data_loader.dataset), embed_dim, dtype=torch.float)
teacher_nn_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.long)
teacher_sim_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.float)
teacher_nn_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.long)
teacher_sim_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.float)

student_features_cpu = torch.zeros(len(data_loader.dataset), embed_dim, dtype=torch.float)
student_nn_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.long)
student_sim_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.float)
student_nn_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.long)
student_sim_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.float)
```

Replace with:

```python
# Create memmap directory
memmap_dir = os.path.join(args.output_dir, 'memmap_storage')
Path(memmap_dir).mkdir(parents=True, exist_ok=True)

# Initialize memmap tensors
num_samples = len(data_loader.dataset)
teacher_features_memmap = MemmapTensor(
    shape=(num_samples, embed_dim),
    dtype=np.float32,
    filename="teacher_features.npy",
    dir_path=memmap_dir
)
teacher_nn_tensor_memmap = MemmapTensor(
    shape=(num_samples, args.topk),
    dtype=np.int64,
    filename="teacher_nn_tensor.npy",
    dir_path=memmap_dir
)
teacher_sim_tensor_memmap = MemmapTensor(
    shape=(num_samples, args.topk),
    dtype=np.float32,
    filename="teacher_sim_tensor.npy",
    dir_path=memmap_dir
)
teacher_nn_matrix_memmap = MemmapTensor(
    shape=(num_samples, args.vote_nn_nb, args.topk),
    dtype=np.int64,
    filename="teacher_nn_matrix.npy",
    dir_path=memmap_dir
)
teacher_sim_matrix_memmap = MemmapTensor(
    shape=(num_samples, args.vote_nn_nb, args.topk),
    dtype=np.float32,
    filename="teacher_sim_matrix.npy",
    dir_path=memmap_dir
)

# Student tensors
student_features_memmap = MemmapTensor(
    shape=(num_samples, embed_dim),
    dtype=np.float32,
    filename="student_features.npy",
    dir_path=memmap_dir
)
student_nn_tensor_memmap = MemmapTensor(
    shape=(num_samples, args.topk),
    dtype=np.int64,
    filename="student_nn_tensor.npy",
    dir_path=memmap_dir
)
student_sim_tensor_memmap = MemmapTensor(
    shape=(num_samples, args.topk),
    dtype=np.float32,
    filename="student_sim_tensor.npy",
    dir_path=memmap_dir
)
student_nn_matrix_memmap = MemmapTensor(
    shape=(num_samples, args.vote_nn_nb, args.topk),
    dtype=np.int64,
    filename="student_nn_matrix.npy",
    dir_path=memmap_dir
)
student_sim_matrix_memmap = MemmapTensor(
    shape=(num_samples, args.vote_nn_nb, args.topk),
    dtype=np.float32,
    filename="student_sim_matrix.npy",
    dir_path=memmap_dir
)

# Create CPU tensors from memory-mapped arrays (for compatibility)
teacher_features_cpu = torch.from_numpy(np.array(teacher_features_memmap.memmap))
teacher_nn_tensor_cpu = torch.from_numpy(np.array(teacher_nn_tensor_memmap.memmap))
teacher_sim_tensor_cpu = torch.from_numpy(np.array(teacher_sim_tensor_memmap.memmap))
teacher_nn_matrix_cpu = torch.from_numpy(np.array(teacher_nn_matrix_memmap.memmap))
teacher_sim_matrix_cpu = torch.from_numpy(np.array(teacher_sim_matrix_memmap.memmap))

student_features_cpu = torch.from_numpy(np.array(student_features_memmap.memmap))
student_nn_tensor_cpu = torch.from_numpy(np.array(student_nn_tensor_memmap.memmap))
student_sim_tensor_cpu = torch.from_numpy(np.array(student_sim_tensor_memmap.memmap))
student_nn_matrix_cpu = torch.from_numpy(np.array(student_nn_matrix_memmap.memmap))
student_sim_matrix_cpu = torch.from_numpy(np.array(student_sim_matrix_memmap.memmap))
```

## Step 3: Replace Graph Initialization

Replace PyG graph initialization with memory-mapped versions:

```python
# Initialize PyG Data objects for graphs
num_nodes = len(data_loader.dataset)
feature_dim = embed_dim
max_edges_per_node = args.edges_per_node * args.vote_nn_nb
max_edges = num_nodes * max_edges_per_node

# Create memory-mapped graphs
teacher_memmap_graph = MemmapGraph(
    num_nodes=num_nodes,
    feature_dim=feature_dim,
    max_edges=max_edges,
    dir_path=os.path.join(memmap_dir, 'graphs'),
    prefix='teacher'
)
student_memmap_graph = MemmapGraph(
    num_nodes=num_nodes,
    feature_dim=feature_dim,
    max_edges=max_edges,
    dir_path=os.path.join(memmap_dir, 'graphs'),
    prefix='student'
)

# Convert to PyG Data objects (for compatibility)
teacher_graph = teacher_memmap_graph.to_pyg_data(device)
student_graph = student_memmap_graph.to_pyg_data(device)

# Test graphs
num_nodes_test = len(data_loader_test.dataset)
max_edges_test = num_nodes_test * max_edges_per_node

teacher_memmap_graph_test = MemmapGraph(
    num_nodes=num_nodes_test,
    feature_dim=feature_dim,
    max_edges=max_edges_test,
    dir_path=os.path.join(memmap_dir, 'graphs'),
    prefix='teacher_test'
)
student_memmap_graph_test = MemmapGraph(
    num_nodes=num_nodes_test,
    feature_dim=feature_dim,
    max_edges=max_edges_test,
    dir_path=os.path.join(memmap_dir, 'graphs'),
    prefix='student_test'
)

teacher_graph_test = teacher_memmap_graph_test.to_pyg_data(device)
student_graph_test = student_memmap_graph_test.to_pyg_data(device)
```

## Step 4: Update Checkpoint Loading/Saving

Modify the checkpoint loading/saving code to handle memory-mapped arrays:

```python
# Add these variables to to_restore dictionary
to_restore = {"epoch": 0,
              "teacher_features_memmap": teacher_features_memmap,
              "teacher_nn_tensor_memmap": teacher_nn_tensor_memmap,
              "teacher_sim_tensor_memmap": teacher_sim_tensor_memmap,
              "teacher_nn_matrix_memmap": teacher_nn_matrix_memmap,
              "teacher_sim_matrix_memmap": teacher_sim_matrix_memmap,
              "student_features_memmap": student_features_memmap,
              "student_nn_tensor_memmap": student_nn_tensor_memmap,
              "student_sim_tensor_memmap": student_sim_tensor_memmap,
              "student_nn_matrix_memmap": student_nn_matrix_memmap,
              "student_sim_matrix_memmap": student_sim_matrix_memmap,
              "teacher_memmap_graph": teacher_memmap_graph,
              "student_memmap_graph": student_memmap_graph,
              "teacher_memmap_graph_test": teacher_memmap_graph_test,
              "student_memmap_graph_test": student_memmap_graph_test,
              # ... existing entries ...
             }
```

And modify the checkpoint saving:

```python
save_dict = {
    'student': student.state_dict(),
    'teacher': teacher.state_dict(),
    # ... other existing entries ...
    'epoch': epoch + 1,
    # Add memmap info
    'teacher_features_memmap_info': {
        'dir_path': teacher_features_memmap.dir_path,
        'filename': teacher_features_memmap.filename,
        'shape': teacher_features_memmap.shape,
        'dtype': str(teacher_features_memmap.dtype)
    },
    # Add similar info for other memmap tensors and graphs
}
```

## Step 5: Update State Update Functions

Modify the `update_state` and `update_nn` functions to work with memory-mapped arrays:

```python
def update_state(feats_local, indices_local, sims_knn_local, indices_knn_local, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor, requires_grad, features_memmap=None, nn_tensor_memmap=None, sim_tensor_memmap=None):
    # ... original code ...
    
    # Also update the memmap tensors
    if features_memmap is not None:
        features_memmap.update_from_torch(features_batch_all, indices_batch_all.cpu().numpy())
    if nn_tensor_memmap is not None:
        nn_tensor_memmap.update_from_torch(indices_knn_batch_all, indices_batch_all.cpu().numpy())
    if sim_tensor_memmap is not None:
        sim_tensor_memmap.update_from_torch(sims_knn_batch_all, indices_batch_all.cpu().numpy())
```

## Step 6: Update Graph-Related Functions

Modify the graph update functions to work with memory-mapped graphs:

```python
def update_graph_from_model_multi_gpu(images, indices, model, features, graph, len_train, args, memmap_graph=None):
    # ... original code with graph updates ...
    
    # Update memmap graph if provided
    if memmap_graph is not None:
        memmap_graph.update_from_pyg_data(graph)
    
    return graph
```

## Step 7: Add Command Line Arguments

Add command line arguments to control memory mapping:

```python
parser.add_argument('--use_memmap', default=True, type=utils.bool_flag,
                    help="Whether to use memory mapping for large tensors")
parser.add_argument('--memmap_dir', default='memmap_storage',
                    help="Directory to store memory-mapped files")
```

## Step 8: Add Batch Processing for Large Operations

For operations that need to process the entire dataset, implement batch processing:

```python
def process_in_batches(func, data_size, batch_size=1000, *args, **kwargs):
    """Process large data in batches to avoid memory issues."""
    results = []
    for start_idx in range(0, data_size, batch_size):
        end_idx = min(start_idx + batch_size, data_size)
        batch_result = func(start_idx, end_idx, *args, **kwargs)
        results.append(batch_result)
    return results
```

## Implementation Notes

1. **Memory Management**: Memory-mapped files will still use page cache in RAM, but the OS can swap them out if needed.
   
2. **Performance**: There might be performance implications due to disk I/O. Using SSDs (preferably NVMe) is recommended.

3. **Distributed Training**: In a distributed setting, ensure each process has access to the memory-mapped files. Best to use a shared filesystem.

4. **Checkpointing**: Don't save the full memory-mapped arrays in checkpoints; save file references instead.

5. **Error Handling**: Add proper error handling for disk I/O operations.

## Testing

Test with smaller dataset first to ensure memory mapping works correctly before scaling to ImageNet-1K. 