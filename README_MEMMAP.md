# Using Memory Mapping with AdaSim for ImageNet-1K

This guide explains how to use memory mapping in AdaSim to efficiently handle ImageNet-1K and other large datasets that don't fit in RAM.

## Overview

Memory mapping allows large arrays to be stored on disk while still being accessed as if they were in memory. The operating system handles paging data between disk and RAM as needed, which dramatically reduces memory usage while maintaining reasonable performance.

## Setup

1. First, make sure you have the necessary utilities:

```bash
# Check if files exist
ls memmap_utils.py graph_memmap.py convert_to_memmap.py memmap_args.py

# If not, create them using the provided scripts
```

2. Install necessary dependencies:

```bash
pip install numpy torch torch_geometric
```

## Memory Map Implementation Details

The memory mapping implementation consists of:

1. **memmap_utils.py**: Basic utilities for memory mapping tensors
2. **graph_memmap.py**: Memory mapping for PyG graph data structures
3. **convert_to_memmap.py**: Tool to convert existing checkpoints to use memory mapping
4. **memmap_args.py**: Adds memory mapping command line arguments to the parser

## Usage Options

There are two main approaches to use memory mapping with AdaSim:

### Option 1: Start Training from Scratch with Memory Mapping

1. Modify `main_adasim.py` to import and use memory mapping:

```python
# Add at top of main_adasim.py
from memmap_utils import MemmapTensor, torch_to_memmap
from graph_memmap import MemmapGraph
from memmap_args import add_memmap_args

# In get_args_parser()
parser = add_memmap_args(parser)
```

2. Initialize your data structures with memory mapping in `train_adasim()`:

```python
# Add after loading data
if args.use_memmap:
    # Create memmap directory
    memmap_dir = os.path.join(args.output_dir, args.memmap_dir)
    Path(memmap_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize teacher/student features, nn_tensors, etc. with memory mapping
    # (Follow the implementation guide in memmap_implementation_guide.md)
```

### Option 2: Convert Existing Checkpoint to Use Memory Mapping

1. Run the conversion tool on an existing checkpoint:

```bash
python convert_to_memmap.py --checkpoint /path/to/checkpoint.pth --output_dir /path/to/memmap_storage
```

2. Update your code to load the memory-mapped checkpoint:

```python
# Load memory-mapped checkpoint
checkpoint = torch.load(args.checkpoint_path)
memmap_info = checkpoint.get('memmap_info', {})

# Load tensors from memmap files using the info
if 'teacher_features_cpu' in memmap_info:
    info = memmap_info['teacher_features_cpu']
    teacher_features_memmap = MemmapTensor(
        shape=info['shape'], 
        dtype=np.dtype(info['dtype']),
        filename=info['filename'],
        dir_path=info['dir_path'],
        existing=True
    )
    teacher_features_cpu = torch.from_numpy(np.array(teacher_features_memmap.memmap))
    # ... similarly for other tensors
```

## Best Practices for Memory Mapping with ImageNet-1K

1. **Storage Requirements**: Make sure you have sufficient disk space (preferably SSD) for memory-mapped files:
   - Feature tensors: ~20GB (1.28M samples × 768-dim embeddings × 4 bytes × 2 for teacher/student)
   - Graph structures: ~10-30GB depending on connections
   - Total: ~50-100GB for full ImageNet-1K

2. **Performance Optimization**:
   - Use SSDs (preferably NVMe) for memory-mapped storage
   - Process data in batches to minimize page faults
   - Consider using a RAM disk if you have some RAM to spare but not enough for all tensors

3. **Distributed Training**:
   - Use a shared filesystem for memory-mapped files in distributed settings
   - Ensure proper synchronization between processes

4. **Checkpointing**:
   - Don't save the full memory-mapped arrays in checkpoints
   - Save file paths and metadata instead

## Example Command

```bash
python main_adasim.py \
    --use_memmap \
    --memmap_dir /path/to/ssd/memmap_storage \
    --memmap_batch_size 1000 \
    --data_path /path/to/imagenet \
    --output_dir ./output_imagenet1k \
    --batch_size_per_gpu 64 \
    --epochs 100
```

## Troubleshooting

1. **Slow Performance**: 
   - Check if your storage is fast enough (SSD recommended)
   - Increase `memmap_batch_size` to reduce disk I/O

2. **Out of Disk Space**:
   - Free up disk space or use a larger drive
   - Consider reducing precision (e.g., fp16 instead of fp32)

3. **Process Killed**:
   - OS might kill process if it still uses too much memory
   - Check actual memory usage and adjust batch sizes

## Memory Usage Comparison

| Configuration | RAM Usage (Approx.) |
|---------------|---------------------|
| Standard      | 250-300GB           |
| With Memory Mapping | 30-50GB       |

## Performance Impact

Memory mapping generally introduces some I/O overhead, but with proper settings:
- On NVMe SSD: ~10% slowdown
- On SATA SSD: ~15-20% slowdown
- On HDD: Significant slowdown (not recommended)

## References

- NumPy Memmap documentation: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- Operating System Paging: https://en.wikipedia.org/wiki/Paging 