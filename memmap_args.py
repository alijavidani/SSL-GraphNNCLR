def add_memmap_args(parser):
    """
    Add memory mapping arguments to an existing ArgParser.
    
    Args:
        parser (argparse.ArgumentParser): Argument parser
    
    Returns:
        argparse.ArgumentParser: Updated argument parser
    """
    # Memory mapping options
    parser.add_argument('--use_memmap', action='store_true',
                        help="Use memory mapping for large tensors to reduce RAM usage")
    parser.add_argument('--memmap_dir', default='memmap_storage',
                        help="Directory to store memory-mapped files")
    parser.add_argument('--memmap_batch_size', type=int, default=1000,
                        help="Batch size for processing memory-mapped tensors")
    parser.add_argument('--convert_checkpoint', action='store_true',
                        help="Convert existing checkpoint to use memory mapping")
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help="Path to checkpoint to convert to memory mapping")
    parser.add_argument('--keep_in_ram', action='store_true',
                        help="Keep the original tensors in RAM after converting to memory maps")
    
    return parser 