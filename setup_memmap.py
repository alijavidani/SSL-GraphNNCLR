#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import argparse
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Setup memory mapping for AdaSim')
    parser.add_argument('--backup_main', action='store_true',
                        help='Backup the original main_adasim.py file')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check if required files exist')
    return parser.parse_args()

def check_files_exist():
    """Check if required files exist."""
    required_files = [
        'memmap_utils.py',
        'graph_memmap.py',
        'convert_to_memmap.py',
        'memmap_args.py',
        'memmap_implementation_guide.md',
        'README_MEMMAP.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def backup_main():
    """Backup main_adasim.py file."""
    if os.path.exists('main_adasim.py'):
        shutil.copy('main_adasim.py', 'main_adasim_backup.py')
        print("Backed up main_adasim.py to main_adasim_backup.py")
    else:
        print("Warning: main_adasim.py not found")

def setup_memmap():
    """
    Prepare the project for memory mapping by adding necessary imports
    and modifying the main_adasim.py file.
    """
    if not os.path.exists('main_adasim.py'):
        print("Error: main_adasim.py not found")
        return False
    
    with open('main_adasim.py', 'r') as f:
        content = f.read()
    
    # Add imports
    import_block = """
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"

import torch
import argparse
import sys
import datetime
import time
import math
import json
from pathlib import Path
# Memory mapping imports
from memmap_utils import MemmapTensor, create_memmap_tensor, load_memmap_tensor
from graph_memmap import MemmapGraph
from memmap_args import add_memmap_args
"""
    
    content = content.replace("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"

import torch
import argparse
import sys
import datetime
import time
import math
import json
from pathlib import Path
""", import_block)
    
    # Modify args parsing
    if "parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])" in content:
        content = content.replace(
            "parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])",
            "parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])\nparser = add_memmap_args(parser)"
        )
    
    # Write modified content
    with open('main_adasim_memmap.py', 'w') as f:
        f.write(content)
    
    print("Created main_adasim_memmap.py with memory mapping imports")
    print("See memmap_implementation_guide.md for further implementation instructions")
    
    return True

def main():
    args = parse_args()
    
    # Check for required files
    missing_files = check_files_exist()
    if missing_files:
        print("Missing required files:", missing_files)
        print("Please create these files before proceeding")
        return
    
    if args.check_only:
        print("All required files exist")
        return
    
    # Backup original main file if requested
    if args.backup_main:
        backup_main()
    
    # Setup memory mapping
    setup_memmap()
    
    print("\nSetup complete! Next steps:")
    print("1. Follow the instructions in memmap_implementation_guide.md to complete the implementation")
    print("2. Read README_MEMMAP.md for usage guidance")
    print("3. Use convert_to_memmap.py to convert existing checkpoints")

if __name__ == '__main__':
    main() 