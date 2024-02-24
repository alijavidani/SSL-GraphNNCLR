import os
import torch
import argparse
import sys
import datetime
import time
import math
import json
from pathlib import Path
from adasim_utils.loss import AdaSimLoss
from adasim_utils.dataset import DatasetFolderAdaSim
import subprocess
import tarfile
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from adasim_utils.parser import get_args_parser
import igraph as ig
from main_adasim import DataAugmentationDINO

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()
if 'CXI_FORK_SAFE_HP' in os.environ:
    del os.environ['CXI_FORK_SAFE_HP']
if 'CXI_FORK_SAFE' in os.environ:
    del os.environ['CXI_FORK_SAFE']
utils.init_distributed_mode(args)
utils.fix_random_seeds(args.seed)
print("git:\n  {}\n".format(utils.get_sha()))
print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
cudnn.benchmark = True

# ============ preparing data ... ============
transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args
    )
dataset = DatasetFolderAdaSim(args.data_path, args, transform=transform, return_index_instead_of_target=False)
sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=args.batch_size_per_gpu,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
)
print(f"Data loaded: there are {len(dataset)} images.")

# Initialize an empty dictionary for mapping
index_to_image_name = {}

metric_logger = utils.MetricLogger(delimiter="  ")
header = 'Epoch: [{}/{}]'.format(1, args.epochs)
for it, (images, indices, same_im_bool, neighbors) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    #update graph:
    if neighbors !=[]:
        edge_list_updates = []
        for i in range(len(indices)):
            for j in range(neighbors.shape[1]):
                edge_list_updates.append((indices[i],neighbors[i,j]))
        # graph.add_edges(edge_list_updates)

# def get_image_name():

# Assuming 'dataloader' is your DataLoader object
# for it, (image, index, _) in enumerate(data_loader):
#     print(image)