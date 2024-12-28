# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"#1,2,3

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
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from adasim_utils.parser import get_args_parser
import igraph as ig
import networkx as nx
from torch_geometric.utils import from_networkx
# import cudf
# import cugraph
from my_functions import *
from graph import *
from torch_geometric.data import Data
from augmentation import DataAugmentationDINO
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from combined_model import CombinedModel

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def train_adasim(args):
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

    ###############################################################################################
    start_copy_time = time.time()
    print('Start to copy')
    if len(args.untar_path) > 0 and args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]

    start_copy_time = time.time()
    if args.data_path.split('/')[-1] == 'ilsvrc2012.tar':
        if int(args.gpu) == 0:
            with tarfile.open(args.data_path, 'r') as f:
                f.extractall(args.untar_path)

            print('Time taken for untar:', time.time() - start_copy_time)
            print(os.listdir(args.untar_path))

        args.data_path = os.path.join(args.untar_path, 'ilsvrc2012', 'ILSVRC2012_img_train')
        args.data_path_val = os.path.join(args.untar_path, 'ilsvrc2012', 'ILSVRC2012_img_val')
    else:
        args.data_path = os.path.join(args.untar_path, 'train')
        args.data_path_val = os.path.join(args.untar_path, 'test')

    torch.distributed.barrier()

    index_label_int = make_index_label(args, transform)

    dataset = DatasetFolderAdaSim(args.data_path, args, transform=transform, return_index_instead_of_target=True)
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

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim

    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # Initialize the graph models
    # Note:
    input_dim = embed_dim # Input feature dimension per node
    hidden_dims = [512, embed_dim]  # [512, 1024, 512, embed_dim] Hidden layer dimensions
    # output_dim = hidden_dims[-1]       # Output dimension (graph-level embedding)
    num_layers = len(hidden_dims)        # Number of GNN layers

    # Create DINOHead instance
    student_dino_head = DINOHead(
        in_dim = hidden_dims[-1],
        out_dim = args.out_dim,
        use_bn = args.use_bn_in_head,
        norm_last_layer = args.norm_last_layer,
    )

    teacher_dino_head = DINOHead(
        in_dim = hidden_dims[-1],
        out_dim = args.out_dim,
        use_bn = args.use_bn_in_head,
        norm_last_layer = args.norm_last_layer,
    )

    teacher_nn_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.long)


    # Initialize PyG Data objects for graphs
    # Number of nodes in the graph
    num_nodes = len(data_loader.dataset)
    feature_dim = embed_dim # or args.out_dim embed_dim

    teacher_graph = Data(num_nodes=num_nodes)
    student_graph = Data(num_nodes=num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_graph.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    student_graph.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

    teacher_graph.x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32).to(device)
    student_graph.x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32).to(device)

    # Ensure the graphs are on the correct device
    teacher_graph = teacher_graph.to(device)
    student_graph = student_graph.to(device)


    # ============ optionally resume training ... ============n
    to_restore = {"epoch": 0,
                  'teacher_nn_tensor_cpu': teacher_nn_tensor_cpu,
                #   'teacher_sim_tensor_cpu': teacher_sim_tensor_cpu,
                #   'teacher_features_cpu': teacher_features_cpu,
                #   "teacher_nn_matrix_cpu": teacher_nn_matrix_cpu,
                #   "teacher_sim_matrix_cpu": teacher_sim_matrix_cpu,
                #   'student_nn_tensor_cpu': student_nn_tensor_cpu,
                #   'student_sim_tensor_cpu': student_sim_tensor_cpu,
                #   'student_features_cpu': student_features_cpu,
                #   "student_nn_matrix_cpu": student_nn_matrix_cpu,
                #   "student_sim_matrix_cpu": student_sim_matrix_cpu,
                  "teacher_graph": teacher_graph,
                  "student_graph": student_graph,
                #   "optimizer": optimizer,
                  }
    default_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if not os.path.isfile(default_checkpoint_path) and os.path.isfile(args.start_checkpoint_path):
        utils.master_copy_from_to(args.start_checkpoint_path, default_checkpoint_path)

    torch.distributed.barrier()

    try:
        utils.restart_from_checkpoint(
            default_checkpoint_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            # student_graph_model=student_graph_model,
            # teacher_graph_model=teacher_graph_model,
            # student_combined_model=student_combined_model,
            # teacher_combined_model=teacher_combined_model,
            # optimizer=optimizer,
            # fp16_scaler=fp16_scaler,
            # adasim_loss=adasim_loss,
        )
    except:
        # If checkpoint is corrupted, used backedup checkpoint
        utils.restart_from_checkpoint(
            default_checkpoint_path + '.backup',
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            # student_graph_model=student_graph_model,
            # teacher_graph_model=teacher_graph_model,
            # student_combined_model=student_combined_model,
            # teacher_combined_model=teacher_combined_model,
            # optimizer=optimizer,
            # fp16_scaler=fp16_scaler,
            # adasim_loss=adasim_loss,
        )
    # start_epoch = to_restore["epoch"]
    # teacher_features_cpu = to_restore["teacher_features_cpu"]
    teacher_nn_tensor_cpu = to_restore["teacher_nn_tensor_cpu"]
    # teacher_sim_tensor_cpu = to_restore["teacher_sim_tensor_cpu"]
    # teacher_nn_matrix_cpu = to_restore["teacher_nn_matrix_cpu"]
    # teacher_sim_matrix_cpu = to_restore["teacher_sim_matrix_cpu"]

    # student_features_cpu = to_restore["student_features_cpu"]
    # student_nn_tensor_cpu = to_restore["student_nn_tensor_cpu"]
    # student_sim_tensor_cpu = to_restore["student_sim_tensor_cpu"]
    # student_nn_matrix_cpu = to_restore["student_nn_matrix_cpu"]
    # student_sim_matrix_cpu = to_restore["student_sim_matrix_cpu"]

    teacher_graph = to_restore["teacher_graph"]
    student_graph = to_restore["student_graph"]
 
    # optimizer=to_restore['optimizer']
    
    # Ensure the graphs are on the correct device
    teacher_graph = teacher_graph.to(device)
    student_graph = student_graph.to(device)


def get_nn_acuracy(dataset, nn_tensor):
    t = torch.Tensor(dataset.targets)
    return sum(t[nn_tensor] == t).item() / len(nn_tensor)


def get_nn_self_accuracy(nn_tensor):
    return sum(nn_tensor == torch.Tensor(range(len(nn_tensor)))).item() / len(nn_tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    # For benchmarking the code
    if args.epochs <= 2:
        args.warmup_teacher_temp_epochs = 0
        args.warmup_epochs = 0

    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_adasim(args)