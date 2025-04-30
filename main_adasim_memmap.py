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
from augmentation import DataAugmentationDINO, DataAugmentationDINO2
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from combined_model import CombinedModel
import numpy as np

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
        args.data_path_val = os.path.join(args.untar_path, 'val')

    torch.distributed.barrier()

    # index_label_int = make_index_label(args, transform)

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

    
    # ============ preparing data ... ============
    transform2 = DataAugmentationDINO2(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args
    )

    # Initialize the custom dataset
    test_dataset = DatasetFolderAdaSim(
        root=args.data_path_val,
        args=args,
        transform=transform2,
        return_index_instead_of_target=True
    )
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=False)

    # DataLoader Initialization
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )

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

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # student = utils.MultiCropWrapper(student)
    # teacher = utils.MultiCropWrapper(teacher)
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # Initialize the graph models
    # Note:
    input_dim = embed_dim # Input feature dimension per node
    hidden_dims = [512, embed_dim]  # [512, 1024, 512, embed_dim] Hidden layer dimensions
    # output_dim = hidden_dims[-1]       # Output dimension (graph-level embedding)
    num_layers = len(hidden_dims)        # Number of GNN layers

    # Create DINOHead instance
    # student_dino_head = DINOHead(
    #     in_dim = hidden_dims[-1],
    #     out_dim = args.out_dim,
    #     use_bn = args.use_bn_in_head,
    #     norm_last_layer = args.norm_last_layer,
    # )

    # teacher_dino_head = DINOHead(
    #     in_dim = hidden_dims[-1],
    #     out_dim = args.out_dim,
    #     use_bn = args.use_bn_in_head,
    #     norm_last_layer = args.norm_last_layer,
    # )

    # Student graph model
    # student_graph_model = GraphNetWithDINO(input_dim, hidden_dims, num_layers, student_dino_head).cuda()

    # Teacher graph model
    # teacher_graph_model = GraphNetWithDINO(input_dim, hidden_dims, num_layers, teacher_dino_head).cuda()

    # Synchronize batch norms (if any)
    # if utils.has_batchnorms(student_graph_model):
    #     student_graph_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_graph_model)
    #     teacher_graph_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_graph_model)

    #     # Wrap models with DDP
    #     student_graph_model = nn.parallel.DistributedDataParallel(student_graph_model, device_ids=[args.gpu])
    #     teacher_graph_model = nn.parallel.DistributedDataParallel(teacher_graph_model, device_ids=[args.gpu])
    #     teacher_graph_model_without_ddp = teacher_graph_model.module
    # else:
    #     student_graph_model = nn.parallel.DistributedDataParallel(student_graph_model, device_ids=[args.gpu])
    #     teacher_graph_model_without_ddp = teacher_graph_model

    # Initialize the teacher graph model with the student's parameters
    # teacher_graph_model_without_ddp.load_state_dict(student_graph_model.module.state_dict())

    # Freeze teacher graph model parameters
    # for p in teacher_graph_model.parameters():
    #     p.requires_grad = False

    # teacher_graph_model.eval()

    print("Student Graph Model and Teacher Graph Model are built and wrapped with DDP.")

    # Create student and teacher instances
    student_combined_model = CombinedModel(student, args) #, student_graph_model
    teacher_combined_model = CombinedModel(teacher, args) #, teacher_graph_model

    # Freeze teacher graph model parameters
    for p in teacher_combined_model.parameters():
        p.requires_grad = False

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataset size and feature dimensions
    num_samples = len(data_loader.dataset)
    
    if args.use_memmap:
        # Create memmap directory
        memmap_dir = os.path.join(args.output_dir, args.memmap_dir)
        Path(memmap_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Using memory mapping for large tensors in directory: {memmap_dir}")
        
        # Initialize memmap tensors for teacher
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
        
        # Initialize memmap tensors for student
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
    else:
        # Standard PyTorch tensor initialization (original code)
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
        
        # Set memory mapping variables to None
        teacher_features_memmap = None
        teacher_nn_tensor_memmap = None
        teacher_sim_tensor_memmap = None
        teacher_nn_matrix_memmap = None
        teacher_sim_matrix_memmap = None
        
        student_features_memmap = None
        student_nn_tensor_memmap = None
        student_sim_tensor_memmap = None
        student_nn_matrix_memmap = None
        student_sim_matrix_memmap = None

    # Initialize PyG Data objects for graphs
    # Number of nodes in the graph
    num_nodes = len(data_loader.dataset)
    feature_dim = embed_dim # or args.out_dim embed_dim
    max_edges_per_node = args.edges_per_node * args.vote_nn_nb
    max_edges = num_nodes * max_edges_per_node

    if args.use_memmap:
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
    else:
        # Standard PyG graph initialization (original code)
        teacher_graph = Data(num_nodes=num_nodes)
        student_graph = Data(num_nodes=num_nodes)
        
        teacher_graph.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        student_graph.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
        teacher_graph.x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32).to(device)
        student_graph.x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32).to(device)
        
        # Reserve space for edges: [2, num_nodes * max_edges_per_node]
        teacher_graph.reserved_edge_index = torch.zeros(2, num_nodes * max_edges_per_node, dtype=torch.long, device=device)
        student_graph.reserved_edge_index = torch.zeros(2, num_nodes * max_edges_per_node, dtype=torch.long, device=device)
        
        teacher_graph.edge_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)  # Track edge counts for each node
        student_graph.edge_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # Initialize PyG Data objects for TEST graphs
        # Number of nodes in the graph
        num_nodes_test = len(data_loader_test.dataset)
        
        teacher_graph_test = Data(num_nodes=num_nodes_test)
        student_graph_test = Data(num_nodes=num_nodes_test)
        
        teacher_graph_test.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        student_graph_test.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
        teacher_graph_test.x = torch.zeros((num_nodes_test, feature_dim), dtype=torch.float32).to(device)
        student_graph_test.x = torch.zeros((num_nodes_test, feature_dim), dtype=torch.float32).to(device)
        
        # Reserve space for edges: [2, num_nodes_test * max_edges_per_node]
        teacher_graph_test.reserved_edge_index = torch.zeros(2, num_nodes_test * max_edges_per_node, dtype=torch.long, device=device)
        student_graph_test.reserved_edge_index = torch.zeros(2, num_nodes_test * max_edges_per_node, dtype=torch.long, device=device)
        
        teacher_graph_test.edge_counts = torch.zeros(num_nodes_test, dtype=torch.long, device=device)
        student_graph_test.edge_counts = torch.zeros(num_nodes_test, dtype=torch.long, device=device)
        
        # Set memory mapping variables to None
        teacher_memmap_graph = None
        student_memmap_graph = None
        teacher_memmap_graph_test = None
        student_memmap_graph_test = None

    # Ensure the graphs are on the correct device
    teacher_graph = teacher_graph.to(device)
    teacher_graph = teacher_graph.to(device)
    student_graph = student_graph.to(device)

    teacher_graph_test = teacher_graph_test.to(device)
    student_graph_test = student_graph_test.to(device)

    # ============ preparing loss ... ============
    adasim_loss = AdaSimLoss(
        args.out_dim,
        embed_dim,
        2,  # total number of crops = 2 global crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs, args=args, writer=writer
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student_combined_model)#, graph_model
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    graph_momentum_schedule = utils.cosine_scheduler(args.momentum_teacher_graph, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============n
    to_restore = {"epoch": 0,
                  'teacher_nn_tensor_cpu': teacher_nn_tensor_cpu,
                  'teacher_sim_tensor_cpu': teacher_sim_tensor_cpu,
                  'teacher_features_cpu': teacher_features_cpu,
                  "teacher_nn_matrix_cpu": teacher_nn_matrix_cpu,
                  "teacher_sim_matrix_cpu": teacher_sim_matrix_cpu,
                  'student_nn_tensor_cpu': student_nn_tensor_cpu,
                  'student_sim_tensor_cpu': student_sim_tensor_cpu,
                  'student_features_cpu': student_features_cpu,
                  "student_nn_matrix_cpu": student_nn_matrix_cpu,
                  "student_sim_matrix_cpu": student_sim_matrix_cpu,
                  "teacher_graph": teacher_graph,
                  "student_graph": student_graph,
                  "teacher_graph_test": teacher_graph_test,
                  "student_graph_test": student_graph_test,
                  "optimizer": optimizer,
                #   "node_center": adasim_loss.node_center,
                #   "global_center": adasim_loss.global_center,
                #   "center": adasim_loss.center
                  }
    default_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if not os.path.isfile(default_checkpoint_path) and os.path.isfile(args.start_checkpoint_path):
        utils.master_copy_from_to(args.start_checkpoint_path, default_checkpoint_path)

    torch.distributed.barrier()

    try:
        # ============ preparing checkpointing: keep the 10 last checkpoints and restore if possible ============
        utils.restart_from_checkpoint(
            args.start_checkpoint_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            # student_graph_model=student_graph_model,
            # teacher_graph_model=teacher_graph_model,
            student_combined_model=student_combined_model,
            teacher_combined_model=teacher_combined_model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            adasim_loss=adasim_loss,
        )

        default_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
        to_restore = {"epoch": 0}
        if args.use_memmap:
            # Include memory-mapped tensors in the state to restore
            to_restore.update({
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
            })
        
        # Add standard CPU tensors to the state to restore
        to_restore.update({
            "teacher_features_cpu": teacher_features_cpu,
            "teacher_nn_tensor_cpu": teacher_nn_tensor_cpu,
            "teacher_sim_tensor_cpu": teacher_sim_tensor_cpu,
            "teacher_nn_matrix_cpu": teacher_nn_matrix_cpu,
            "teacher_sim_matrix_cpu": teacher_sim_matrix_cpu,
            "student_features_cpu": student_features_cpu,
            "student_nn_tensor_cpu": student_nn_tensor_cpu,
            "student_sim_tensor_cpu": student_sim_tensor_cpu,
            "student_nn_matrix_cpu": student_nn_matrix_cpu,
            "student_sim_matrix_cpu": student_sim_matrix_cpu,
            "teacher_graph": teacher_graph,
            "student_graph": student_graph,
            "teacher_graph_test": teacher_graph_test,
            "student_graph_test": student_graph_test,
        })

        utils.restart_from_checkpoint(
            default_checkpoint_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            # student_graph_model=student_graph_model,
            # teacher_graph_model=teacher_graph_model,
            student_combined_model=student_combined_model,
            teacher_combined_model=teacher_combined_model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            adasim_loss=adasim_loss,
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
            student_combined_model=student_combined_model,
            teacher_combined_model=teacher_combined_model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            adasim_loss=adasim_loss,
        )
    start_epoch = to_restore["epoch"]
    teacher_features_cpu = to_restore["teacher_features_cpu"]
    teacher_nn_tensor_cpu = to_restore["teacher_nn_tensor_cpu"]
    teacher_sim_tensor_cpu = to_restore["teacher_sim_tensor_cpu"]
    teacher_nn_matrix_cpu = to_restore["teacher_nn_matrix_cpu"]
    teacher_sim_matrix_cpu = to_restore["teacher_sim_matrix_cpu"]

    student_features_cpu = to_restore["student_features_cpu"]
    student_nn_tensor_cpu = to_restore["student_nn_tensor_cpu"]
    student_sim_tensor_cpu = to_restore["student_sim_tensor_cpu"]
    student_nn_matrix_cpu = to_restore["student_nn_matrix_cpu"]
    student_sim_matrix_cpu = to_restore["student_sim_matrix_cpu"]

    teacher_graph = to_restore["teacher_graph"]
    student_graph = to_restore["student_graph"]
 
    teacher_graph_test = to_restore["teacher_graph_test"]
    student_graph_test = to_restore["student_graph_test"]

    # optimizer=to_restore['optimizer']
    
    # Ensure the graphs are on the correct device
    teacher_graph = teacher_graph.to(device)
    student_graph = student_graph.to(device)

    teacher_graph_test = teacher_graph_test.to(device)
    student_graph_test = student_graph_test.to(device)

    teacher_nn_matrix_cpu = teacher_nn_matrix_cpu[:, -args.vote_nn_nb:]
    teacher_sim_matrix_cpu = teacher_sim_matrix_cpu[:, -args.vote_nn_nb:]
    teacher_nn_matrix_cpu_flag = True
    # if teacher_nn_matrix_cpu is not None:
    #     teacher_nn_matrix_cpu_flag = True
    # else:
    #     teacher_nn_matrix_cpu_flag = False

    student_nn_matrix_cpu = student_nn_matrix_cpu[:, -args.vote_nn_nb:]
    student_sim_matrix_cpu = student_sim_matrix_cpu[:, -args.vote_nn_nb:]

    teacher_features = teacher_features_cpu.cuda()
    teacher_nn_tensor = teacher_nn_tensor_cpu.cuda()
    teacher_sim_tensor = teacher_sim_tensor_cpu.cuda()

    student_features = student_features_cpu.cuda()
    student_nn_tensor = student_nn_tensor_cpu.cuda()
    student_sim_tensor = student_sim_tensor_cpu.cuda()

    start_time = time.time()
    print("Starting AdaSim training !")
    bootstrap_myself_tensor = torch.zeros(len(dataset), dtype=torch.long)
    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        data_loader.sampler.set_epoch(epoch)

        if epoch >= args.vote_nn_nb:
            data_loader.dataset.nn_matrix_cpu = teacher_nn_matrix_cpu
            data_loader.dataset.sim_matrix_cpu = teacher_sim_matrix_cpu
            
            # data_loader.dataset.student_nn_matrix_cpu = student_nn_matrix_cpu
            # data_loader.dataset.student_sim_matrix_cpu = student_sim_matrix_cpu
            
        try:
            # ============ training one epoch of DINO ... ============

            if args.use_memmap:
                train_stats = train_one_epoch(
                    student_combined_model, teacher_combined_model, student, teacher,
                    teacher_without_ddp, adasim_loss,
                    data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, graph_momentum_schedule,
                    epoch, fp16_scaler, teacher_features, teacher_nn_tensor, teacher_sim_tensor, bootstrap_myself_tensor,
                    teacher_graph, args, student_features, student_nn_tensor, student_sim_tensor, student_graph, teacher_nn_matrix_cpu_flag,
                    teacher_features_memmap, teacher_nn_tensor_memmap, teacher_sim_tensor_memmap,
                    student_features_memmap, student_nn_tensor_memmap, student_sim_tensor_memmap,
                    teacher_memmap_graph, student_memmap_graph
                )
            else:
                train_stats = train_one_epoch(
                    student_combined_model, teacher_combined_model, student, teacher,
                    teacher_without_ddp, adasim_loss,
                    data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, graph_momentum_schedule,
                    epoch, fp16_scaler, teacher_features, teacher_nn_tensor, teacher_sim_tensor, bootstrap_myself_tensor,
                    teacher_graph, args, student_features, student_nn_tensor, student_sim_tensor, student_graph, teacher_nn_matrix_cpu_flag
                )
            
            teacher_graph_test, student_graph_test = update_test_graphs(data_loader_test, epoch, len(dataset), student, teacher, teacher_features, student_features, teacher_graph_test, student_graph_test, args, teacher_memmap_graph_test, student_memmap_graph_test)
            
            teacher_nn_tensor_cpu = teacher_nn_tensor.cpu()
            teacher_sim_tensor_cpu = teacher_sim_tensor.cpu()
            teacher_features_cpu = teacher_features.cpu()

            student_nn_tensor_cpu = student_nn_tensor.cpu()
            student_sim_tensor_cpu = student_sim_tensor.cpu()
            student_features_cpu = student_features.cpu()

            if args.use_memmap:
                update_nn(teacher_nn_tensor_cpu, teacher_sim_tensor_cpu, teacher_nn_matrix_cpu, teacher_sim_matrix_cpu, 
                         teacher_nn_matrix_memmap, teacher_sim_matrix_memmap)
                update_nn(student_nn_tensor_cpu, student_sim_tensor_cpu, student_nn_matrix_cpu, student_sim_matrix_cpu,
                         student_nn_matrix_memmap, student_sim_matrix_memmap)
            else:
                update_nn(teacher_nn_tensor_cpu, teacher_sim_tensor_cpu, teacher_nn_matrix_cpu, teacher_sim_matrix_cpu)
                update_nn(student_nn_tensor_cpu, student_sim_tensor_cpu, student_nn_matrix_cpu, student_sim_matrix_cpu)

        except Exception as e:
            print(e, flush=True)
            with open(os.path.join(args.output_dir, 'error_{}'.format(args.rank)), 'w') as f:
                f.write(str(e))
            time.sleep(3)  # Should not be needed

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),            
            # 'teacher_graph_model': teacher_graph_model.state_dict(),
            # 'student_graph_model': student_graph_model.state_dict(),
            'teacher_combined_model': teacher_combined_model.state_dict(),
            'student_combined_model': student_combined_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'adasim_loss': adasim_loss.state_dict(),
        }
        
        # Add CPU tensors to save_dict
        save_dict.update({
            'teacher_nn_tensor_cpu': teacher_nn_tensor_cpu,
            'teacher_sim_tensor_cpu': teacher_sim_tensor_cpu,
            'teacher_features_cpu': teacher_features_cpu,
            'teacher_nn_matrix_cpu': teacher_nn_matrix_cpu,
            'teacher_sim_matrix_cpu': teacher_sim_matrix_cpu,
            'student_nn_tensor_cpu': student_nn_tensor_cpu,
            'student_sim_tensor_cpu': student_sim_tensor_cpu,
            'student_features_cpu': student_features_cpu,
            'student_nn_matrix_cpu': student_nn_matrix_cpu,
            'student_sim_matrix_cpu': student_sim_matrix_cpu,
            'teacher_graph': teacher_graph,
            'student_graph': student_graph,
            'teacher_graph_test': teacher_graph_test,
            'student_graph_test': student_graph_test
        })
        
        # If memory mapping is enabled, add metadata about memory-mapped tensors and graphs
        if args.use_memmap:
            memmap_info = {}
            
            # Add teacher tensor memory map info
            memmap_info['teacher_features_memmap'] = {
                'dir_path': teacher_features_memmap.dir_path,
                'filename': teacher_features_memmap.filename,
                'shape': teacher_features_memmap.shape,
                'dtype': str(teacher_features_memmap.dtype)
            }
            memmap_info['teacher_nn_tensor_memmap'] = {
                'dir_path': teacher_nn_tensor_memmap.dir_path,
                'filename': teacher_nn_tensor_memmap.filename,
                'shape': teacher_nn_tensor_memmap.shape,
                'dtype': str(teacher_nn_tensor_memmap.dtype)
            }
            memmap_info['teacher_sim_tensor_memmap'] = {
                'dir_path': teacher_sim_tensor_memmap.dir_path,
                'filename': teacher_sim_tensor_memmap.filename,
                'shape': teacher_sim_tensor_memmap.shape,
                'dtype': str(teacher_sim_tensor_memmap.dtype)
            }
            memmap_info['teacher_nn_matrix_memmap'] = {
                'dir_path': teacher_nn_matrix_memmap.dir_path,
                'filename': teacher_nn_matrix_memmap.filename,
                'shape': teacher_nn_matrix_memmap.shape,
                'dtype': str(teacher_nn_matrix_memmap.dtype)
            }
            memmap_info['teacher_sim_matrix_memmap'] = {
                'dir_path': teacher_sim_matrix_memmap.dir_path,
                'filename': teacher_sim_matrix_memmap.filename,
                'shape': teacher_sim_matrix_memmap.shape,
                'dtype': str(teacher_sim_matrix_memmap.dtype)
            }
            
            # Add student tensor memory map info
            memmap_info['student_features_memmap'] = {
                'dir_path': student_features_memmap.dir_path,
                'filename': student_features_memmap.filename,
                'shape': student_features_memmap.shape,
                'dtype': str(student_features_memmap.dtype)
            }
            memmap_info['student_nn_tensor_memmap'] = {
                'dir_path': student_nn_tensor_memmap.dir_path,
                'filename': student_nn_tensor_memmap.filename,
                'shape': student_nn_tensor_memmap.shape,
                'dtype': str(student_nn_tensor_memmap.dtype)
            }
            memmap_info['student_sim_tensor_memmap'] = {
                'dir_path': student_sim_tensor_memmap.dir_path,
                'filename': student_sim_tensor_memmap.filename,
                'shape': student_sim_tensor_memmap.shape,
                'dtype': str(student_sim_tensor_memmap.dtype)
            }
            memmap_info['student_nn_matrix_memmap'] = {
                'dir_path': student_nn_matrix_memmap.dir_path,
                'filename': student_nn_matrix_memmap.filename,
                'shape': student_nn_matrix_memmap.shape,
                'dtype': str(student_nn_matrix_memmap.dtype)
            }
            memmap_info['student_sim_matrix_memmap'] = {
                'dir_path': student_sim_matrix_memmap.dir_path,
                'filename': student_sim_matrix_memmap.filename,
                'shape': student_sim_matrix_memmap.shape,
                'dtype': str(student_sim_matrix_memmap.dtype)
            }
            
            # Add graph memory map info
            memmap_info['teacher_memmap_graph'] = {
                'dir_path': os.path.join(memmap_dir, 'graphs'),
                'prefix': 'teacher',
                'num_nodes': teacher_memmap_graph.num_nodes,
                'feature_dim': teacher_memmap_graph.feature_dim,
                'max_edges': teacher_memmap_graph.max_edges
            }
            memmap_info['student_memmap_graph'] = {
                'dir_path': os.path.join(memmap_dir, 'graphs'),
                'prefix': 'student',
                'num_nodes': student_memmap_graph.num_nodes,
                'feature_dim': student_memmap_graph.feature_dim,
                'max_edges': student_memmap_graph.max_edges
            }
            memmap_info['teacher_memmap_graph_test'] = {
                'dir_path': os.path.join(memmap_dir, 'graphs'),
                'prefix': 'teacher_test',
                'num_nodes': teacher_memmap_graph_test.num_nodes,
                'feature_dim': teacher_memmap_graph_test.feature_dim,
                'max_edges': teacher_memmap_graph_test.max_edges
            }
            memmap_info['student_memmap_graph_test'] = {
                'dir_path': os.path.join(memmap_dir, 'graphs'),
                'prefix': 'student_test',
                'num_nodes': student_memmap_graph_test.num_nodes,
                'feature_dim': student_memmap_graph_test.feature_dim,
                'max_edges': student_memmap_graph_test.max_edges
            }
            
            save_dict['memmap_info'] = memmap_info

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        if epoch == 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                         'epoch_time': time.time() - start_epoch_time}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                         'epoch_time': time.time() - start_epoch_time,
                         'nn_accuracy_top1': get_nn_acuracy(dataset, teacher_nn_tensor_cpu[:, 0]),
                         'nn_accuracy_top2': get_nn_acuracy(dataset, teacher_nn_tensor_cpu[:, 1]),
                         'nn_self_accuracy_top1': get_nn_self_accuracy(teacher_nn_tensor_cpu[:, 0]),
                         'teacher_overall_neighbors_accuracy': get_overall_neighbors_accuracy(1300, max_edges_per_node, teacher_graph.edge_index[1,:num_nodes*max_edges_per_node]),
                         'student_overall_neighbors_accuracy': get_overall_neighbors_accuracy(1300, max_edges_per_node, student_graph.edge_index[1,:num_nodes*max_edges_per_node]),
                        #  'nn_self_accuracy_top2': get_nn_self_accuracy(teacher_nn_tensor_cpu[:, 1]),
                         'nn_ratio_self': bootstrap_myself_tensor.sum().item() / len(bootstrap_myself_tensor)}
            writer.add_scalar('nn_accuracy_top1', log_stats['nn_accuracy_top1'], epoch)
            writer.add_scalar('nn_accuracy_top2', log_stats['nn_accuracy_top2'], epoch)
            writer.add_scalar('nn_self_accuracy_top1', log_stats['nn_self_accuracy_top1'], epoch)
            writer.add_scalar('teacher_overall_neighbors_accuracy', log_stats['teacher_overall_neighbors_accuracy'], epoch)
            writer.add_scalar('student_overall_neighbors_accuracy', log_stats['student_overall_neighbors_accuracy'], epoch)
            # writer.add_scalar('nn_self_accuracy_top2', log_stats['nn_self_accuracy_top2'], epoch)
            writer.add_scalar('nn_ratio_self', log_stats['nn_ratio_self'], epoch)
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # print(f"Number of edges in the teacher graph: {teacher_graph.edge_index.size(1)}")
    # print(f"Number of edges in the student graph: {student_graph.edge_index.size(1)}")


    # Example Parameters
    # number_of_samples_per_class = 50
    # max_edges_per_node = 10
    # total_samples = 2500

    # # Simulate nearest neighbor indices
    # nearest_neighbor = torch.randint(0, total_samples, (total_samples * max_edges_per_node,))
    # nn_accuracy(1300,max_edges_per_node, teacher_graph.edge_index[1,:])

    # Calculate NN Accuracy
    # accuracy = get_overall_neighbors_accuracy(1300, max_edges_per_node, teacher_graph.edge_index[1,:num_nodes*max_edges_per_node])


def get_overall_neighbors_accuracy(number_of_samples_per_class: int, max_edges_per_node: int, nearest_neighbor: torch.Tensor):
    """
    Calculates the nearest neighbor accuracy based on sample-class alignment and provided nearest neighbors.
    
    Args:
        number_of_samples_per_class (int): Number of samples in each class.
        max_edges_per_node (int): Maximum number of nearest neighbors per sample.
        nearest_neighbor (torch.Tensor): Precomputed nearest neighbors as a 1D tensor.
    
    Returns:
        float: Nearest neighbor accuracy.
    """
    # Total number of samples
    total_samples = nearest_neighbor.size(0) // max_edges_per_node
    
    # Validate nearest_neighbor dimensions
    if nearest_neighbor.numel() % max_edges_per_node != 0:
        raise ValueError("nearest_neighbor size must be divisible by max_edges_per_node.")
    
    # Create a tensor representing the samples
    samples = torch.arange(total_samples)
    
    # Class index for each sample
    sample_classes = samples // number_of_samples_per_class  # Class for each sample
    
    # Expand sample_classes to match nearest_neighbor structure
    sample_classes_expanded = sample_classes.repeat_interleave(max_edges_per_node)
    
    # Determine class index for each nearest neighbor
    nn_classes = nearest_neighbor // number_of_samples_per_class  # Class index for neighbors
    nn_classes = nn_classes.cpu()
    # Compare if nearest neighbor classes match the sample's class
    correct_neighbors = (nn_classes == sample_classes_expanded)
    
    # Reshape to [total_samples, max_edges_per_node] and count correct neighbors
    correct_neighbors = correct_neighbors.view(total_samples, max_edges_per_node)
    correct_counts = correct_neighbors.sum(dim=1)
    
    # Calculate accuracy
    nn_accuracy = correct_counts.float().mean() / max_edges_per_node
    
    print(f"Total Samples: {total_samples}")
    print(f"Correct Neighbors per Node (avg): {correct_counts.float().mean().item()}")
    print(f"Nearest Neighbor Accuracy: {nn_accuracy.item() * 100:.2f}%")
    
    return nn_accuracy.item()


def get_nn_acuracy(dataset, nn_tensor):
    t = torch.Tensor(dataset.targets)
    return sum(t[nn_tensor] == t).item() / len(nn_tensor)


def get_nn_self_accuracy(nn_tensor):
    return sum(nn_tensor == torch.Tensor(range(len(nn_tensor)))).item() / len(nn_tensor)


def update_state(feats_local, indices_local, sims_knn_local, indices_knn_local, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor, requires_grad, features_memmap=None, nn_tensor_memmap=None, sim_tensor_memmap=None):
    if not requires_grad:
        with torch.no_grad():
            _update_state_inner(feats_local, indices_local, sims_knn_local, indices_knn_local, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor)
            
            # Update memory-mapped arrays if provided
            if features_memmap is not None or nn_tensor_memmap is not None or sim_tensor_memmap is not None:
                # Convert to CPU for memory mapping
                features_cpu = features.cpu()
                nn_tensor_cpu = nn_tensor.cpu()
                sim_tensor_cpu = sim_tensor.cpu()
                
                # Update memory-mapped arrays
                if features_memmap is not None:
                    features_memmap.update_from_torch(features_cpu)
                if nn_tensor_memmap is not None:
                    nn_tensor_memmap.update_from_torch(nn_tensor_cpu)
                if sim_tensor_memmap is not None:
                    sim_tensor_memmap.update_from_torch(sim_tensor_cpu)
    else:
        _update_state_inner(feats_local, indices_local, sims_knn_local, indices_knn_local, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor)
        
        # Update memory-mapped arrays if provided
        if features_memmap is not None or nn_tensor_memmap is not None or sim_tensor_memmap is not None:
            # Convert to CPU for memory mapping
            features_cpu = features.cpu()
            nn_tensor_cpu = nn_tensor.cpu()
            sim_tensor_cpu = sim_tensor.cpu()
            
            # Update memory-mapped arrays
            if features_memmap is not None:
                features_memmap.update_from_torch(features_cpu)
            if nn_tensor_memmap is not None:
                nn_tensor_memmap.update_from_torch(nn_tensor_cpu)
            if sim_tensor_memmap is not None:
                sim_tensor_memmap.update_from_torch(sim_tensor_cpu)


def _update_state_inner(feats_local, indices_local, sims_knn_local, indices_knn_local, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor):
    # feats_local = F.normalize(teacher_output[1].reshape(2, -1, teacher_output[1].shape[-1]), p=2, dim=-1)
    # if args.nn_rep_type == "mean":
    #     feats_local = feats_local.mean(dim=0)
    #     feats_local = F.normalize(feats_local, p=2, dim=-1)
    # elif args.nn_rep_type == "first":
    #     feats_local = feats_local[0]
    # elif args.nn_rep_type == "second":
    #     feats_local = feats_local[1]
    # else:
    #     raise NotImplemented

    # Compute knn
    # similarity = feats_local @ features.T
    # sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

    indices_batch_all = gather_all_gpus(indices_local)
    features_batch_all = gather_all_gpus(feats_local)
    indices_knn_batch_all = gather_all_gpus(indices_knn_local)
    sims_knn_batch_all = gather_all_gpus(sims_knn_local)

    # Convert feats_local to the same type as features
    feats_local = feats_local.to(features.dtype)
    features_batch_all = features_batch_all.to(features.dtype)
    sims_knn_batch_all = sims_knn_batch_all.to(sim_tensor.dtype)

    features.index_copy_(0, indices_batch_all, features_batch_all)
    nn_tensor.index_copy_(0, indices_batch_all, indices_knn_batch_all)
    sim_tensor.index_copy_(0, indices_batch_all, sims_knn_batch_all)

    # Same as above but for cpu tensor
    bootstrap_myself_tensor[indices_batch_all.cpu()] = gather_all_gpus(same_im_bool).cpu()


def update_nn(nn_tensor_cpu, sim_tensor_cpu, nn_matrix_cpu, sim_matrix_cpu, nn_matrix_memmap=None, sim_matrix_memmap=None):
    with torch.no_grad():
        # Shift tensor by 1 index
        nn_matrix_cpu[:, :-1] = nn_matrix_cpu[:, 1:]
        nn_matrix_cpu[:, -1] = nn_tensor_cpu

        sim_matrix_cpu[:, :-1] = sim_matrix_cpu[:, 1:]
        sim_matrix_cpu[:, -1] = sim_tensor_cpu
        
        # Update memory-mapped arrays if provided
        if nn_matrix_memmap is not None:
            nn_matrix_memmap.update_from_torch(nn_matrix_cpu)
        if sim_matrix_memmap is not None:
            sim_matrix_memmap.update_from_torch(sim_matrix_cpu)


def gather_all_gpus(local_tensor):
    feats_all = torch.empty(dist.get_world_size(), *local_tensor.shape, dtype=local_tensor.dtype,
                            device=local_tensor.device)
    output_l = list(feats_all.unbind(0))
    output_all_reduce = torch.distributed.all_gather(output_l, local_tensor, async_op=True)
    output_all_reduce.wait()
    return torch.cat(output_l)


def gather_all_gpus_graph(tensor):
    """
    Gathers `tensor` (same shape on all ranks) along dim=0 and concatenates.
    If your local tensors differ in length per rank, you need a more advanced gather.
    """
    world_size = dist.get_world_size()
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)


def update_graph_from_model_multi_gpu(images, indices, model, features, graph, len_train, args, memmap_graph=None):
    """
    Multi-GPU version of 'update_graph_from_model' that ensures all GPUs
    have a consistent view of the updated graph via all-gather + index_copy_.
    
    Args:
        images (list): Batch of images
        indices (Tensor): Batch indices
        model (nn.Module): Model (teacher or student)
        features (Tensor): Precomputed features for similarity calculation
        graph (Data): Graph object to be updated
        len_train (int): Offset for indexing during edge updates
        args (Namespace): Configuration arguments
        memmap_graph (MemmapGraph, optional): Memory-mapped graph to update if provided
        
    Returns:
        graph (Data): Updated graph object
    """
    # === 1. Forward Pass ===
    output, _ = model(images)  # shape: [2, batch_size, feat_dim]
    feats = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
    if args.nn_rep_type == "mean":
        feats = feats.mean(dim=0)
        feats = F.normalize(feats, p=2, dim=-1)
    elif args.nn_rep_type == "first":
        feats = feats[0]
    elif args.nn_rep_type == "second":
        feats = feats[1]
    else:
        raise NotImplementedError("Invalid nn_rep_type for graph update.")

    # === 2. Compute kNN for Local Batch ===
    similarity = feats @ features.T  # [batch_size, num_nodes]
    sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)
    # We'll not necessarily gather sims_knn_local or indices_knn_local unless you need them globally.

    # === 3. Prepare Local Edge Updates ===
    batch_size = indices.size(0)
    world_size = dist.get_world_size()

    # Per-node range in reserved_edge_index
    start_indices = indices * args.edges_per_node * args.vote_nn_nb

    # Edge positions (wrapping around in a circular buffer)
    edge_positions = graph.edge_counts[indices].unsqueeze(1) \
                     + torch.arange(args.edges_per_node, device=indices.device).unsqueeze(0)
    edge_positions %= (args.edges_per_node * args.vote_nn_nb)
    # Flatten
    reserved_indices_local = (start_indices.unsqueeze(1) + edge_positions).flatten()

    # Build src/dst nodes
    src_nodes_local = indices.repeat_interleave(args.edges_per_node) + len_train
    dst_nodes_local = indices_knn_local[:, :args.edges_per_node].flatten()
    new_edges_local = torch.stack([src_nodes_local, dst_nodes_local], dim=0)
    # shape [2, batch_size * edges_per_node]

    # Next edge_counts
    edge_counts_local = (graph.edge_counts[indices] + args.edges_per_node) \
                        % (args.edges_per_node * args.vote_nn_nb)

    # === 4. Gather All Local Updates to Every GPU ===
    # We'll gather:
    #   - indices and feats for node feature updates
    #   - reserved_indices_local, new_edges_local for edges
    #   - edge_counts_local for edge_counts
    # Then each GPU performs the same index_copy_ so they end up identical.

    all_indices        = gather_all_gpus_graph(indices)               # shape [world_size * batch_size]
    all_feats          = gather_all_gpus_graph(feats)                 # shape [world_size * batch_size, feat_dim]
    all_resv_indices   = gather_all_gpus_graph(reserved_indices_local) # shape [world_size * batch_size * edges_per_node]
    all_new_edges      = gather_all_gpus_graph(new_edges_local)        # shape [2 * world_size * batch_size * edges_per_node]
    all_edge_counts    = gather_all_gpus_graph(edge_counts_local)      # shape [world_size * batch_size]

    all_new_edges = all_new_edges.view(world_size, 2, batch_size*args.edges_per_node)
    all_new_edges = all_new_edges.permute(1, 0, 2)
    all_new_edges = all_new_edges.reshape(2, -1)

    # === 5. Apply Updates to 'graph' Using index_copy_ ===

    # 5.1. Node features
    graph.x.index_copy_(0, all_indices, all_feats)

    # 5.2. Edge counts
    graph.edge_counts.index_copy_(0, all_indices, all_edge_counts)

    # 5.3. reserved_edge_index
    # index_copy_ along dim=1 for the 2 x capacity array
    graph.reserved_edge_index.index_copy_(1, all_resv_indices, all_new_edges)

    # 5.4. Update edge_index
    graph.edge_index = graph.reserved_edge_index.clone()
    
    # Update memory-mapped graph if provided
    if memmap_graph is not None:
        memmap_graph.update_from_pyg_data(graph)

    return graph


def update_graph_from_model(images, indices, model, features, graph, len_train, args):
    """
    Perform forward pass, compute kNN similarities, and update graph.
    
    Args:
        images (list): Batch of images.
        indices (Tensor): Batch indices.
        model (nn.Module): Model (teacher or student).
        features (Tensor): Precomputed features for similarity calculation.
        graph (Data): Graph object to be updated.
        len_train (int): Offset for indexing during edge updates.
        args (Namespace): Configuration arguments.

    Returns:
        graph (Data): Updated graph object.
    """
    # === Forward Pass ===
    output = model(images)
    feats = F.normalize(output.reshape(2, -1, output.shape[-1]), p=2, dim=-1)
    if args.nn_rep_type == "mean":
        feats = feats.mean(dim=0)
        feats = F.normalize(feats, p=2, dim=-1)
    elif args.nn_rep_type == "first":
        feats = feats[0]
    elif args.nn_rep_type == "second":
        feats = feats[1]
    else:
        raise NotImplementedError("Invalid nn_rep_type for graph update.")

    # === Compute kNN Graph Update ===
    similarity = feats @ features.T
    sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

    # === Update reserved edge storage (Vectorized) ===
    batch_size = indices.size(0)
    start_indices = indices * args.edges_per_node * args.vote_nn_nb
    end_indices = start_indices + args.edges_per_node * args.vote_nn_nb
    
    # Create edge positions
    edge_positions = graph.edge_counts[indices].unsqueeze(1) + torch.arange(
        args.edges_per_node, device=indices.device).unsqueeze(0)
    edge_positions %= args.edges_per_node * args.vote_nn_nb

    # Create src and dst nodes in a batched manner
    src_nodes = indices.repeat_interleave(args.edges_per_node) + len_train
    dst_nodes = indices_knn_local[:, :args.edges_per_node].flatten()

    # Update reserved_edge_index
    reserved_indices = start_indices.unsqueeze(1) + edge_positions
    reserved_indices = reserved_indices.flatten()

    graph.reserved_edge_index[:, reserved_indices] = torch.stack(
        [src_nodes, dst_nodes], dim=0
    )

    # Update edge counts
    graph.edge_counts[indices] = (
        graph.edge_counts[indices] + args.edges_per_node
    ) % (args.edges_per_node * args.vote_nn_nb)

    # Update edge_index
    graph.edge_index = graph.reserved_edge_index.clone()

    # Update node features (x)
    graph.x[indices] = feats

    return graph


def update_test_graphs(data_loader_test, epoch, len_train, student, teacher, teacher_features, student_features, teacher_graph_test, student_graph_test, args, teacher_memmap_graph_test=None, student_memmap_graph_test=None):
    """
    Updates teacher and student test graphs based on the similarity between test samples
    and precomputed teacher and student features.
    
    Args:
        data_loader_test: Data loader for test data
        epoch: Current epoch number
        len_train: Number of training samples
        student: Student model
        teacher: Teacher model
        teacher_features: Teacher features tensor
        student_features: Student features tensor
        teacher_graph_test: Teacher graph for test data
        student_graph_test: Student graph for test data
        args: Configuration arguments
        teacher_memmap_graph_test: Memory-mapped teacher graph (optional)
        student_memmap_graph_test: Memory-mapped student graph (optional)
        
    Returns:
        Tuple of updated (teacher_graph_test, student_graph_test)
    """
    student.eval()
    teacher.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    with torch.no_grad():          
        for it, (images, indices, same_im_bool) in enumerate(metric_logger.log_every(data_loader_test, 10, header)):
            images = [img.cuda(non_blocking=True) for img in images]
            indices = indices.cuda(non_blocking=True)

            # Update Teacher Graph
            teacher_graph_test = update_graph_from_model_multi_gpu(
                images=images,
                indices=indices,
                model=teacher,
                features=teacher_features,
                graph=teacher_graph_test,
                len_train=len_train,
                args=args,
                memmap_graph=teacher_memmap_graph_test
            )

            # Update Student Graph
            student_graph_test = update_graph_from_model_multi_gpu(
                images=images,
                indices=indices,
                model=student,
                features=student_features,
                graph=student_graph_test,
                len_train=len_train,
                args=args,
                memmap_graph=student_memmap_graph_test
            )
    return teacher_graph_test, student_graph_test


def train_one_epoch(student_combined_model, teacher_combined_model, student, teacher, teacher_without_ddp, adasim_loss, data_loader, 
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, graph_momentum_schedule, epoch, fp16_scaler,
                    teacher_features, teacher_nn_tensor, teacher_sim_tensor, bootstrap_myself_tensor, teacher_graph, args,
                    student_features, student_nn_tensor, student_sim_tensor, student_graph, teacher_nn_matrix_cpu_flag,
                    teacher_features_memmap=None, teacher_nn_tensor_memmap=None, teacher_sim_tensor_memmap=None,
                    student_features_memmap=None, student_nn_tensor_memmap=None, student_sim_tensor_memmap=None,
                    teacher_memmap_graph=None, student_memmap_graph=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # all_indices = set()
    for it, (images, indices, same_im_bool) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        indices = indices.cuda(non_blocking=True)

        # all_indices.update(gather_all_gpus(indices).tolist())

        same_im_bool = same_im_bool.cuda(non_blocking=True)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Forward pass through teacher model
            teacher_output, teacher_head_output, teacher_graph, teacher_sims_knn_local,\
                teacher_indices_knn_local, teacher_feats_local = teacher_combined_model(
                images, indices, same_im_bool, teacher_features, teacher_nn_tensor,
                teacher_sim_tensor, bootstrap_myself_tensor, teacher_graph,
                teacher_nn_matrix_cpu_flag, is_student=False, fp16_scaler=fp16_scaler
            )
            # , teacher_node_embeddings, teacher_global_embedding

            # Forward pass through student model
            student_output, student_head_output, student_graph, student_sims_knn_local, \
                student_indices_knn_local, student_feats_local = student_combined_model(
                images, indices, same_im_bool, student_features, student_nn_tensor,
                student_sim_tensor, bootstrap_myself_tensor, student_graph,
                teacher_nn_matrix_cpu_flag, is_student=True, fp16_scaler=fp16_scaler
            )
            #, student_node_embeddings, student_global_embedding

            # Compute loss
            loss = adasim_loss(
                student_head_output, teacher_head_output, epoch, it,
                # teacher_node_embeddings, teacher_global_embedding,
                # student_node_embeddings, student_global_embedding,
                indices  # Passing indices as indices_batch_all
            )

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            subprocess.run("scancel $SLURM_JOB_ID", shell=True, check=True, env=dict(os.environ))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student_combined_model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_combined_model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student_combined_model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_combined_model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        teacher_graph.x = teacher_graph.x.detach()
        teacher_graph.edge_index = teacher_graph.edge_index.detach()
        student_graph.x = student_graph.x.detach()
        student_graph.edge_index = student_graph.edge_index.detach()

        teacher_features = teacher_features.detach()
        student_features = student_features.detach()
        teacher_nn_tensor = teacher_nn_tensor.detach()
        student_nn_tensor = student_nn_tensor.detach()
        teacher_sim_tensor = teacher_sim_tensor.detach()
        student_sim_tensor = student_sim_tensor.detach()
        bootstrap_myself_tensor = bootstrap_myself_tensor.detach()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Update state with memory-mapped support
        update_state(teacher_feats_local, indices, teacher_sims_knn_local, teacher_indices_knn_local, teacher_features, teacher_nn_tensor, teacher_sim_tensor, same_im_bool, bootstrap_myself_tensor,
                     requires_grad=False, features_memmap=teacher_features_memmap, nn_tensor_memmap=teacher_nn_tensor_memmap, sim_tensor_memmap=teacher_sim_tensor_memmap)
        update_state(student_feats_local, indices, student_sims_knn_local, student_indices_knn_local, student_features, student_nn_tensor, student_sim_tensor, same_im_bool, bootstrap_myself_tensor,
                     requires_grad=True, features_memmap=student_features_memmap, nn_tensor_memmap=student_nn_tensor_memmap, sim_tensor_memmap=student_sim_tensor_memmap)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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