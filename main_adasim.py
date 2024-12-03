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
os.environ["CUDA_VISIBLE_DEVICES"] ="1,2,3"

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

    # multi-crop wrapper handles forward with inputs of different resolutions
    # student = utils.MultiCropWrapper(student, DINOHead(
    #     embed_dim,
    #     args.out_dim,
    #     use_bn=args.use_bn_in_head,
    #     norm_last_layer=args.norm_last_layer,
    # ))
    # teacher = utils.MultiCropWrapper(
    #     teacher,
    #     DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    # )
    student = utils.MultiCropWrapper(student)
    teacher = utils.MultiCropWrapper(teacher)
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

    # Student graph model
    student_graph_model = GraphNetWithDINO(input_dim, hidden_dims, num_layers, student_dino_head).cuda()

    # Teacher graph model
    teacher_graph_model = GraphNetWithDINO(input_dim, hidden_dims, num_layers, teacher_dino_head).cuda()

    # Synchronize batch norms (if any)
    if utils.has_batchnorms(student_graph_model):
        student_graph_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_graph_model)
        teacher_graph_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_graph_model)

        # Wrap models with DDP
        student_graph_model = nn.parallel.DistributedDataParallel(student_graph_model, device_ids=[args.gpu])
        teacher_graph_model = nn.parallel.DistributedDataParallel(teacher_graph_model, device_ids=[args.gpu])
        teacher_graph_model_without_ddp = teacher_graph_model.module
    else:
        student_graph_model = nn.parallel.DistributedDataParallel(student_graph_model, device_ids=[args.gpu])
        teacher_graph_model_without_ddp = teacher_graph_model

    # Initialize the teacher graph model with the student's parameters
    teacher_graph_model_without_ddp.load_state_dict(student_graph_model.module.state_dict())

    # Freeze teacher graph model parameters
    for p in teacher_graph_model.parameters():
        p.requires_grad = False

    # teacher_graph_model.eval()

    print("Student Graph Model and Teacher Graph Model are built and wrapped with DDP.")

    # Create student and teacher instances
    student_combined_model = CombinedModel(student, student_graph_model, args)
    teacher_combined_model = CombinedModel(teacher, teacher_graph_model, args)

    # Freeze teacher graph model parameters
    for p in teacher_combined_model.parameters():
        p.requires_grad = False

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

    # ============ preparing loss ... ============
    adasim_loss = AdaSimLoss(
        args.out_dim,
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
        utils.restart_from_checkpoint(
            default_checkpoint_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            student_graph_model=student_graph_model,
            teacher_graph_model=teacher_graph_model,
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
            student_graph_model=student_graph_model,
            teacher_graph_model=teacher_graph_model,
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
 
    # optimizer=to_restore['optimizer']
    
    # Ensure the graphs are on the correct device
    teacher_graph = teacher_graph.to(device)
    student_graph = student_graph.to(device)

    teacher_nn_matrix_cpu = teacher_nn_matrix_cpu[:, -args.vote_nn_nb:]
    teacher_sim_matrix_cpu = teacher_sim_matrix_cpu[:, -args.vote_nn_nb:]
    if teacher_nn_matrix_cpu is not None:
        teacher_nn_matrix_cpu_flag = True
    else:
        teacher_nn_matrix_cpu_flag = False

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

            train_stats = train_one_epoch(student_combined_model, teacher_combined_model, student, teacher, teacher_without_ddp, student_graph_model, teacher_graph_model, teacher_graph_model_without_ddp, adasim_loss,
                                          data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, graph_momentum_schedule,
                                          epoch, fp16_scaler, teacher_features, teacher_nn_tensor, teacher_sim_tensor, bootstrap_myself_tensor,
                                          teacher_graph, args, student_features, student_nn_tensor, student_sim_tensor, student_graph, teacher_nn_matrix_cpu_flag)
            teacher_nn_tensor_cpu = teacher_nn_tensor.cpu()
            teacher_sim_tensor_cpu = teacher_sim_tensor.cpu()
            teacher_features_cpu = teacher_features.cpu()

            student_nn_tensor_cpu = student_nn_tensor.cpu()
            student_sim_tensor_cpu = student_sim_tensor.cpu()
            student_features_cpu = student_features.cpu()

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
            'teacher_graph_model': teacher_graph_model.state_dict(),
            'student_graph_model': student_graph_model.state_dict(),
            'teacher_combined_model': teacher_combined_model.state_dict(),
            'student_combined_model': student_combined_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'adasim_loss': adasim_loss.state_dict(),
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
            'student_graph': student_graph
        }

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
                         'nn_self_accuracy_top2': get_nn_self_accuracy(teacher_nn_tensor_cpu[:, 1]),
                         'nn_ratio_self': bootstrap_myself_tensor.sum().item() / len(bootstrap_myself_tensor)}
            writer.add_scalar('nn_accuracy_top1', log_stats['nn_accuracy_top1'], epoch)
            writer.add_scalar('nn_accuracy_top2', log_stats['nn_accuracy_top2'], epoch)
            writer.add_scalar('nn_self_accuracy_top1', log_stats['nn_self_accuracy_top1'], epoch)
            writer.add_scalar('nn_self_accuracy_top2', log_stats['nn_self_accuracy_top2'], epoch)
            writer.add_scalar('nn_ratio_self', log_stats['nn_ratio_self'], epoch)
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # print(f"Number of edges in the teacher graph: {teacher_graph.edge_index.size(1)}")
    # print(f"Number of edges in the student graph: {student_graph.edge_index.size(1)}")

def get_nn_acuracy(dataset, nn_tensor):
    t = torch.Tensor(dataset.targets)
    return sum(t[nn_tensor] == t).item() / len(nn_tensor)


def get_nn_self_accuracy(nn_tensor):
    return sum(nn_tensor == torch.Tensor(range(len(nn_tensor)))).item() / len(nn_tensor)


def update_state(feats_local, indices_local, sims_knn_local, indices_knn_local,features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor, requires_grad):
    if not requires_grad:
        with torch.no_grad():
            _update_state_inner(feats_local, indices_local, sims_knn_local, indices_knn_local,features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor)
    else:
        _update_state_inner(feats_local, indices_local, sims_knn_local, indices_knn_local,features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor)

def _update_state_inner(feats_local, indices_local, sims_knn_local, indices_knn_local,features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor):
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


def update_nn(nn_tensor_cpu, sim_tensor_cpu, nn_matrix_cpu, sim_matrix_cpu):
    with torch.no_grad():
        # Shift tensor by 1 index
        nn_matrix_cpu[:, :-1] = nn_matrix_cpu[:, 1:]
        nn_matrix_cpu[:, -1] = nn_tensor_cpu

        sim_matrix_cpu[:, :-1] = sim_matrix_cpu[:, 1:]
        sim_matrix_cpu[:, -1] = sim_tensor_cpu


def gather_all_gpus(local_tensor):
    feats_all = torch.empty(dist.get_world_size(), *local_tensor.shape, dtype=local_tensor.dtype,
                            device=local_tensor.device)
    output_l = list(feats_all.unbind(0))
    output_all_reduce = torch.distributed.all_gather(output_l, local_tensor, async_op=True)
    output_all_reduce.wait()
    return torch.cat(output_l)


def train_one_epoch(student_combined_model, teacher_combined_model, student, teacher, teacher_without_ddp, student_graph_model, teacher_graph_model, teacher_graph_model_without_ddp, adasim_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, graph_momentum_schedule, epoch, fp16_scaler,
                    teacher_features, teacher_nn_tensor, teacher_sim_tensor, bootstrap_myself_tensor, teacher_graph, args,
                    student_features, student_nn_tensor, student_sim_tensor, student_graph, teacher_nn_matrix_cpu_flag):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

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
        same_im_bool = same_im_bool.cuda(non_blocking=True)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Forward pass through teacher model
            teacher_output, teacher_node_embeddings, teacher_global_embedding, teacher_graph, teacher_sims_knn_local, teacher_indices_knn_local, teacher_feats_local = teacher_combined_model(
                images, indices, same_im_bool, teacher_features, teacher_nn_tensor,
                teacher_sim_tensor, bootstrap_myself_tensor, teacher_graph,
                teacher_nn_matrix_cpu_flag, is_student=False, fp16_scaler=fp16_scaler
            )

            # Forward pass through student model
            student_output, student_node_embeddings, student_global_embedding, student_graph, student_sims_knn_local, student_indices_knn_local, student_feats_local = student_combined_model(
                images, indices, same_im_bool, student_features, student_nn_tensor,
                student_sim_tensor, bootstrap_myself_tensor, student_graph,
                teacher_nn_matrix_cpu_flag, is_student=True, fp16_scaler=fp16_scaler
            )

            writer.add_scalar('teacher_graph_edges', teacher_graph.edge_index.size(1), it)
            writer.add_scalar('student_graph_edges', student_graph.edge_index.size(1), it)

            print(f"Number of edges in the teacher graph: {teacher_graph.edge_index.size(1)}")
            print(f"Number of edges in the student graph: {student_graph.edge_index.size(1)}")

            # Compute loss
            loss = adasim_loss(
                student_output, teacher_output, epoch, it,
                teacher_node_embeddings, teacher_global_embedding,
                student_node_embeddings, student_global_embedding,
                indices  # Passing indices as indices_batch_all
            )
        
            # dot = make_dot(loss, params=dict(student_combined_model.named_parameters()))
            # dot.render("computation_graph_gcloss", format="png")
            # exit()
            # break

            # Register hooks
            # student_output.register_hook(grad_hook('student_output'))
            # student_graph.x.register_hook(grad_hook('student_graph.x'))
            # student_node_embeddings.register_hook(grad_hook('student_node_embeddings'))

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

        # for name, param in student_combined_model.named_parameters():
        #     if param.grad is not None:
        #         num_elements = param.grad.numel()
        #         print(f"Parameter '{name}' gradient has {num_elements} elements.")
        #         if num_elements > 0:
        #             if torch.isnan(param.grad).any():
        #                 print(f"Parameter '{name}' gradient contains NaN values, skipping histogram.")
        #             elif torch.isinf(param.grad).any():
        #                 print(f"Parameter '{name}' gradient contains Inf values, skipping histogram.")
        #             elif torch.all(param.grad == 0):
        #                 print(f"Parameter '{name}' gradient is all zeros, skipping histogram.")
        #             else:
        #                 # Check if gradient is sparse
        #                 if param.grad.is_sparse:
        #                     grad = param.grad.to_dense()
        #                 else:
        #                     grad = param.grad
        #                 writer.add_histogram(f"{name}_grad", grad, it)
        #         else:
        #             print(f"Skipping histogram for '{name}' because it has an empty gradient.")

        # for name, param in student_combined_model.named_parameters():
        #     if param.grad is not None:
        #         print(f'{name}: {param.grad.norm()}')
        #     else:
        #         print(f'{name} has no gradient')

        # for name, param in teacher_combined_model.named_parameters():
        #     if param.grad is not None:
        #         print(f'{name}: {param.grad.norm()}')
        #     else:
        #         print(f'{name} has no gradient')

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # EMA update for the teacher graph model
            m_graph = graph_momentum_schedule[it]  # momentum parameter for graph networks
            for param_q, param_k in zip(student_graph_model.module.parameters(), teacher_graph_model_without_ddp.parameters()):
                param_k.data.mul_(m_graph).add_((1 - m_graph) * param_q.detach().data)
                

        update_state(teacher_feats_local, indices, teacher_sims_knn_local, teacher_indices_knn_local, teacher_features, teacher_nn_tensor, teacher_sim_tensor, same_im_bool, bootstrap_myself_tensor,
                     requires_grad=False)
        update_state(student_feats_local, indices, student_sims_knn_local, student_indices_knn_local, student_features, student_nn_tensor, student_sim_tensor, same_im_bool, bootstrap_myself_tensor,
                     requires_grad=True)
        
        # After loss.backward()
        # for name, param in student_combined_model.named_parameters():
        #     if param.grad is not None:
        #         writer.add_histogram(f"{name}_grad", param.grad, it)


        # Visualize the computation graph
        # dot = make_dot(loss, params=dict(student_combined_model.named_parameters()))
        # dot.render("computation_graph", format="png")
        # exit()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def grad_hook(name):
    def hook(grad):
        print(f'Gradient at {name}: {grad.norm()}')
    return hook

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
