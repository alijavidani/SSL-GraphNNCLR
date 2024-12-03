# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms

# # Writer will output to ./runs/ directory by default
# writer = SummaryWriter()

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# model = torchvision.models.resnet50(False)
# # Have ResNet model take in grayscale rather than RGB
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# images, labels = next(iter(trainloader))

# grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
# writer.close()
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"

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
# import caffe2


# Initialize the graph models
# Note:
embed_dim = 384
input_dim = embed_dim # Input feature dimension per node
hidden_dims = [512, 1024, 512, embed_dim]  # Hidden layer dimensions
# output_dim = hidden_dims[-1]       # Output dimension (graph-level embedding)
num_layers = len(hidden_dims)        # Number of GNN layers

# Create DINOHead instance
student_dino_head = DINOHead(
    in_dim = hidden_dims[-1],
    out_dim = 10000,
    use_bn = True,
    norm_last_layer = True,
    # nlayers = 3,
    # hidden_dim = 2048,
    # bottleneck_dim = 256
)

teacher_dino_head = DINOHead(
    in_dim = hidden_dims[-1],
    out_dim = 10000,
    use_bn = True,
    norm_last_layer =True,
    # nlayers = 3,
    # hidden_dim = 2048,
    # bottleneck_dim = 256
)
# input_dim = args.out_dim       # Input feature dimension per node
# hidden_dims = [50000, 30000, 20000]  # Hidden layer dimensions
# output_dim = 20000       # Output dimension (graph-level embedding)
# num_layers = 3        # Number of GNN layers

# Student graph model
student_graph_model = GraphNetWithDINO(input_dim, hidden_dims, num_layers, student_dino_head).cuda()

# Teacher graph model
teacher_graph_model = GraphNetWithDINO(input_dim, hidden_dims, num_layers, teacher_dino_head).cuda()

# student = utils.MultiCropWrapper(student, DINOHead(
#     embed_dim,
#     args.out_dim,
#     use_bn=args.use_bn_in_head,
#     norm_last_layer=args.norm_last_layer,
# ))

# Synchronize batch norms (if any)
# if utils.has_batchnorms(student_graph_model):
#     student_graph_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_graph_model)
#     teacher_graph_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_graph_model)

#     # Wrap models with DDP
#     student_graph_model = nn.parallel.DistributedDataParallel(student_graph_model, device_ids=[0])
#     teacher_graph_model = nn.parallel.DistributedDataParallel(teacher_graph_model, device_ids=[0])
#     teacher_graph_model_without_ddp = teacher_graph_model.module
# else:
#     student_graph_model = nn.parallel.DistributedDataParallel(student_graph_model, device_ids=[0])
#     teacher_graph_model_without_ddp = teacher_graph_model

# # Initialize the teacher graph model with the student's parameters
# teacher_graph_model_without_ddp.load_state_dict(student_graph_model.module.state_dict())

# Freeze teacher graph model parameters
# for p in teacher_graph_model.parameters():
#     p.requires_grad = False

# teacher_graph_model.eval()

print("Student Graph Model and Teacher Graph Model are built and wrapped with DDP.")

# Create student and teacher instances
# student_combined_model = CombinedModel(student, student_graph_model, args)
# teacher_combined_model = CombinedModel(teacher, teacher_graph_model, args)

# teacher_features_cpu = torch.zeros(len(data_loader.dataset), embed_dim, dtype=torch.float)
# teacher_nn_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.long)
# teacher_sim_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.float)
# teacher_nn_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.long)
# teacher_sim_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.float)

# student_features_cpu = torch.zeros(len(data_loader.dataset), embed_dim, dtype=torch.float)
# student_nn_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.long)
# student_sim_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.float)
# student_nn_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.long)
# student_sim_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.float)

# Initialize PyG Data objects for graphs
# Number of nodes in the graph
num_nodes = 50000 #len(data_loader.dataset)
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

# Define the number of nodes and node features
num_nodes = 10  # Example number of nodes
num_node_features = input_dim  # Should match input_dim from your model

# Create random node features
x = torch.randn((num_nodes, num_node_features))

# Define edge indices (simple chain graph for illustration)
edge_index = torch.tensor([
    [i for i in range(num_nodes - 1)],
    [i + 1 for i in range(num_nodes - 1)]
], dtype=torch.long)

# Create the Data object
dummy_data = Data(x=x, edge_index=edge_index)
writer.add_graph(student_graph_model, dummy_data)
writer.close()
