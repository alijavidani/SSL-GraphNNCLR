import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import utils
import torch
import numpy as np
from torch_geometric_example import *

features_cpu = torch.zeros(100, 192, dtype=torch.float)
nn_tensor_cpu = torch.zeros(100, 10, dtype=torch.long)
sim_tensor_cpu = torch.zeros(100, 10, dtype=torch.float)
nn_matrix_cpu = torch.zeros(100, 50, 10, dtype=torch.long)
sim_matrix_cpu = torch.zeros(100, 50, 10, dtype=torch.float)
graph = ig.Graph(n=100, directed=True)
to_restore = {"epoch": 0,
                  'nn_tensor_cpu': nn_tensor_cpu,
                  'sim_tensor_cpu': sim_tensor_cpu,
                  'features_cpu': features_cpu,
                  "nn_matrix_cpu": nn_matrix_cpu,
                  "sim_matrix_cpu": sim_matrix_cpu,
                  "graph":graph}

# Insert the checkpoint path to restore the saved graph:
default_checkpoint_path = os.path.join('./dino_cifar10_3', "checkpoint.pth")
utils.restart_from_checkpoint(
            default_checkpoint_path,
            run_variables=to_restore,
            # student=student,
            # teacher=teacher,
            # optimizer=optimizer,
            # fp16_scaler=fp16_scaler,
            # adasim_loss=adasim_loss,
            # graph = graph
        )

start_epoch = to_restore["epoch"]
features_cpu = to_restore["features_cpu"]
# nn_tensor_cpu = to_restore["nn_tensor_cpu"]
# sim_tensor_cpu = to_restore["sim_tensor_cpu"]
# nn_matrix_cpu = to_restore["nn_matrix_cpu"]
# sim_matrix_cpu = to_restore["sim_matrix_cpu"]
# nn_matrix_cpu = nn_matrix_cpu[:, -args.vote_nn_nb:]
# sim_matrix_cpu = sim_matrix_cpu[:, -args.vote_nn_nb:]

graph = to_restore["graph"]
# adj = graph.get_adjacency()
# print(sum(sum(sublist) for sublist in adj.data))
print(graph.vcount())
print(graph.ecount())

filename = 'index_label_image_Cifar10.txt'
# The file exists, read its contents
with open(filename, 'r') as file:
    index_label_image = [line.strip() for line in file]

# Function to create a custom layout based on labels
def custom_layout(labels):
    # Map each unique label to an integer
    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
    # Group vertices by label
    groups = {}
    for idx, label in enumerate(labels):
        label_int = label_to_int[label]
        if label_int in groups:
            groups[label_int].append(idx)
        else:
            groups[label_int] = [idx]
    
    # Generate positions
    layout = [(0, 0)] * len(labels)
    for label_int, indices in groups.items():
        # Place vertices in a horizontal line
        x_positions = np.linspace(-len(indices), len(indices), num=len(indices))
        for i, idx in enumerate(indices):
            layout[idx] = (x_positions[i], label_int)
    
    return layout

# Compute the layout
layout = custom_layout(index_label_image)

# Plot the graph
# ig.plot(graph, target='an.png', layout=layout, vertex_label=index_label_image, bbox=(300, 300), margin=20)

# groups = 
# layout = ig.create_custom_layout(graph, groups)

# print(graph.community_infomap())
# print(graph.community_walktrap())


# Start with a force-directed layout
layout = graph.layout('fruchterman_reingold')

# Function to adjust layout based on string labels for clustering
def adjust_layout_for_clustering(layout, labels):
    # Group vertices by label
    label_positions = {}
    for idx, label in enumerate(labels):
        if label in label_positions:
            label_positions[label].append(layout[idx])
        else:
            label_positions[label] = [layout[idx]]

    # Adjust positions within each group to bring them closer
    for label, positions in label_positions.items():
        if len(positions) > 1:
            center_x, center_y = np.mean(positions, axis=0)
            for pos in positions:
                pos[0] = center_x + np.random.uniform(-0.05, 0.05)  # Small random adjustment
                pos[1] = center_y + np.random.uniform(-0.05, 0.05)

    return layout

# Adjust the layout
adjusted_layout = adjust_layout_for_clustering(layout, index_label_image)

# Plot the graph
# ig.plot(graph, target='an.png', layout=adjusted_layout, vertex_label=index_label_image, bbox=(300, 300), margin=20)

# Get edge list and convert to tensor
edge_list = torch.tensor([edge.tuple for edge in graph.es], dtype=torch.long).t().contiguous()

# Optionally, get node features if your graph has them
# For this example, we'll create a dummy feature for each node
node_features = torch.randn((graph.vcount(), 10))  # Example features

from torch_geometric.data import Data

num_nodes = graph.vcount()  # Assuming 100 nodes in your graph

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
num_train = int(num_nodes * 0.6)  # 60% for training
num_val = int(num_nodes * 0.4)    # 20% for validation
# num_test = num_nodes - num_train - num_val  # Remaining for testing

indices = torch.randperm(num_nodes)
train_mask[indices[:num_train]] = True
val_mask[indices[num_train:num_train+num_val]] = True
y=torch.tensor([int(s) for s in index_label_image])
# Create Data object for torch_geometric
data = Data(x=features_cpu, edge_index=edge_list, train_mask=train_mask, val_mask=val_mask, y=y)

gcn = GCN(data.num_features, 16, 10)
print(gcn)

# Train and test
# train(gcn, data)
# acc = test(gcn, data)
# print(f'\nGCN test accuracy: {acc*100:.2f}%\n')