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

index_label_int = list(map(int, index_label_image))

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

# Perform Community Detection on the graph:
# communities = graph.community_infomap()
# print(communities)
# print(len(communities))

from sklearn.cluster import SpectralClustering
# import numpy as np
# import igraph as ig

# Assume 'graph' is your igraph Graph object

# # Convert the igraph graph to an adjacency matrix
# adjacency_matrix = np.array(graph.get_adjacency().data)

# # Specify the number of communities you wish to detect
# n_communities = 10

# # Initialize and fit the Spectral Clustering model
# sc = SpectralClustering(n_clusters=n_communities, affinity='precomputed', n_init=10, assign_labels='discretize')
# labels = sc.fit_predict(adjacency_matrix)

import networkx
import community as community_louvain
A = graph.get_edgelist()
G = networkx.DiGraph(A) # In case your graph is directed
G_undirected = G.to_undirected()
partition = community_louvain.best_partition(G_undirected)
# print(partition)

num_nodes = graph.vcount()
partition_list = [-1] * num_nodes  # Initialize with -1 or any other placeholder

# Populate the list with partition IDs
for node_id, partition_id in partition.items():
    partition_list[node_id] = partition_id

# from sklearn.metrics import adjusted_rand_score
# ari_score = adjusted_rand_score(index_label_int, partition_list)

# from sklearn.metrics import normalized_mutual_info_score
# nmi_score = normalized_mutual_info_score(index_label_int, partition_list)

# from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
# homogeneity = homogeneity_score(index_label_int, partition_list)
# completeness = completeness_score(index_label_int, partition_list)
# v_measure = v_measure_score(index_label_int, partition_list)

# #print computed metrics:
# print(f'ARI: {ari_score:.4f}')
# print(f'NMI: {nmi_score:.4f}')
# print(f'Homogeneity: {homogeneity:.4f}')
# print(f'Completeness: {completeness:.4f}')
# print(f'V-measure: {v_measure:.4f}')

# import igraph as ig
# import leidenalg as la

# # Assuming 'G' is an igraph Graph
# partition = la.find_partition(graph, la.ModularityVertexPartition)
# print(partition)
# print('leiden finished')

# import igraph as ig

# # G = ig.Graph.Erdos_Renyi(n=1000, m=5000)
# dendrogram = graph.to_undirected().community_fastgreedy()
# clusters = dendrogram.as_clustering()
# print(clusters)
# print('fastgreedy finished')

import infomap

im = infomap.Infomap("--two-level")

# Assuming 'G' is a NetworkX graph
for edge in G_undirected.edges():
    im.addLink(*edge)

im.run()

print("Found {} modules.".format(im.numTopModules()))

# Getting the community for each node
communities = {node: module for node, module in im.modules}
print(communities)
print('infomap finished')

# import networkx as nx
# from networkx.algorithms.community import girvan_newman
# # Assuming 'G' is your NetworkX graph
# communities_generator = girvan_newman(G_undirected)

# # Desired number of clusters
# k = 10

# # Initialize
# limited_communities = None

# for communities in communities_generator:
#     limited_communities = communities  # Update the communities at each iteration
#     if len(communities) == k:
#         break

# print(limited_communities)

# 'limited_communities' now holds the clusters when the first 'k' clusters are formed
# import matplotlib.pyplot as plt

# # Assuming 'G' is your graph and 'limited_communities' contains your communities
# pos = nx.spring_layout(G)  # positions for all nodes

# # Color the nodes according to their community
# for i, comm in enumerate(limited_communities):
#     list_nodes = list(comm)
#     nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
#                            node_color=str(i / len(limited_communities)))

# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()
