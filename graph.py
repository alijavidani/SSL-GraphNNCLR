import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import utils
import torch

# # Create the graph
# g = ig.Graph(n=5, directed=True)
# g.add_edge(source=0, target=4)
# g.add_edge(source=0, target=4)
# g.add_edge(source=0, target=4)

# g.add_edges([(1,2),(1,3),(2,4),(1,2)])

# adj_matrix = g.get_adjacency()

# # Define the vertex labels as their indices
# vertex_labels = [str(index) for index in range(g.vcount())]

# # Save the igraph plot to a temporary file with vertex labels
# temp_file = 'temp_graph_with_labels.png'
# ig.plot(g, target=temp_file, vertex_label=vertex_labels)

# # Read the temporary file into matplotlib
# img = mpimg.imread(temp_file)
# plt.imshow(img)
# plt.axis('off')  # Hide the axis
# plt.show()

from adasim_utils.parser import get_args_parser
import argparse

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
default_checkpoint_path = os.path.join('./output2', "checkpoint.pth")
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

# start_epoch = to_restore["epoch"]
# features_cpu = to_restore["features_cpu"]
# nn_tensor_cpu = to_restore["nn_tensor_cpu"]
# sim_tensor_cpu = to_restore["sim_tensor_cpu"]
# nn_matrix_cpu = to_restore["nn_matrix_cpu"]
# sim_matrix_cpu = to_restore["sim_matrix_cpu"]
# graph = to_restore["graph"]
# nn_matrix_cpu = nn_matrix_cpu[:, -args.vote_nn_nb:]
# sim_matrix_cpu = sim_matrix_cpu[:, -args.vote_nn_nb:]