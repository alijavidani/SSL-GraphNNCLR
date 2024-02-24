import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import utils
import torch

# Create the graph
g = ig.Graph(n=5, directed=True)
g.add_edge(source=0, target=4)
g.add_edge(source=0, target=4)
g.add_edge(source=0, target=4)

g.add_edges([(1,2),(1,3),(2,4),(1,2)])

adj_matrix = g.get_adjacency()

# Define the vertex labels as their indices
vertex_labels = [str(index) for index in range(g.vcount())]

# Save the igraph plot to a temporary file with vertex labels
temp_file = 'temp_graph_with_labels.png'
ig.plot(g, target=temp_file, vertex_label=vertex_labels)

# Read the temporary file into matplotlib
img = mpimg.imread(temp_file)
plt.imshow(img)
plt.axis('off')  # Hide the axis
plt.show()