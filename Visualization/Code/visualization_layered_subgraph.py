import torch
import networkx as nx
import matplotlib.pyplot as plt

# 1) Load the data from your .pth file
pth_file = "/home/alij/SSL-GraphNNCLR/saved_graphs/ImageNet100/ImageNet100_graphs_epn15.pth"
graphs_dict = torch.load(pth_file)
student_graph = graphs_dict["student_graph"]
teacher_graph = graphs_dict["teacher_graph"]

# 2) Choose the node you want to visualize
node_index = 42  # example node index

def build_subgraph_3hop_and_shells(edge_index, center_node):
    """
    1) Builds a directed DiGraph from edge_index.
    2) Finds all nodes within distance <= 3 of `center_node`.
    3) Returns (subG, shells), where `subG` is the 3-hop subgraph and
       `shells` is a list-of-lists for shell_layout:
         layer0 = [center_node]
         layer1 = nodes at distance=1
         layer2 = distance=2
         layer3 = distance=3
    """
    # Create directed graph
    G = nx.DiGraph()
    num_edges = edge_index.size(1)
    for i in range(num_edges):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(src, dst)
    
    # BFS up to 3 hops
    bfs_result = nx.single_source_shortest_path_length(G, center_node, cutoff=3)
    nodes_within_3hops = list(bfs_result.keys())
    subG = G.subgraph(nodes_within_3hops).copy()

    # Separate nodes by distance layer: 0,1,2,3
    layer0 = [center_node]
    layer1 = [n for n, dist in bfs_result.items() if dist == 1]
    layer2 = [n for n, dist in bfs_result.items() if dist == 2]
    layer3 = [n for n, dist in bfs_result.items() if dist == 3]

    # Build shells for shell_layout
    shells = []
    if layer0:
        shells.append(layer0)
    if layer1:
        shells.append(layer1)
    if layer2:
        shells.append(layer2)
    if layer3:
        shells.append(layer3)

    return subG, shells

# 3) Build the 3-hop subgraphs + shell lists for student/teacher
subG_student, shells_student = build_subgraph_3hop_and_shells(student_graph.edge_index, node_index)
subG_teacher, shells_teacher = build_subgraph_3hop_and_shells(teacher_graph.edge_index, node_index)

# 4) Plot them side-by-side for comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

#
# Student subgraph
#
pos_stu = nx.shell_layout(subG_student, nlist=shells_student)
# Draw the center node in red (larger node size)
nx.draw_networkx_nodes(
    subG_student, pos_stu,
    nodelist=[node_index],
    node_color='red',
    node_size=800,  # bigger circle
    ax=axes[0],
)
# Draw all other nodes in blue
nx.draw_networkx_nodes(
    subG_student, pos_stu,
    nodelist=[n for n in subG_student.nodes if n != node_index],
    node_color='lightblue',
    node_size=800,  # bigger circle
    ax=axes[0],
)
nx.draw_networkx_edges(
    subG_student, pos_stu,
    arrowstyle='-|>',
    arrows=True,
    arrowsize=12,
    ax=axes[0],
)
nx.draw_networkx_labels(
    subG_student, pos_stu,
    font_color='black',
    font_size=9,
    ax=axes[0],
)
axes[0].set_title(f"Student Graph (3-hop) - Center Node {node_index}")

#
# Teacher subgraph
#
pos_tea = nx.shell_layout(subG_teacher, nlist=shells_teacher)
nx.draw_networkx_nodes(
    subG_teacher, pos_tea,
    nodelist=[node_index],
    node_color='red',
    node_size=800,
    ax=axes[1],
)
nx.draw_networkx_nodes(
    subG_teacher, pos_tea,
    nodelist=[n for n in subG_teacher.nodes if n != node_index],
    node_color='lightgreen',
    node_size=800,
    ax=axes[1],
)
nx.draw_networkx_edges(
    subG_teacher, pos_tea,
    arrowstyle='-|>',
    arrows=True,
    arrowsize=12,
    ax=axes[1],
)
nx.draw_networkx_labels(
    subG_teacher, pos_tea,
    font_color='black',
    font_size=9,
    ax=axes[1],
)
axes[1].set_title(f"Teacher Graph (3-hop) - Center Node {node_index}")

plt.tight_layout()

# 5) Save figure to file
output_filename = "layered_3hop_subgraphs.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved as '{output_filename}'")
