# import torch
# import torch.nn as nn
# import torch_geometric
# from torch_geometric.nn import GCNConv
# import networkx as nx
# from torch_geometric.utils import from_networkx

# # Define a simple GCN model
# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
    
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

# def graph_consistency_loss(teacher_graph, student_graph, input_dim, hidden_dim, output_dim):
#     # Convert NetworkX graphs to PyTorch Geometric format
#     teacher_data = from_networkx(teacher_graph)
#     student_data = from_networkx(student_graph)
    
#     # Initialize GCN models
#     gcn_teacher = GCN(input_dim, hidden_dim, output_dim)
#     gcn_student = GCN(input_dim, hidden_dim, output_dim)
    
#     # Extract node embeddings
#     teacher_embeddings = gcn_teacher(teacher_data.x, teacher_data.edge_index)
#     student_embeddings = gcn_student(student_data.x, student_data.edge_index)
    
#     # Compute the consistency loss (e.g., Mean Squared Error between embeddings)
#     loss_fn = nn.MSELoss()
#     loss = loss_fn(teacher_embeddings, student_embeddings)
    
#     return loss

# # Example usage
# input_dim = 16  # Input feature dimension for nodes
# hidden_dim = 32  # Hidden layer dimension
# output_dim = 10  # Output embedding dimension

# # Assume teacher_graph and student_graph are NetworkX graphs with node features
# # Create example graphs
# teacher_graph = nx.karate_club_graph()
# student_graph = nx.karate_club_graph()

# # Initialize node features (here just random for example purposes)
# for graph in [teacher_graph, student_graph]:
#     for node in graph.nodes():
#         graph.nodes[node]['x'] = torch.rand(input_dim)

# # Call the graph_consistency_loss function
# loss = graph_consistency_loss(teacher_graph, student_graph, input_dim, hidden_dim, output_dim)
# print("Graph Consistency Loss:", loss.item())


# import torch
# import torch.nn as nn
# import torch_geometric
# from torch_geometric.nn import GCNConv, global_mean_pool

# # Define a GCN model with global pooling
# class GCNWithPooling(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCNWithPooling, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#         self.pool = global_mean_pool  # Global mean pooling
    
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)  # First GCN layer
#         x = torch.relu(x)              # Non-linearity
#         x = self.conv2(x, edge_index)  # Second GCN layer
        
#         # Apply global pooling to get graph-level embedding
#         # x = self.pool(x, batch)  # batch is needed to identify different graphs in a batch
        
#         return x

# # Example usage
# input_dim = 16  # Input feature dimension
# hidden_dim = 32  # Hidden layer dimension
# output_dim = 16  # Output feature dimension

# gcn_model = GCNWithPooling(input_dim, hidden_dim, output_dim)

# # Example input data
# x = torch.randn((10, input_dim))  # 10 nodes with input_dim features
# edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # Example edges
# batch = torch.tensor([0, 0, 0, 0, 0, 0 , 0, 0, 0, 0], dtype=torch.long)  # Batch vector to indicate graph membership of each node

# # Forward pass
# output = gcn_model(x, edge_index, batch)

# # Print the output
# print("Graph-level embedding:\n", output)
# print(output.shape)

##################################################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# class GraphNet(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, num_layers=3):
#         super(GraphNet, self).__init__()
        
#         # Define the layers
#         self.convs = nn.ModuleList()
#         self.convs.append(GCNConv(input_dim, hidden_dims[0]))  # First GCN layer
#         self.convs.append(GATConv(hidden_dims[0], hidden_dims[1], heads=1))  # Second GAT layer
#         self.convs.append(GCNConv(hidden_dims[1], hidden_dims[2]))  # Third GCN layer
        
#         self.num_layers = num_layers
    
#     def forward(self, x, edge_index, batch):
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.5, training=self.training)
        
#         # Global pooling (to get graph-level embedding)
#         # x = global_mean_pool(x, batch)
        
#         return x

# # Example usage:
# input_dim = 384       # Input feature dimension per node
# hidden_dims = [256, 128, 64]  # Hidden layer dimensions
# output_dim = 64       # Output dimension (graph-level embedding)
# num_layers = 3        # Number of GNN layers

# model = GraphNet(input_dim, hidden_dims, output_dim, num_layers)

# # Dummy input data for two graphs
# num_nodes_per_graph = 50000

# x = torch.randn(num_nodes_per_graph * 2, input_dim)          # Node features
# edge_index = torch.randint(0, num_nodes_per_graph, (2, 250000))  # Edge indices for graph 1
# edge_index_2 = torch.randint(0, num_nodes_per_graph, (2, 250000))  # Edge indices for graph 2

# # Shift edge indices of the second graph to avoid overlap
# edge_index_2 += num_nodes_per_graph

# # Combine edge indices
# combined_edge_index = torch.cat([edge_index, edge_index_2], dim=1)

# # Correctly create the batch tensor
# batch = torch.cat([torch.zeros(num_nodes_per_graph), torch.ones(num_nodes_per_graph)]).long()

# # Forward pass through the model for both graphs
# graph_embedding = model(x, combined_edge_index, batch)

# # Separate embeddings for the two graphs
# graph_embedding_1 = graph_embedding[0]
# graph_embedding_2 = graph_embedding[1]

# # Assuming binary classification with target label 1
# # target = torch.tensor([1])

# # Define the loss function and compute loss
# # loss_fn = nn.CrossEntropyLoss()
# # loss = loss_fn(graph_embedding, target)

# print(f'Graph-Level Embedding 1: {graph_embedding_1}')
# print(f'Graph-Level Embedding 2: {graph_embedding_2}')
# # print(f'Loss: {loss.item()}')

###############################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=3):
        super(GraphNet, self).__init__()
        
        # Define the layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))  # First GCN layer
        self.convs.append(GATConv(hidden_dims[0], hidden_dims[1], heads=1))  # Second GAT layer
        self.convs.append(GCNConv(hidden_dims[1], hidden_dims[2]))  # Third GCN layer
        
        self.num_layers = num_layers
    
    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Global pooling (to get graph-level embedding)
        # x = global_mean_pool(x, batch)
        
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage:
input_dim = 384       # Input feature dimension per node
hidden_dims = [256, 128, 64]  # Hidden layer dimensions
output_dim = 64       # Output dimension (graph-level embedding)
num_layers = 3        # Number of GNN layers

# Initialize model and move it to the GPU
model = GraphNet(input_dim, hidden_dims, output_dim, num_layers).to(device)

# Dummy input data for two graphs
num_nodes_per_graph = 50000

# Generate node features and move to GPU
x = torch.randn(num_nodes_per_graph * 2, input_dim).to(device)          # Node features

# Generate edge indices for the two graphs and move to GPU
edge_index = torch.randint(0, num_nodes_per_graph, (2, 250000)).to(device)  # Edge indices for graph 1
edge_index_2 = torch.randint(0, num_nodes_per_graph, (2, 250000)).to(device)  # Edge indices for graph 2

# Shift edge indices of the second graph to avoid overlap
edge_index_2 += num_nodes_per_graph

# Combine edge indices and move to GPU
combined_edge_index = torch.cat([edge_index, edge_index_2], dim=1).to(device)

# Correctly create the batch tensor and move to GPU
batch = torch.cat([torch.zeros(num_nodes_per_graph), torch.ones(num_nodes_per_graph)]).long().to(device)

# Forward pass through the model for both graphs
graph_embedding = model(x, combined_edge_index, batch)

# Separate embeddings for the two graphs (if needed)
graph_embedding_1 = graph_embedding[batch == 0]
graph_embedding_2 = graph_embedding[batch == 1]

# Print the graph embeddings
print(f'Graph-Level Embedding 1: {graph_embedding_1}')
print(f'Graph-Level Embedding 2: {graph_embedding_2}')
