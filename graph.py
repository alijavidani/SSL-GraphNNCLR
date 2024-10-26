import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=3):
        super(GraphNet, self).__init__()
        
        # Define the layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(input_dim, hidden_dims[i]))
            else:
                self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling (to get graph-level embedding)
        y = global_mean_pool(x, batch)
        
        return x, y

    
# Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize the graph model:
# input_dim = 384       # Input feature dimension per node
# hidden_dims = [256, 128, 64]  # Hidden layer dimensions
# output_dim = 64       # Output dimension (graph-level embedding)
# num_layers = 3        # Number of GNN layers
# graph_model = GraphNet(input_dim, hidden_dims, output_dim, num_layers).to(device)