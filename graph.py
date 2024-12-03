import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers=3):
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
        x_initial = x
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        # x += x_initial  # Residual connection
        y = global_mean_pool(x, batch)
        return x, y


class GraphNetWithDINO(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, dino_head):
        super(GraphNetWithDINO, self).__init__()
        
        self.graph_net = GraphNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
        )
        self.dino_head = dino_head  # DINOHead instance
        
    def forward(self, x, edge_index, batch):
        # Check if the model is in training mode
        if self.training:
            # Training phase: Use GNN layers
            node_embeddings, global_embedding = self.graph_net(x, edge_index, batch)
        else:
            # Inference phase: Skip GNN layers
            node_embeddings = x  # Use backbone features directly
            global_embedding = x.mean(dim=0, keepdim=True)  # Optional: Compute a global embedding

        # Apply DINOHead to embeddings
        node_embeddings = self.dino_head(node_embeddings)
        global_embedding = self.dino_head(global_embedding)

        return node_embeddings, global_embedding