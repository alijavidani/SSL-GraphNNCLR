import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.datasets import Planetoid
import torch
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="CiteSeer")
data = dataset[0]

# Print information about the dataset
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
      super().__init__()
      self.gcn1 = GCNConv(dim_in, dim_h)
      self.gcn2 = GCNConv(dim_h, dim_out)
      self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.01,
                                        weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200

    model.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
          
    return model

@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc



# Create GCN model
# gcn = GCN(dataset.num_features, 16, dataset.num_classes)
# print(gcn)

# # # Train and test
# train(gcn, data)
# acc = test(gcn, data)
# print(f'\nGCN test accuracy: {acc*100:.2f}%\n')

# gat = GAT(dataset.num_features, 8, dataset.num_classes)
# print(gat)

# # Train and test
# train(gat, data)
# acc = test(gat, data)
# print(f'\nGAT test accuracy: {acc*100:.2f}%\n')

# # Initialize new untrained model
# untrained_gat = GAT(dataset.num_features, 8, dataset.num_classes)

# # Get embeddings
# h, _ = untrained_gat(data.x, data.edge_index)

# # Train TSNE
# tsne = TSNE(n_components=2, learning_rate='auto',
#          init='pca').fit_transform(h.detach())

# # Plot TSNE
# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
# plt.show()

# Get embeddings
# h, _ = gat(data.x, data.edge_index)

# # Train TSNE
# tsne = TSNE(n_components=2, learning_rate='auto',
#          init='pca').fit_transform(h.detach())

# # Plot TSNE
# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
# plt.show()