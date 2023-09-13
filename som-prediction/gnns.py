import torch.nn as nn
import torch_geometric.nn as gnn
from torch.nn import Linear

class GCN(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        n_layers = 2
        width = 256
        drop_prop = 0.27
        layers = []
        layers.append((gnn.GCNConv(num_node_features, width), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        for i in range(n_layers):
            layers.append((gnn.GCNConv(width, width), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GCNConv(width, width), 'x, edge_index -> x'))
        layers.append((gnn.Linear(width, 1)))


        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)

class ChebConv(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        width = 512
        n_layers = 3
        k_choice = 5
        drop_prop = 0.37
        layers = []
        layers.append((gnn.ChebConv(num_node_features, width, K=k_choice), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        for i in range(n_layers):
            layers.append((gnn.ChebConv(width, width, K=k_choice), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.ChebConv(width, width, K=k_choice), 'x, edge_index -> x'))
        layers.append((gnn.Linear(width, 1)))

        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)
    
class GATv2(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        width = 512
        drop_prop = 0.27
        gat_drop = 0.12
        head = 9
        layers = []
        layers.append((gnn.GATv2Conv(num_node_features, num_node_features, heads = head, dropout = gat_drop), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GATv2Conv(num_node_features*head, num_node_features*4, heads = head, dropout = gat_drop), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GATv2Conv(num_node_features*4*head, width, heads = 1, dropout = gat_drop), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.Linear(width, 1)))

        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)