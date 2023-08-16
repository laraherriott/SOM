import torch.nn as nn
import torch_geometric.nn as gnn
from torch.nn import Linear

class GCN(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        n_layers = 4
        width = 256
        drop_prop = 0.22 

        # in channels is number of features
        # out channel is number of classes to predict, here 1 since just predicting atom SOM
        layers = []
        layers.append((gnn.GCNConv(num_node_features, int(width/2)), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append((gnn.GCNConv(int(width/2), int(3*(width/4))), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GCNConv(int(3*(width/4)), width), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        for i in range(n_layers):
            layers.append((gnn.GCNConv(width, width), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GCNConv(width, int(3*width/4)), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GCNConv(int(3*(width/4)), int(width/2)), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.GCNConv(int(width/2), 1), 'x, edge_index -> x'))

        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)
    
# class SAGE()

# class GATV2 - state of the art?
    
# class GIN()

class ChebConv(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        width = 128
        n_layers = 5
        k_choice = 5
        drop_prop = 0.266
        layers = []
        layers.append((gnn.ChebConv(num_node_features, int(width/2), K=k_choice), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append((gnn.ChebConv(int(width/2), int(3*(width/4)), K=k_choice), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.ChebConv(int(3*(width/4)), width, K=k_choice), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        for i in range(n_layers):
            layers.append((gnn.ChebConv(width, width, K=k_choice), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.ChebConv(width, int(3*width/4), K=k_choice), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.ChebConv(int(3*(width/4)), int(width/2), K=k_choice), 'x, edge_index -> x'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prop))
        layers.append((gnn.ChebConv(int(width/2), 1, K=k_choice), 'x, edge_index -> x'))

        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)
    
class GATv2(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        width = 256
        drop_prop = 0.413
        gat_drop = 0.122
        head = 4
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
        layers.append((gnn.GCNConv(width, 1), 'x, edge_index -> x'))

        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)