import torch.nn as nn
import torch_geometric.nn as gnn
from torch.nn import Linear

class GCN(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        # in channels is number of features
        # out channel is number of classes to predict, here 1 since just predicting atom SOM
        self.conv1 = gnn.GCNConv(num_node_features, 128)
        self.conv2 = gnn.GCNConv(128, 128)
        self.conv3 = gnn.GCNConv(128, 128)
        self.conv4 = gnn.GCNConv(128, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        #self.output = gnn.Linear(512, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv4(x, edge_index)

        return x
    
# class SAGE()

# class GATV2 - state of the art?
    
# class GIN()

class ChebConv:
    def __init__(self, width, depth, featureCount):
        self.width = width
        self.depth = depth
        self.featureCount = featureCount
        self.conv = lambda a, b: gnn.ChebConv(a, b, K=15)
    def createGnnSom(self):
        layerSizes = [self.featureCount] + [self.width] * self.depth + [1]

        modules = []
        for i in range(len(layerSizes) - 1):
            modules.append((self.conv(layerSizes[i], layerSizes[i + 1]), 'x, edge_index -> x'))
            if i != len(layerSizes) - 2:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(0.5))
        return gnn.Sequential('x, edge_index', modules)