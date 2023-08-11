import pandas as pd
import random
import os
import sys
import json
import statistics

from rdkit import Chem
from rdkit.Chem import PandasTools

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import sklearn.preprocessing as sklearn_pre

import optuna

from preprocessing import PreProcessing

class GCN(nn.Module):
    def __init__(self, trial, num_node_features):
        super().__init__()
        n_layers = trial.suggest_int("n_layers", 1, 5)
        width = trial.suggest_int("width", 128, 512)
        drop_prop = trial.suggest_float("drop_prop", 0.2, 0.5)

        relu = nn.ReLU()
        drop = nn.Dropout(drop_prop)

        # in channels is number of features
        # out channel is number of classes to predict, here 1 since just predicting atom SOM
        layers = []
        layers.append((gnn.GCNConv(num_node_features, width/2), 'x, edge_index -> x'))
        layers.append(relu())
        layers.append(drop())
        layers.append((gnn.GCNCcnv(width/2, 3*(width/4)), 'x, edge_index -> x'))
        layers.append(relu())
        layers.append(drop())
        layers.append((gnn.GCNConv(3*(width/4), width), 'x, edge_index -> x'))
        layers.append(relu())
        layers.append(drop())
        for i in range(n_layers):
            layers.append((gnn.GCNConv(width, width), 'x, edge_index -> x'))
            if i != n_layers - 2:
                layers.append(relu())
                layers.append(drop())
        layers.append((gnn.GCNConv(width, width/2), 'x, edge_index -> x'))
        layers.append((gnn.GCNConv(width/2, 3*(width/4)), 'x, edge_index -> x'))
        layers.append((gnn.GCNConv(width/2, 1), 'x, edge_index -> x'))

        self.layers = gnn.Sequential('x, edge_index', layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        return self.layers(x, edge_index)

def objective(trial):
    # Create a convolutional neural network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(trial, num_node_features).to(device)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(max_length-1))


    for epoch in range(epochs):
        validate_loss = list()
        # set model to training mode
        model.train()
        # loop over minibatches for training
        for batch, data in enumerate(train_loader):
            data = data.to(device)
            flat_list = []
            for row in data.y:
                flat_list += row
            data.y = torch.tensor(flat_list).view(-1, 1)
            # set past gradient to zero
            optimiser.zero_grad()
            # compute current value of loss function via forward pass
            output = model(data)
            print('Training for batch {} complete...'.format(batch))
            loss_function_value = loss_function(output, data.y.to(device).float())
            
            # compute current gradient via backward pas
            loss_function_value.backward()
            # update model weights using gradient and optimisation method
            optimiser.step()

        model.eval() # prep model for evaluation
        with torch.no_grad():
            for batch, d in enumerate(validate_loader):
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(d)
                # calculate the loss
                loss = loss_function(output, d.y.to(device).float())
                # record validation loss
                validate_loss.append(loss.item())

        check_validate_loss = statistics.mean(validate_loss)

        trial.report(check_validate_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return loss

# set seed
random.seed(1)

# take file name from command line
file = sys.argv[1]
with open(file) as config_file:
    config = json.load(config_file)

# Load data from sdf
XenoSite_sdf = PandasTools.LoadSDF(config['data'])
MOLS_XenoSite = XenoSite_sdf['ROMol']
SOM_XenoSite = XenoSite_sdf['PRIMARY_SOM']

# in cases where multiple SOMs are listed, take the first only
SOM_XenoSite = [int([*i][0]) for i in SOM_XenoSite]

# preprocessing for featurisation, test/train split, and locading into batches
dataset = PreProcessing(MOLS_XenoSite[:10], SOM_XenoSite[:10], config['split'], config['batch_size']) # smiles, soms, split, batch_size

train_loader, validate_loader, test_loader, num_node_features, max_length = dataset.create_data_loaders()

epochs = config['n_epochs']

study = optuna.create_study(direction="minimize") # will use median pruner by default
study.optimize(objective, n_trials=100, timeout=600)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

