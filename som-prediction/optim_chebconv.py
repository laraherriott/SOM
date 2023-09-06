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

from process_no_pad import PreProcessing

class Cheb(nn.Module):
    def __init__(self, trial, num_node_features):
        super().__init__()
        n_layers = trial.suggest_int("n_layers", 1, 5)
        width = int(trial.suggest_categorical("width", ['128', '256', '512']))
        k_choice = int(trial.suggest_categorical("K", ['5', '10', '15']))
        drop_prop = trial.suggest_float("drop_prop", 0.2, 0.5)

        # in channels is number of features
        # out channel is number of classes to predict, here 1 since just predicting atom SOM
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

def objective(trial):
    # Create a convolutional neural network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Cheb(trial, num_node_features).to(device)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weighting = trial.suggest_int('pos_weighting', int((max(max_length)-1)/2), int(max(max_length)-1))
    loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weighting))


    for epoch in range(epochs):
        validate_loss = list()
        # set model to training mode
        model.train()
        # loop over minibatches for training
        for batch, (data, nodes) in enumerate(train_loader):
            data = data.to(device)
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
            for batch, (d, n) in enumerate(validate_loader):
                d = d.to(device)
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
random.seed(10)

# take file name from command line
file = sys.argv[1]
with open(file) as config_file:
    config = json.load(config_file)

# Load data from sdf
XenoSite_sdf = PandasTools.LoadSDF(config['data'])
MOLS_XenoSite = XenoSite_sdf['ROMol']
SOM_XenoSite = XenoSite_sdf['PRIMARY_SOM']

new_SOMS = []
for i in SOM_XenoSite:
    if len(i) == 1:
        new_SOMS.append([int(i)])
    else:
        strings = i.split()
        soms = [int(j) for j in strings]
        new_SOMS.append(soms)

# preprocessing for featurisation, test/train split, and locading into batches
dataset = PreProcessing(MOLS_XenoSite, new_SOMS, config['split'], config['batch_size']) # smiles, soms, split, batch_size

train_loader, validate_loader, test_loader, num_node_features, max_length = dataset.create_data_loaders()

epochs = config['n_epochs']

study = optuna.create_study(direction="minimize") # will use median pruner by default
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

results = pd.DataFrame(trial.params, index=[0])
results.to_csv('optuna_cheb_output.csv')
