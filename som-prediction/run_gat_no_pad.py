import pandas as pd
import random
import os
import sys
import json

from rdkit import Chem
from rdkit.Chem import PandasTools

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import sklearn.preprocessing as sklearn_pre

from process_no_pad import PreProcessing
from gnns import GATv2
from train_no_pad import train, predict, soms_match_fn_test

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
Secondary = XenoSite_sdf['SECONDARY_SOM'].fillna(0)
Tertiary = XenoSite_sdf['TERTIARY_SOM'].fillna(0)

new_SOMS = []
for i in SOM_XenoSite:
    if len(i) == 1:
        new_SOMS.append([int(i)])
    else:
        strings = i.split()
        soms = [int(j) for j in strings]
        new_SOMS.append(soms)

second_SOMS = []
for i in Secondary:
    if i ==0:
        second_SOMS.append([0])
    elif len(i) == 1:
        second_SOMS.append([int(i)])
    else:
        strings = i.split()
        soms = [int(j) for j in strings]
        second_SOMS.append(soms)
        
third_SOMS = []
for i in Tertiary:
    if i == 0:
        third_SOMS.append([0])
    elif len(i) == 1:
        third_SOMS.append([int(i)])
    else:
        strings = i.split()
        soms = [int(j) for j in strings]
        third_SOMS.append(soms)

# preprocessing for featurisation, test/train split, and locading into batches
dataset = PreProcessing(MOLS_XenoSite, new_SOMS, second_SOMS, third_SOMS, config['split'], config['batch_size'], all_soms=True) # smiles, soms, split, batch_size

train_loader, validate_loader, test_loader, num_node_features, max_length,  smiles_train, smiles_validate, smiles_test, secondary_test, tertiary_test = dataset.create_data_loaders_gnn_som()

# set parameters for model training
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
epochs = config['n_epochs']

# instantiate model
model = GATv2(num_node_features)
optimiser = torch.optim.Adam(model.parameters(), lr=0.0062)
loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(44))

model.to(device)

model_output_dir = config['model']
model_file_name = 'GATv2_3'

# train model
y_scaler = sklearn_pre.StandardScaler() # subtract mean and scale to unit variance
loss_list, loss_validate, som_match, top_pred = train(model, train_loader, validate_loader, epochs, device, optimiser, loss_function, model_output_dir, model_file_name)
if len(loss_validate) == 0:
    loss_validate = [0]*len(loss_list)

# save trained model
torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))

# make predictions for test dataset
G_test, P_test, L_test = predict(model, device, test_loader)

actual_som_test, predictions, top_pred_test = soms_match_fn_test(G_test, P_test, L_test)

df_training_history = pd.DataFrame({"loss": loss_list, "validation_loss": loss_validate, "actual_soms": som_match, "predicted_primary_soms": top_pred})
df_training_history.to_csv(os.path.join(os.path.join(config['output'],"training_logs"), model_file_name + ".csv"))

test_results = pd.DataFrame({"actual_soms": actual_som_test, "predicted_primary_soms": top_pred_test, "all_probabilities": predictions, "molecule_lengths": L_test})
test_results.to_csv(os.path.join(os.path.join(config['output']), model_file_name + ".csv"))

test_extra_info = pd.DataFrame({'SMILES': smiles_test, 'SECONDARY': secondary_test, 'TERTIARY': tertiary_test })
test_extra_info.to_csv(os.path.join(os.path.join(config['output']), model_file_name + "_extra.csv"))

train_extra_info = pd.DataFrame({'SMILES': smiles_train})
train_extra_info.to_csv(os.path.join(os.path.join(config['output']), model_file_name + "_train_extra.csv"))

validate_extra_info = pd.DataFrame({'SMILES': smiles_validate})
validate_extra_info.to_csv(os.path.join(os.path.join(config['output']), model_file_name + "_validate_extra.csv"))
