import pandas as pd
import random
import os
import sys
import json

from rdkit import Chem
from rdkit.Chem import PandasTools

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn

import sklearn.preprocessing as sklearn_pre

from process_no_pad import PreProcessing
from gnns import GCN
from train_no_pad import train, predict, soms_match_fn_test

# set seed
random.seed(1)

# take file name from command line
file = sys.argv[1]
with open(file) as config_file:
    config = json.load(config_file)

XenoSite_sdf = PandasTools.LoadSDF(config['data'])
MOLS_XenoSite = XenoSite_sdf['ROMol']
SOM_XenoSite = XenoSite_sdf['PRIMARY_SOM']

# in cases where multiple SOMs are listed, take the first only
#SOM_XenoSite = [int([*i][0]) for i in SOM_XenoSite]

# or take all primary SOMS
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

# set parameters for model training
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
epochs = config['n_epochs']

# instantiate model
model = GCN(73)
optimiser = torch.optim.Adam(model.parameters(), lr=0.00035)
#print('previous positive weighting: ', (max_length-1))
loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(47))

model.to(device)

# train model
y_scaler = sklearn_pre.StandardScaler() # subtract mean and scale to unit variance
loss_list, loss_validate, som_match, top_pred = train(model, train_loader, validate_loader, epochs, device, optimiser, loss_function)
if len(loss_validate) == 0:
    loss_validate = [0]*len(loss_list)

# save trained model
model_output_dir = config['model']
model_file_name = 'GCN_2'
torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))

# make predictions for test dataset
G_test, P_test, L_test = predict(model, device, test_loader)

actual_som_test, predictions, top_pred_test = soms_match_fn_test(G_test, P_test, L_test)

df_training_history = pd.DataFrame({"loss": loss_list, "validation_loss": loss_validate, "som_actual": som_match, "top_predictions": top_pred})
df_training_history.to_csv(os.path.join(os.path.join(config['output'],"training_logs"), model_file_name + ".csv"))

test_results = pd.DataFrame({"actual_soms": actual_som_test, "predicted_primary_soms": top_pred_test, "all_probabilities": predictions, "molecule_lengths": L_test})
test_results.to_csv(os.path.join(os.path.join(config['output']), model_file_name + ".csv"))

#print("Top 1 acuracy: ", number_correct/len(som_actual_test[0][0]))

# correct = (pred[test_loader.test_mask] == test_loader.y[test_loader.test_mask]).sum()
# acc = int(correct) / int(test_loader.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')
