import pandas as pd
import random
import os
import sys
import json
import torch

import sklearn.preprocessing as sklearn_pre

from preprocessing import PreProcessing
from gnns import GCN
from training import train, predict, soms_match_fn

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
dataset = PreProcessing(MOLS_XenoSite[:100], SOM_XenoSite[:100], config['split'], config['batch_size']) # smiles, soms, split, batch_size

train_loader, validate_loader, test_loader, num_node_features, max_length = dataset.create_data_loaders()

# set parameters for model training
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device=torch.device('cpu')
epochs = config['n_epochs']

# instantiate model
model = GCN(num_node_features)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(max_length-1))

model.to(device)

# train model
y_scaler = sklearn_pre.StandardScaler() # subtract mean and scale to unit variance
loss_list, loss_validate, som_match, top_pred = train(model, train_loader, validate_loader, epochs, device, optimiser, loss_function, max_length)
if len(loss_validate) == 0:
    loss_validate = [0]*len(loss_list)

# save trained model
model_output_dir = '../models'
model_file_name = 'GCN_1'
torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))

# make predictions for test dataset
G_test, P_test = predict(model, device, test_loader)

predictions, top_pred_test = soms_match_fn(G_test, P_test, max_length)

som_actual = [tup[0] for tup in som_match]
som_predict = [tup[1] for tup in som_match]

som_actual_test = predictions[0]
som_probabilities = predictions[1]


df_training_history = pd.DataFrame({"loss": loss_list, "validation_loss": loss_validate, "som_actual": som_actual, "top_predictions": top_pred, "som_probabilities": som_predict})
df_training_history.to_csv(os.path.join(os.path.join("../output","training_logs"), model_file_name + ".csv"))

test_results = pd.DataFrame({"actual_soms": [som_actual_test], "predicted_primary_soms": [top_pred_test], "som_probabilities": [som_probabilities]})
test_results.to_csv(os.path.join(os.path.join("../output"), model_file_name + ".csv"))

#print("Top 1 acuracy: ", number_correct/len(som_actual_test[0][0]))

# correct = (pred[test_loader.test_mask] == test_loader.y[test_loader.test_mask]).sum()
# acc = int(correct) / int(test_loader.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')