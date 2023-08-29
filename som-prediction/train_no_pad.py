import math
import statistics
import numpy as np
import torch
import torch.nn as nn

def train(model, train_loader, validate_loader, epochs, device, optimiser, loss_function):

# model, device, loss_fn, train_loader, valid_loader, test_loader, optimizer, model_file_name, model_output_dir, n_epochs,
#                is_GNN, train_data.y_scaler
    # create lists to store information in
    loss_list = list()
    loss_validate = list()
    som_match = list()
    top_pred = list()

    print('Beginning training...')

    # loop over training epochs
    for epoch in range(epochs):
        validate_loss = list()
        train_loss = list()
        # set model to training mode
        model.train()
        # loop over minibatches for training
        for batch, (data, nodes) in enumerate(train_loader):
            data = data.to(device)
            # flat_list = []
            # for row in data.y:
            #     flat_list += row
            # data.y = torch.tensor(flat_list).view(-1, 1)
            # set past gradient to zero
            optimiser.zero_grad()
            # compute current value of loss function via forward pass
            output = model(data)
            print('Training for batch {} complete...'.format(batch))
            loss_function_value = loss_function(output, data.y.to(device).float())
            train_loss.append(loss_function_value.item())
            
            # compute current gradient via backward pas
            loss_function_value.backward()
            # update model weights using gradient and optimisation method
            optimiser.step()

        model.eval() # prep model for evaluation
        for batch, (d, n) in enumerate(validate_loader):
            d = d.to(device)
            # flat_list = []
            # for row in d.y:
            #     flat_list += row
            # d.y = torch.tensor(flat_list).view(-1, 1)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(d)
            # calculate the loss
            loss = loss_function(output, d.y.to(device).float())
            # record validation loss
            validate_loss.append(loss.item())

        check_validate_loss = statistics.mean(validate_loss)
        check_train_loss = statistics.mean(train_loss)

        G_train, P_train, L_train = predict(model, device, train_loader)

        actual_som, top_prediction = soms_match_fn(G_train, P_train, L_train)
        som_match.append(actual_som)
        top_pred.append(top_prediction)
        loss_list.append(check_train_loss)
        loss_validate.append(check_validate_loss)

            # early_stopping(check_train_loss, check_validate_loss)
        
            # if early_stopping:
            #     print("Early stopping")
            #     break

            # check_validate_loss = None
            # check_train_loss = None

        print('Epoch {} completed'.format(epoch))
    
    return loss_list, loss_validate, som_match, top_pred

def predict(model, device, loader, mc_dropout=False, verbose=True):
    """
    function performing predictions of a GNN

    model: torch.nn.model
        the model to be trained
    device: torch.device
        indicates whether model is trained on GPU or CPU
    loader:
        data loader for data which predictions are performed on
    mc_dropout: bool
        whether we perform prediction in the course of a MC dropout model
    verbose: bool
        whether to print how many data points prediction is performed on
    y_scaler: sklearn.preprocessing.StandardScaler
        standard scaler transforming the target variable
    """
    sigmoid = nn.Sigmoid()
    model.eval()
    if mc_dropout:
        model.dropout_layer.train()
    total_preds = list()
    total_labels = list()
    total_lengths = list()
    if verbose:
        print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for (data, num_nodes) in loader:
            data = data.to(device)
            output = model(data)
            total_preds.append(sigmoid(output))
            total_labels.append(data.y.view(-1, 1))
            total_lengths.append(num_nodes)

    return total_labels, total_preds, total_lengths


# def early_stopping(train_loss, valid_loss):


def soms_match_fn(G, P, L):
    real = list()
    #predictions = list()
    max_atoms = list()
    for k in range(len(G)):
        new_G = []
        for i in G[k]:
            new_G.append(int(i))
        per_mol = list()
        count = 0
        for i in L[k]:
            per_mol.append(new_G[count:count+int(i)])
            count += i
        
        batch_real = list()
        for mol in per_mol:
            batch_real.append(mol.index(1)+1)
            
        real.append(batch_real)

        new_P = []
        for i in P[k]:
            new_P.append(int(i))
        per_mol_P = list()
        count = 0
        for i in L[k]:
            per_mol_P.append(new_P[count:int(i)+count])
            count += i

        max_atom = list()
        #batch_predictions = list()
        for mol_P in per_mol_P:
            max_atom.append(mol_P.index(max(mol_P))+1)
            #batch_predictions.append(mol_P)
        #predictions.append(batch_predictions)
        max_atoms.append(max_atom)

    return real, max_atoms

def soms_match_fn_test(G, P, L):
    real = list()
    predictions = list()
    max_atoms = list()
    for k in range(len(G)):
        new_G = []
        for i in G[k]:
            new_G.append(int(i))
        per_mol = list()
        count = 0
        for i in L[k]:
            per_mol.append(new_G[count:int(i)+count])
            count += i
        
        batch_real = list()
        for mol in per_mol:
            batch_real.append(mol.index(1)+1)
        
        real.append(batch_real)

        new_P = []
        for i in P[k]:
            new_P.append(int(i))
        per_mol_P = list()
        count = 0
        for i in L[k]:
            per_mol_P.append(new_P[count:int(i)+count])
            count += i

        max_atom = list()
        batch_predictions = list()
        for mol_P in per_mol_P:
            max_atom.append(mol_P.index(max(mol_P))+1)
            batch_predictions.append(mol_P)
        
        predictions.append(batch_predictions)
        max_atoms.append(max_atom)

    return real, predictions, max_atoms

# F score? because binary classification