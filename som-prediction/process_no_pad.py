import math
import random
import torch_geometric
#from poptorch_geometric import FixedSizeDataLoader, FixedSizeOptions
from rdkit import Chem
import torch
from torch_geometric.data import Dataset

from feature_no_pad import Featurisation

class PreProcessing:

    def __init__(self, mols, soms, split, batch_size):
        self.mols = mols
        self.training_prop = split[0]
        self.validate_prop = split[1]
        self.testing_prop = split[2]
        self.batch_size = batch_size

        self.no_atoms, self.no_bonds = self.find_largest_molecule(self.mols)
        self.soms = self.create_som_list(soms, self.no_atoms)

    def find_largest_molecule(self, mols):
        lengths = [mol.GetNumAtoms() for mol  in mols]
        bonds = [mol.GetNumBonds() for mol  in mols]
        # max_length = max(lengths)
        # max_bonds = max(bonds)

        return lengths, bonds

    def create_som_list(self, soms, max):
        new_soms = []

        for (som, atom_number) in zip(soms, max):
            mol_list = [int(0)]*atom_number
            mol_list[som-1] = int(1)
            new_soms.append(mol_list)

        return new_soms

    def featurise(self):
        features = Featurisation(self.mols, self.soms, self.no_atoms, self.no_bonds)

        graph_object, num_node_features, max_length = features.create_pytorch_geometric_graph_data_list_from_smiles_and_labels()
        return graph_object, num_node_features, max_length
    
    def test_train_split(self):

        data, num_features, lengths = self.featurise()

        random.shuffle(data)

        train_length = math.ceil(self.training_prop*len(data))
        remaining = data[train_length:]
        validate_length = math.ceil(self.validate_prop*len(remaining))
        test_length = len(data) - (train_length+validate_length)

        train_dataset = data[:train_length]
        validate_dataset = data[train_length:(train_length+validate_length)]
        test_dataset = data[(train_length+validate_length):]

        return train_dataset, test_dataset, validate_dataset, num_features, lengths


    def create_data_loaders(self):
        train, test, validate, num_features, lengths = self.test_train_split()

        # torch.save(train, 'training_no_pad.pt')
        # torch.save(validate, 'validate_no_pad.pt')
        # torch.save(test, 'test_no_pad.pt')

        train_dataset = MyDataset(train)
        test_dataset = MyDataset(test)
        validate_dataset = MyDataset(validate)

        train_loader = torch_geometric.loader.DataLoader(train_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle = True)
        test_loader = torch_geometric.loader.DataLoader(test_dataset,
                                                        batch_size=self.batch_size)
        if self.validate_prop != 0:
            validate_loader = torch_geometric.loader.DataLoader(validate_dataset,
                                                                batch_size=self.batch_size)
        else:
            validate_loader = None

        
        return train_loader, validate_loader, test_loader, num_features, lengths
    

class MyDataset(Dataset):
    def __init__(self, dataset, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.dataset = dataset
    
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        return data, len(data.x)
