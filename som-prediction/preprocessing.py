import math
import random
import torch_geometric
#from poptorch_geometric import FixedSizeDataLoader, FixedSizeOptions
from rdkit import Chem

from featurisation import Featurisation

class PreProcessing:

    def __init__(self, mols, soms, split, batch_size):
        self.mols = mols
        self.training_prop = split[0]
        self.validate_prop = split[1]
        self.testing_prop = split[2]
        self.batch_size = batch_size

        self.max_atoms, self.max_bonds = self.find_largest_molecule(self.mols)
        self.soms = self.create_som_list(soms, self.max_atoms)

    def find_largest_molecule(self, mols):
        lengths = [mol.GetNumAtoms() for mol  in mols]
        bonds = [mol.GetNumBonds() for mol  in mols]
        max_length = max(lengths)
        max_bonds = max(bonds)

        return max_length, max_bonds

    def create_som_list(self, soms, max):
        new_soms = []

        for atom_number in soms:
            mol_list = [int(0)]*max
            mol_list[atom_number-1] = int(1)
            for i in mol_list:
                new_soms.append([i])

        return new_soms

    def featurise(self):
        features = Featurisation(self.mols, self.soms, self.max_atoms, self.max_bonds)

        graph_object, num_node_features, max_length = features.create_pytorch_geometric_graph_data_list_from_smiles_and_labels()

        return graph_object, num_node_features, max_length
    
    def test_train_split(self):

        data, num_features, max_length = self.featurise()
        random.shuffle(data)

        train_length = math.ceil(self.training_prop*len(data))
        remaining = data[train_length:]
        validate_length = math.ceil(self.validate_prop*len(remaining))
        test_length = len(data) - (train_length+validate_length)

        train_dataset = data[:train_length]
        validate_dataset = data[train_length:(train_length+validate_length)]
        test_dataset = data[(train_length+validate_length):]

        return train_dataset, test_dataset, validate_dataset, num_features, max_length


    def create_data_loaders(self):
        train, test, validate, num_features, max_length = self.test_train_split()

        train_loader = torch_geometric.loader.DataLoader(train,
                                                         batch_size=self.batch_size,
                                                         shuffle=True)
        test_loader = torch_geometric.loader.DataLoader(test,
                                                        batch_size=self.batch_size)
        if self.validate_prop != 0:
            validate_loader = torch_geometric.loader.DataLoader(validate,
                                                                batch_size=self.batch_size,
                                                                shuffle=True)
        else:
            validate_loader = None

        
        return train_loader, validate_loader, test_loader, num_features, max_length
        