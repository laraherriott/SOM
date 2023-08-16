import numpy as np
import os
import pandas as pd

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data

# import jazzy functions
from jazzy.api import molecular_vector_from_smiles
from jazzy.api import atomic_tuples_from_mol

class Featurisation:

    def __init__(self, mols: list, soms: list, max_atoms, max_bonds):
        self.mols = mols
        self.soms = soms
        self.max_atoms = max_atoms
        self.max_bonds = max_bonds

    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding

    def get_molecular_features(self, smi):
        mol_vec = molecular_vector_from_smiles(smi)

        return list(mol_vec.values())

    def get_jazzy_features(self, mol):
        atomic_features = atomic_tuples_from_mol(mol)

        return atomic_features

    def get_atom_features(self, atom, 
                          use_chirality = True, 
                          hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms
        
        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
        indices = list(range(len(permitted_list_of_atoms)))
        encoded = dict(list(indices, permitted_list_of_atoms))

        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
        
        # compute atom features
        
        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        
        n_heavy_neighbors_enc = [int(atom.GetDegree())]
        
        is_in_a_ring_enc = [int(atom.IsInRing())]
        
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        
        #atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + is_in_a_ring_enc + is_aromatic_enc + vdw_radius_scaled + covalent_radius_scaled
                                        
        if use_chirality == True:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc
        
        if hydrogens_implicit == True:
            n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc
        return atom_feature_vector, encoded.values().index(str(atom.GetSymbol()))

    def combine_atom_features(self, mol, matrix):
        # get jazzy features
        atomic_features = self.get_jazzy_features(mol)
        # list with element for each atom
        # then that element is itself a list of tuples for each feature in the format (feature, value)
        # all numeric except hybridisation which needs one hot encoding
        if atomic_features is None:
            return None
        target = list()
        else:
            for atom in mol.GetAtoms():
                generic_list, label = self.get_atom_features(atom)
                target.append(label)
                jazzy_list = [x[1] for x in atomic_features[atom.GetIdx()]]
                hybridisation = self.one_hot_encoding(jazzy_list[4], ["s", "sp", "sp2", "sp3", "sp3d", "sp3d2", "other"])
                jazzy_list = jazzy_list[0:4] + hybridisation + jazzy_list[5:]
                matrix[atom.GetIdx(), :] = generic_list + jazzy_list
            return matrix, target

    def get_bond_features(self, bond, 
                          use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        
        if use_stereochemistry == True:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

    def get_lengths(self, smile):
        unrelated_mol = Chem.MolFromSmiles(smile)
        jazzy_features = self.get_jazzy_features(unrelated_mol)
        atom1 = jazzy_features[0]
        list_atom1 = [x[1] for x in atom1]
        hybridisation = self.one_hot_encoding(list_atom1[4], ["s", "sp", "sp2", "sp3", "sp3d", "sp3d2", "other"])

        summary_list = list_atom1[0:4] + hybridisation + list_atom1[5:]

        num_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0))) + len(summary_list) # note use the combined features function here
        num_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        return num_node_features, num_edge_features


    def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(self):
        """
        Outputs:
        data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
        """
        som_list = list(self.soms)

        num_node_features, num_edge_features = self.get_lengths("O=O")

        data_list = []
        
        for mol in self.mols:
            som = som_list[:self.max_atoms]
            som_list = som_list[self.max_atoms:]
            assert len(som) == self.max_atoms, 'not equal lengths'
            #molecular_features = self.get_molecular_features(smiles)

            # get feature dimensions
            num_nodes = mol.GetNumAtoms()
            num_edges = 2*mol.GetNumBonds()
            
            # construct node feature matrix X of shape (n_nodes, n_node_features)
            node_feature = np.zeros((self.max_atoms, num_node_features))

            node_feature, label = self.combine_atom_features(mol, node_feature)
            if node_feature is None:
                continue
            else:
                node_feature = torch.tensor(node_feature, dtype = torch.float)
            
                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                edge = torch.stack([torch_rows, torch_cols], dim = 0)
            
                # construct edge feature array EF of shape (n_edges, n_edge_features)
                edge_features = np.zeros((2*self.max_bonds, num_edge_features))
            
                for (k, (i,j)) in enumerate(zip(rows, cols)):
                    edge_features[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
            
                edge_features = torch.tensor(edge_features, dtype = torch.float)
            
                # construct label tensor
                som_tensor = torch.tensor(np.array([som]))
            
                # construct Pytorch Geometric data object and append to data list
                data_list.append(Data(x = node_feature, edge_index = edge, edge_attr = edge_features, y = label))
        return data_list, num_node_features, self.max_atoms

