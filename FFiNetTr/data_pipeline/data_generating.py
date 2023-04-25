import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix

from torch_geometric.data import Data
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from FFiNet.features import atom_features

import networkx as nx
from FFiNet.models.model_utils import cal_distance, cal_angle, cal_dihedral


class DataGenerating:
    def __init__(self,
                 folder: str = './data_files/',
                 raw_dataset_name: str = None,
                 feature_dict_name: str = None,
                 dataset_name: str = None):
        self.folder = folder
        self.raw_dataset_name = raw_dataset_name
        self.feature_dict_name = feature_dict_name
        self.dataset_name = dataset_name
        self.features_dict_exist = False

        # load data file
        data_path = self.folder + self.raw_dataset_name

        self.csv = self.raw_dataset_name.split('.')[-1].lower() == 'csv'
        if self.csv:
            df = pd.read_csv(data_path)
            # find smiles column and collect smiles
            columns = df.columns.to_list()
            columns_lower = [column.lower() for column in columns]
            assert 'smiles' in columns_lower
            df.columns = columns_lower
            self.smiles_list = []
            for smiles in df['smiles']:
                self.smiles_list.append(smiles)
            self.mols = len(self.smiles_list) * [None]
            for i, smiles in enumerate(self.smiles_list):
                self.mols[i] = Chem.MolFromSmiles(self.smiles_list[i])
        else: # for SDF file
            self.mols = Chem.SDMolSupplier(data_path)

    def features_generating(self):
        
        features_dict = {}
        for i, mol in enumerate(tqdm(self.mols, desc='Data Processing')):
            
            features_dict[i] = {}
            # atom position
            if self.csv:

                num_atoms = mol.GetNumAtoms()
                mol_add_hs = Chem.AddHs(mol)
                
                # generate conformer by EDKDG method
                AllChem.EmbedMolecule(mol_add_hs, randomSeed=0xf00d)
                try:
                    conf = mol_add_hs.GetConformers()[0]
                except IndexError:
                    AllChem.EmbedMultipleConfs(mol_add_hs, 50, pruneRmsThresh=0.5)
                    try:
                        conf = mol_add_hs.GetConformers()[0]
                    except IndexError:
                        print(f'{Chem.MolToSmiles(mol)}\'s conformer can not be generated')
                        conf = None
                
                # Throw the molecules with no conformer generated
                if conf != None:
                    features_dict[i]['pos'] = conf.GetPositions()[:num_atoms, :]
                else:
                    features_dict.pop(i)
                    continue
            # Throw the invalid molecules in SDF file
            else:
                if mol == None:
                    print(f'Number of {i} can not generate mol object.')
                    features_dict.pop(i)
                    continue
            
                pos = mol.GetConformer().GetPositions()
                features_dict[i]['pos'] = pos

            # edge index (1-hop index)
            adj = Chem.GetAdjacencyMatrix(mol)
            coo_adj = coo_matrix(adj)
            features_dict[i]['edge_index'] = [coo_adj.row, coo_adj.col]

            # atom features (z for DimeNet)
            x = []
            z = []
            for atom in mol.GetAtoms():
                atom_generator_name_list = atom_features.get_available_features_generators()
                x.append(atom_features.atom_features_union(mol, atom, atom_generator_name_list))
                z.append(atom.GetAtomicNum())

            features_dict[i]['x'] = x
            features_dict[i]['z'] = z

        # save files as npy
        np.save(self.folder + self.feature_dict_name, features_dict)

        self.features_dict_exist = True

    def dataset_creating(self, target_name, dtype=torch.float32):

        # the function feature_generating must be first applied to get the feature dict 
        if self.features_dict_exist:
            features_dict = np.load(self.folder + self.feature_dict_name, allow_pickle=True).item()
            data_list = []

            for i, mol in enumerate(tqdm(self.mols, desc='Dataset creating')):
                if mol == None:
                     print(f'Number of {i} can not generate mol object.')
                     continue
                features = features_dict[i]

                # load label of dataset 
                if type(target_name) is list: # for multi-label tasks like QM9
                    y = [float(mol.GetProp(t)) for t in target_name]
                elif type(target_name) is pd.core.frame.DataFrame:
                    y = target_name.iloc[i, :]
                elif type(target_name) is pd.core.series.Series:
                    y = target_name.iloc[i]
                elif type(target_name) is dict:
                    y = target_name[mol.GetProp('_Name')]
                
                # generate data object
                data = Data(
                    z=torch.tensor(features['z'], dtype=torch.long),
                    x=torch.tensor(features['x'], dtype=dtype),
                    edge_index=torch.tensor(features['edge_index'], dtype=torch.long),
                    pos=torch.tensor(features['pos'], dtype=dtype),
                    y=torch.tensor(y, dtype=dtype))
                
                adj = Chem.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_matrix(adj)
                data.triple_index = subgraph_index(G, 2) # 2-hop index
                data.quadra_index = subgraph_index(G, 3) # 3-hop index

                dihedral_src_index, dihedral_mid2_index, dihedral_mid1_index, dihedral_dst_index = data.quadra_index
                angle_src_index, angle_mid_index, angle_dst_index = data.triple_index
                edge_src_index, edge_dst_index = data.edge_index


                # Calculating distance of each edge
                src_pos, dst_pos = data.pos.index_select(0, edge_src_index), data.pos.index_select(0, edge_dst_index)
                distance_per_edge = cal_distance(src_pos, dst_pos).unsqueeze(-1)
                data.distance_matrix = torch.cat([distance_per_edge, distance_per_edge ** 2], dim=1)


                # distance of the src atom and dst atom in each triedge
                angle_src_pos, angle_dst_pos = data.pos.index_select(0, angle_src_index), data.pos.index_select(0, angle_dst_index)
                distance_per_angle = cal_distance(angle_src_pos, angle_dst_pos).unsqueeze(-1)
                data.distance_matrix_angle = torch.cat([distance_per_angle ** (-6),
                                                distance_per_angle ** (-12),
                                                distance_per_angle ** (-1)], dim=1)
                # angle of each triedge
                angle_src_pos = data.pos.index_select(0, angle_src_index)
                angle_mid_pos = data.pos.index_select(0, angle_mid_index)
                angle_dst_pos = data.pos.index_select(0, angle_dst_index)
                angle_per_triedge = cal_angle(angle_src_pos, angle_mid_pos, angle_dst_pos)
                data.angle_matrix = torch.cat([angle_per_triedge, angle_per_triedge ** 2], dim=1)

                # distance of src atom and dst atom in each quaedge
                dihedral_src_pos = data.pos.index_select(0, dihedral_src_index)
                dihedral_dst_pos = data.pos.index_select(0, dihedral_dst_index)
                distance_per_dihedral = cal_distance(dihedral_src_pos, dihedral_dst_pos).unsqueeze(-1)
                data.distance_matrix_dihedral = torch.cat([distance_per_dihedral ** (-6),
                                                    distance_per_dihedral ** (-12),
                                                    distance_per_dihedral ** (-1)], dim=1)
                # dihedral of each quaedge
                dihedral_mid1_pos = data.pos.index_select(0, dihedral_mid1_index)
                dihedral_mid2_pos = data.pos.index_select(0, dihedral_mid2_index)
                dihedral_per_quaedge = cal_dihedral(dihedral_src_pos, dihedral_mid2_pos, dihedral_mid1_pos, dihedral_dst_pos)
                data.dihedral_matrix = torch.cat([torch.cos(dihedral_per_quaedge),
                                            torch.cos(dihedral_per_quaedge * 2),
                                            torch.cos(dihedral_per_quaedge * 3),
                                            torch.sin(dihedral_per_quaedge),
                                            torch.sin(dihedral_per_quaedge * 2),
                                            torch.sin(dihedral_per_quaedge * 3)], dim=1)


                data.smiles = Chem.MolToSmiles(mol)

                data_list.append(data)

            torch.save(data_list, self.folder + self.dataset_name)
        else:
            raise FileNotFoundError(f"There are no features dictionary in the {self.folder}")


# Finding all paths/walks of given length in a networkx graph
# code from https://www.py4u.net/discuss/162645
def subgraph_index(G, n):

    allpaths = []
    for node in G:
        paths = findPaths2(G, node , n)
        allpaths.extend(paths)
    allpaths = torch.tensor(allpaths, dtype=torch.long).T
    return allpaths

def findPaths2(G,u,n,excludeSet = None):
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths2(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths