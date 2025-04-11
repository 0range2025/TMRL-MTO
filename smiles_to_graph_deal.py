import warnings

from torch_geometric.data import Batch
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch_geometric.nn import GCNConv, Set2Set
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
csv_path = 'data_process/data/bbbp/processed/bbbp_processed_ac.csv'
df = pd.read_csv(csv_path)
y1 = df['label']

le = LabelEncoder()
label = le.fit_transform(y1)

smiles = df['smiles']
ys = label
# 忽略 RDKit 的警告
warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"Br", "C", "Cl", "F", "H", "I", "N", "O"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)





max_nodes = 128
#dataset = MoleculesDataset(root="data2")
class GCNNet(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim,feature_extraction=False):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.fc2 = nn.Linear(2 * hidden_dim, 8)
        self.fc3 = nn.Linear(8, 2)
        self.feature_extraction = feature_extraction

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.set2set(x, batch)

        if self.feature_extraction:
            return x

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
def smiles_to_graph_data(smiles_list, device):
    data_list = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        # 忽略 RDKit 的警告
        warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")
        # 移除氢原子（如果需要的话）
        mol = Chem.RemoveHs(mol)

        embeddings = []
        for atom in mol.GetAtoms():
            embeddings.append(atom_featurizer.encode(atom))
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = torch.tensor(embeddings).to(device)

        edges = []
        edge_attr = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

            edge_attr.append(bond_featurizer.encode(bond))
            edge_attr.append(bond_featurizer.encode(bond))

        edges = torch.tensor(edges, dtype=torch.long).to(device)
        edges = edges.permute(*torch.arange(edges.ndim - 1, -1, -1)).to(device)    #修改过
        edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32)).to(device)

        data = Data(x=embeddings, edge_index=edges, edge_attr=edge_attr)
        data_list.append(data)

    return Batch.from_data_list(data_list)


def load_model_smiles_to_graph(smiles_list, device):
    node_feature_dim, hidden_dim = 24, 24
    best_model_path = 'best_gcn_model.ckpt'
    GCNmodel = GCNNet(node_feature_dim, hidden_dim,feature_extraction=True).to(device)
    GCNmodel.load_state_dict(torch.load(best_model_path, map_location=device))
    #GCNmodel.eval()

    smiles_list_data = smiles_to_graph_data(smiles_list,device)
    smiles_graph_features =GCNmodel(smiles_list_data)


    return smiles_graph_features