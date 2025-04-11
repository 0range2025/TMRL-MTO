import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch_geometric.nn import GCNConv, Set2Set
from tqdm import tqdm
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of GCN train')

    # basic
    parser.add_argument('--datasetname', type=str, default="BBBP",
                        help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--csv_path', type=str, default="./data_process/data/", help='data.csv')

    return parser.parse_args()


args = parse_args()
df = pd.read_csv(args.csv_path)
y1 = df['label']

le = LabelEncoder()
label = le.fit_transform(y1)

smiles = df['smiles']
ys = label


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


class MoleculesDataset(InMemoryDataset):
    def __init__(self, root, transform=None):
        super(MoleculesDataset, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data2.csv'

    @property
    def processed_file_names(self):
        return 'data2.pt'

    def download(self):
        pass

    def process(self):
        datas = []
        for smile, y in zip(smiles, ys):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)

            embeddings = []
            for atom in mol.GetAtoms():
                embeddings.append(atom_featurizer.encode(atom))
            embeddings = np.array(embeddings, dtype=np.float32)
            embeddings = torch.tensor(embeddings)

            edges = []
            edge_attr = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

                edge_attr.append(bond_featurizer.encode(bond))
                edge_attr.append(bond_featurizer.encode(bond))

            edges = torch.tensor(edges).T
            edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32))

            y = torch.tensor(y, dtype=torch.long)

            data = Data(x=embeddings, edge_index=edges, y=y, edge_attr=edge_attr)
            datas.append(data)

        torch.save(self.collate(datas), self.processed_paths[0])


max_nodes = 128
dataset = MoleculesDataset(root="data2")

train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size],
                                                                           generator=torch.Generator().manual_seed(1))

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
val_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=8)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)


class GCNNet(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.fc2 = nn.Linear(2 * hidden_dim, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.set2set(x, batch)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def main(args):
    df = pd.read_csv(args.csv_path)
    y1 = df['label']

    le = LabelEncoder()
    label = le.fit_transform(y1)

    smiles = df['smiles']
    ys = label
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_feature_dim, hidden_dim = 24, 24

    epochs = 100
    model = GCNNet(node_feature_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    criterion = nn.CrossEntropyLoss(reduction="none")

    tra_loss = []
    tra_acc = []
    val_acc = []

    best_val_acc = 0.0
    best_model_path = f'best_gcn_model_{args.datasetname}.ckpt'

    for e in tqdm(range(epochs), desc="Epochs"):
        epoch_info = 'Epoch {}/{}'.format(e + 1, epochs)
        tqdm.write(epoch_info)
        model.train()
        epoch_loss = []

        train_total = 0
        train_correct = 0

        train_preds = []
        train_trues = []

        for data in train_loader:
            data = data.to(device)
            y = data.y
            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out, y)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.cpu().numpy())

            _, predict = torch.max(out.data, 1)
            train_total += y.shape[0] * 1.0
            train_correct += int((y == predict).sum())

            train_preds.extend(predict.detach().cpu().numpy())
            train_trues.extend(y.detach().cpu().numpy())

        epoch_loss = np.average(epoch_loss)

        sklearn_f1 = f1_score(train_trues, train_preds, average='macro')
        tqdm.write(
            'train Loss: {:.4f} Acc:{:.4f} f1:{:.4f}'.format(epoch_loss, train_correct / train_total, sklearn_f1))
        tra_loss.append(epoch_loss)
        tra_acc.append(train_correct / train_total)

        correct = 0
        total = 0

        valid_preds = []
        valid_trues = []

        with torch.no_grad():
            model.eval()
            for data in val_loader:
                data = data.to(device)
                labels = data.y
                outputs = model(data)

                _, predict = torch.max(outputs.data, 1)
                total += labels.shape[0] * 1.0
                correct += int((labels == predict).sum())

                valid_preds.extend(predict.detach().cpu().numpy())
                valid_trues.extend(labels.detach().cpu().numpy())

            sklearn_f1 = f1_score(valid_trues, valid_preds, average='macro')

            tqdm.write('val Acc: {:.4f} f1:{:.4f}'.format(correct / total, sklearn_f1))

            val_accuracy = correct / total
            val_acc.append(val_accuracy)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                tqdm.write(f'Found better model with validation accuracy: {val_accuracy:.4f}.')


if __name__ == "__main__":
    args = parse_args()

    main(args)
