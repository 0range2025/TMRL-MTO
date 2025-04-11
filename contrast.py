import argparse
import math
from torch.utils.data import Dataset
import re
import json
from PIL import Image
import torchvision.transforms as transforms
from Models.ImageMol import ImageMol
import torch.nn.functional as F
from Models.mamba import MambaConfig, Mamba
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch_geometric.nn import GCNConv, Set2Set
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of GCN train')

    # basic
    parser.add_argument('--datasetname', type=str, default="BBBP",
                        help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--csv_path', type=str, default="./data_process/data/", help='data.csv')

    return parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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




class GCNNet(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, feature_extraction=False):
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



def get_max_len(smiles):
    return max(len(re.findall(r'(\[[^\[\]]*\]|Br|Cl|[a-z]|\d|[A-Z]|[^a-zA-Z0-9])', s)) for s in smiles)

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, d_model)
        self.max_len = max_len

    def forward(self, x):
        B, L, D = x.size()
        pos = torch.arange(L, device=x.device)
        rel_pos = pos[None, :] - pos[:, None] + self.max_len - 1  # 偏移确保正值索引
        rel_pos_emb = self.rel_pos_emb(rel_pos)  # (L, L, D)
        rel_pos_emb = rel_pos_emb.mean(dim=1)  # 简化为 (L, D)
        return x + rel_pos_emb.unsqueeze(0)  # (B, L, D)

def get_mamba_config(seq_len: int, d_model: int = 128, n_layers: int = 3):
    if seq_len <= 32:  # 可自定义短序列阈值
        expand_factor = 1
        d_state = 4
    else:
        expand_factor = 2
        d_state = 8

    return MambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        expand_factor=expand_factor,
        d_state=d_state,
        use_cuda=torch.cuda.is_available()
    )
class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mamba_config):
        super(MambaModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.mamba = Mamba(mamba_config)
        self.position_encoding = RelativePositionEncoding(256, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        B, L = src.shape
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.position_encoding(src)
        if self.mamba is None:
            config = get_mamba_config(seq_len=L, d_model=self.hidden_dim)
            self.mamba = Mamba(config).to(src.device)

        output = self.mamba(src)
        output = self.decoder(output)
        return output

def load_model_and_decode(smiles_list):
    # 模型参数
    input_dim = 38
    hidden_dim = 128
    output_dim = 38
    n_layers = 3
    char_to_id = {'<pad>': 0}

    mamba_config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)

    smiles_model = MambaModel(input_dim, hidden_dim, output_dim, mamba_config).to(device)
    smiles_model.load_state_dict(torch.load("smiles_mamba.ckpt"))
    smiles_model.eval()

    def tokenize(smile):
        regex = r'(\[[^\[\]]*\]|Br|Cl|[a-z]|\d|[A-Z]|[^a-zA-Z0-9])'
        return re.findall(regex, smile)


    tokenized_smiles = [tokenize(s) for s in smiles_list]

    max_len = max(len(s) for s in tokenized_smiles)
    input_ids = [[char_to_id.get(ch, char_to_id['<pad>']) for ch in s] + [char_to_id['<pad>']] * (max_len - len(s)) for
                 s in tokenized_smiles]
    input_tensor = torch.tensor(input_ids).to(device)

    with torch.no_grad():
        encoded_vectors = smiles_model.encode(input_tensor)
        return encoded_vectors



def extract_image_features(images):
    jigsaw_classes = 100 + 1
    label1_classes = 100
    label2_classes = 1000
    label3_classes = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Imagemodel = ImageMol("ResNet18", jigsaw_classes, label1_classes, label2_classes, label3_classes).to(device)


    checkpoint = torch.load("data/ImageMol.pth.tar", map_location=device)
    Imagemodel.load_state_dict(checkpoint['state_dict'])

    Imagemodel.eval()
    with torch.no_grad():
        features = Imagemodel(images)[0].to(device)
    return features





def smiles_to_graph_data(smiles_list, device):
    data_list = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)

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

        edges = torch.tensor(edges).T.to(device)
        edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32)).to(device)

        data = Data(x=embeddings, edge_index=edges, edge_attr=edge_attr)
        data_list.append(data)

    return Batch.from_data_list(data_list)


def load_model_smiles_to_graph(smiles_list, device):
    node_feature_dim, hidden_dim = 24, 24
    best_model_path = f'best_gcn_model_{args.datasetname}.ckpt'
    GCNmodel = GCNNet(node_feature_dim, hidden_dim, feature_extraction=True).to(device)
    GCNmodel.load_state_dict(torch.load(best_model_path, map_location=device))


    smiles_list_data = smiles_to_graph_data(smiles_list, device)
    smiles_graph_features = GCNmodel(smiles_list_data)

    return smiles_graph_features



class SMILESImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['image']).convert('RGB')
        img = self.transform(img)
        smiles = item['smiles']
        return smiles, img



class PlainMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            dropout=0.,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def init_weights(self):
        pass

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        E = query.size(-1)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def set_parameters(self, torch_tgt_module):
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim
        self.qkv.weight.data = torch_tgt_module.in_proj_weight.data
        self.qkv.bias.data = torch_tgt_module.in_proj_bias.data
        self.proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data


class SimpleCLIP(nn.Module):
    def __init__(self, text_dim, img_dim, embedding_dim, device):
        super(SimpleCLIP, self).__init__()
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.text_projection = nn.Parameter(torch.empty(text_dim, 128))
        self.img_projection = nn.Parameter(torch.empty(img_dim, 128))

        self.init_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 1))

        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))

        # 规范化权重，使它们的和为1
        self.softmax = nn.Softmax(dim=0)

        # 特征融合的连接层
        # 定义一个全连接层用于匹配SMILES编码器的输出维度
        self.fc = nn.Linear(512, 128)
        self.fc1 = nn.Linear(text_dim + img_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, 1)  # 改为输出1个值
        self.fc4 = nn.Linear(48, 128)  # 将图数据升维
        self.fc5 = nn.Linear(128, 48)  # 将图数据升维
        self.attention = PlainMultiHeadAttention(embed_dim=256, num_heads=8).to(device)  # 原来256
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),  # 增加一层，保持更高维度
            nn.ReLU(),
            nn.Linear(256, 128),  #
            nn.ReLU(),

            nn.Linear(128, 1),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 256),  # 增加一层，保持更高维度
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 2),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),  # 增加一层，保持更高维度
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 12),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 256),  # 增加一层，保持更高维度
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 27),
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(256, 256),  # 增加一层，保持更高维度
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 617),
        )
        self._initialize_weights()

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 1))
        nn.init.normal_(self.text_projection, std=0.02)
        nn.init.normal_(self.img_projection, std=0.02)

    def _initialize_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, text_feat, img_feat, smiles_graph_feat):
        img_feat = self.fc(img_feat)
        smiles_graph_feat = self.fc4(smiles_graph_feat)  # （128,48）映射到（128,128）用于矩阵乘法计算
        # 将文本特征和图像特征归一化

        img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        smiles_graph_feat = smiles_graph_feat / smiles_graph_feat.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_feat @ text_feat.T
        logits_per_text = logit_scale * text_feat @ img_feat.T
        logits_per_smiles = logit_scale * smiles_graph_feat @ img_feat.T

        labels = torch.arange(text_feat.shape[0], device=self.device)
        loss = (F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels) +
                F.cross_entropy(logits_per_smiles, labels)
                ) / 3

        # 对每个特征应用权重
        # text_feat_weighted = self.alpha1 * text_feat
        # img_feat_weighted = self.alpha2 * img_feat
        # smiles_graph_feat_weighted = self.alpha3 * smiles_graph_feat

        # 使用加权后的特征进行融合
        # combined_vec = torch.cat((text_feat_weighted, img_feat_weighted, smiles_graph_feat_weighted), dim=1)

        # 融合特征
        combined_vec = torch.cat((text_feat, img_feat), dim=1)

        # 调整维度以适应多头注意力机制的输入
        combined_vec = combined_vec.unsqueeze(0)  # [1, batch_size, embed_dim]
        attn_output, _ = self.attention(combined_vec, combined_vec, combined_vec)
        # 去除多余的维度
        attn_output = attn_output.squeeze(0)  # [batch_size, embed_dim]

        # 加入残差连接，将原始特征加回到注意力输出中
        attn_output = attn_output + combined_vec.squeeze(0)  # 残差连接

        x = self.mlp(attn_output)
        x1 = self.mlp1(attn_output)
        x2 = self.mlp2(attn_output)
        x3 = self.mlp3(attn_output)
        x4 = self.mlp4(attn_output)

        return loss, x, x1, x2, x3, x4

def main(args):




    char_list = [
        '[nH]', 'F', 'n', '=', '#', '[O]', 'o', 's', '[NH2+]', ')', 'c', '1', '3',
        '[n+]', 'S', '[NH3+]', '[N+]', 'C', 'L', '(', 'R', '5', 'N', '-', '[S+]',
        'O', '7', '[n-]', '6', '[N-]', '8', '4', '[O-]', '[o+]', '2', 'Br', 'Cl'
    ]
    char_to_id = {ch: idx + 1 for idx, ch in enumerate(char_list)}
    char_to_id['<pad>'] = 0

    data_path = './data/pretrain_smiles_train.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    batch_size = 128
    dataset = SMILESImageDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 创建对比学习模型、损失函数和优化器
    text_dim = 128
    img_dim = 128
    embedding_dim = 512
    contrastive_model = SimpleCLIP(text_dim, img_dim, embedding_dim, device).to(device)

    optimizer = torch.optim.Adam(
        contrastive_model.parameters(),
        lr=0.01,
        weight_decay=10 ** -5
    )

    # 训练模型
    num_epochs = 20
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        contrastive_model.train()
        epoch_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:  # tqdm 包裹 dataloader
            for smiles_batch, img_batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

                # 将数据移动到设备
                img_batch = img_batch.to(device)

                # 通过解码获得SMILES向量特征
                smiles_vec = load_model_and_decode(smiles_batch).to(device)  # (batch_size, 128)
                # 提取图像特征
                img_vec = extract_image_features(img_batch).to(device)  # (batch_size, 512)
                # 图数据特征
                smiles_graph_vec = load_model_smiles_to_graph(smiles_batch, device=device).to(device)

                optimizer.zero_grad()
                loss = contrastive_model(smiles_vec, img_vec, smiles_graph_vec)[0]

                loss.backward()
                optimizer.step()  # 更新学习率

                # 获取当前的学习率
                current_lr = optimizer.param_groups[0]['lr']

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=epoch_loss / len(tepoch))

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Current LR: {current_lr:.6f}, Loss: {avg_epoch_loss}')

        # 保存损失最低的模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = contrastive_model.state_dict()

    # 保存对比学习模型
    best_model_path = f'best_contrastive_model_{args.datasetname}.ckpt'
    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        print(f"Best contrastive model saved as best_contrastive_model_{args.datasetname}.ckpt")

if __name__ == "__main__":
    args = parse_args()

    main(args)
