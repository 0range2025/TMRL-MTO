import time

import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.pscan import pscan
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import re

"""
HXJ: 
(B=batch size, L=seq len, D=model dim)
"""


@dataclass
class MambaConfig:
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  #  use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        #  x : (B, L, D)

        #  y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        #  x : (B, L, D)

        #  output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        #  projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output

    def ssm(self, x, z):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  #  (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    #  -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


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

# 定义Mamba模型类
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


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义自定义数据集类
class SMILESDataset(Dataset):
    def __init__(self, smiles, char_to_id, max_len):
        self.smiles = smiles
        self.char_to_id = char_to_id
        self.max_len = max_len
        self.input_ids = self.smiles_to_ids(smiles, char_to_id, max_len)

    def smiles_to_ids(self, smiles, char_to_id, max_len):
        def tokenize(smile):
            # 使用正则表达式拆分SMILES字符串
            regex = r'(\[[^\[\]]*\]|Br|Cl|[a-z]|\d|[A-Z]|[^a-zA-Z0-9])'
            return re.findall(regex, smile)

        tokenized_smiles = [tokenize(s) for s in smiles]
        return [[char_to_id[ch] for ch in s] + [char_to_id['<pad>']] * (max_len - len(s)) for s in tokenized_smiles]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx])


# 自定义字符字典
char_list = [
    '[nH]', 'F', 'n', '=', '#', '[O]', 'o', 's', '[NH2+]', ')', 'c', '1', '3',
    '[n+]', 'S', '[NH3+]', '[N+]', 'C', 'L', '(', 'R', '5', 'N', '-', '[S+]',
    'O', '7', '[n-]', '6', '[N-]', '8', '4', '[O-]', '[o+]', '2', 'Br', 'Cl'
]
char_to_id = {ch: idx + 1 for idx, ch in enumerate(char_list)}
char_to_id['<pad>'] = 0  # 添加填充符

# 读取mols.txt文件中的SMILES字符串
with open('mols.txt', 'r') as file:
    smiles = file.readlines()
smiles = [s.strip() for s in smiles]

# 最大序列长度
max_len = max(len(re.findall(r'(\[[^\[\]]*\]|Br|Cl|[a-z]|\d|[A-Z]|[^a-zA-Z0-9])', s)) for s in smiles)

# 创建数据集和数据加载器
dataset = SMILESDataset(smiles, char_to_id, max_len)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# 模型参数
input_dim = len(char_to_id)
hidden_dim = 128
output_dim = len(char_to_id)
n_layers = 3

# 定义Mamba配置
mamba_config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)

# 实例化Mamba模型
model = MambaModel(input_dim, hidden_dim, output_dim, mamba_config).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()  # 记录开始时间
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, output_dim), batch.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tepoch.set_postfix(loss=epoch_loss / len(tepoch))
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')


# 保存模型
torch.save(model.state_dict(), "smiles_mamba.ckpt")
print("Model saved as smiles_mamba.ckpt")
end_time = time.time()  # 记录结束时间
training_time = end_time - start_time
print(f"Training took {training_time} seconds.")

# 加载模型进行解码
def load_model_and_decode(smiles_list):
    # 实例化模型
    model = MambaModel(input_dim, hidden_dim, output_dim, mamba_config).to(device)
    model.load_state_dict(torch.load("smiles_mamba.ckpt"))
    model.eval()

    def tokenize(smile):
        regex = r'(\[[^\[\]]*\]|Br|Cl|[a-z]|\d|[A-Z]|[^a-zA-Z0-9])'
        return re.findall(regex, smile)

    # 将输入的SMILES转化为ID
    tokenized_smiles = [tokenize(s) for s in smiles_list]
    input_ids = [[char_to_id.get(ch, char_to_id['<pad>']) for ch in s] + [char_to_id['<pad>']] * (max_len - len(s)) for
                 s in tokenized_smiles]
    input_tensor = torch.tensor(input_ids).to(device)

    # with torch.no_grad():
    #     encoded_vectors = model.encode(input_tensor)
    #     return encoded_vectors
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, dim=2)
        decoded_smiles = [''.join([list(char_to_id.keys())[list(char_to_id.values()).index(i)] for i in seq if i > 0])
                          for seq in predicted.tolist()]
        return decoded_smiles


# 示例：解码指定的SMILES
test_smiles = ["O=c1oc2ccc(Br)cc2cc1-c1cc(-c2ccc(Cl)cc2Cl)nc(NCN2CCOCC2)n1", "COCCN(CC1CC1C)c1cc(-c2nnc(C(C)(N)Cc3ccccc3Cl)o2)c(Cl)c(N(C)S(C)(=O)=O)n1"]
decoded_smiles = load_model_and_decode(test_smiles)
print("Decoded SMILES:", decoded_smiles)

