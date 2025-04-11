import torch
import numpy as np
from rdkit import Chem

#对torch.autograd.Variable的包装器
def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):                  #如果输入是NumPy数组，则将其转换为PyTorch张量。
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():                       #如果CUDA可用，将PyTorch张量转换为torch.autograd.Variable并将其移动到GPU上
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)              #如果CUDA不可用，只返回转换为torch.autograd.Variable的PyTorch张量。

#是通过乘以(1 - decrease_by)来减小优化器的学习率
def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:          #对优化器的每个参数组进行迭代。
        param_group['lr'] *= (1 - decrease_by)          #将每个参数组的学习率乘以(1 - decrease_by)，以实现学习率的减小。

#是将来自RNN的输出序列转换为相应的SMILES字符串列表
def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():                      #对RNN的输出序列进行迭代，seqs应该是一个PyTorch张量。
        smiles.append(voc.decode(seq))                  #对每个输出序列调用词汇表的decode方法，将其转换为SMILES字符串，并将其添加到smiles列表中。
    return smiles

#是接受SMILES字符串列表并返回其中有效SMILES字符串的比例
def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:                               #对SMILES字符串列表进行迭代。
        if Chem.MolFromSmiles(smile):                  #使用RDKit库的Chem.MolFromSmiles函数检查SMILES字符串是否有效。
            i += 1                                     #如果SMILES有效，增加计数器i的值。
    return i / len(smiles)                             #返回有效SMILES字符串的比例，即有效SMILES字符串的数量除以总SMILES字符串的数量。

#是找到输入张量arr中的唯一行并返回这些行的索引
def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()                            #将输入张量转换为NumPy数组，以便进行后续操作。
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))       #将NumPy数组转换为连续的数组，并使用view方法将其视图转换为np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))类型的数组。这样可以创建一个适用于np.unique的数组。
    _, idxs = np.unique(arr_, return_index=True)       #使用NumPy的np.unique函数找到唯一行的索引。
    if torch.cuda.is_available():                      #如果CUDA可用，返回已排序的索引，并将其转换为PyTorch张量并移到GPU上。
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))             #如果CUDA不可用，返回已排序的索引，并将其转换为PyTorch张量。
