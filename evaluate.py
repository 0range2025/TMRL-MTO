import argparse
import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from adaptation_layers import inject_adaptation_to_linear_layer
from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets
from model.cnn_model_utils import load_model, evaluate_on_multitask
from model.train_utils import load_smiles
from utils.public_utils import cal_torch_model_params, setup_device
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test

import torch.nn.functional as F


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

        self.softmax = nn.Softmax(dim=0)

        self.fc = nn.Linear(512, 128)
        self.fc1 = nn.Linear(text_dim + img_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, 1)
        self.fc4 = nn.Linear(48, 128)
        self.fc5 = nn.Linear(128, 48)
        self.attention = PlainMultiHeadAttention(embed_dim=256, num_heads=8).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 2),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 12),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 27),
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(256, 256),
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
        smiles_graph_feat = self.fc4(smiles_graph_feat)

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

        combined_vec = torch.cat((text_feat, img_feat), dim=1)

        combined_vec = combined_vec.unsqueeze(0)  # [1, batch_size, embed_dim]
        attn_output, _ = self.attention(combined_vec, combined_vec, combined_vec)

        attn_output = attn_output.squeeze(0)  # [batch_size, embed_dim]

        attn_output = attn_output + combined_vec.squeeze(0)

        x = self.mlp(attn_output)
        x1 = self.mlp1(attn_output)
        x2 = self.mlp2(attn_output)
        x3 = self.mlp3(attn_output)
        x4 = self.mlp4(attn_output)

        return loss, x, x1, x2, x3, x4, attn_output


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')


    parser.add_argument('--dataset', type=str, default="Human_fibroblast_toxicity",
                        help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # evaluation
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='logs/finetune/Human_fibroblast_toxicity_valid_best.pth', type=str,
                        metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 41) to split dataset')
    parser.add_argument('--runseed', type=int, default=2024, help='random seed to run model (default: 2024)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced'],
                        help='regularization of classification loss')

    return parser.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
    args.verbose = True

    device, device_ids = setup_device(1)

    # architecture name
    if args.verbose:
        print('Architecture: {}'.format(args.image_model))

    ##################################### initialize some parameters #####################################
    if args.task_type == "classification":
        eval_metric = "rocauc"
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ##################################### load data #####################################
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    if args.split == "random":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx, train_smiles, val_smiles, test_smiles = split_train_val_test_idx(
            list(range(0, len(names))), smiles, frac_train=0.01,
            frac_valid=0.01, frac_test=0.98, seed=args.seed)
    elif args.split == "stratified":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx, train_smiles, val_smiles, test_smiles = split_train_val_test_idx_stratified(
            list(range(0, len(names))), smiles, labels,
            frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
    elif args.split == "scaffold":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx, train_smiles, val_smiles, test_smiles = scaffold_split_train_val_test(
            list(range(0, len(names))), smiles, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
    elif args.split == "random_scaffold":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx, train_smiles, val_smiles, test_smiles = random_scaffold_split_train_val_test(
            list(range(0, len(names))), smiles,
            frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
    elif args.split == "scaffold_balanced":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx, train_smiles, val_smiles, test_smiles = scaffold_split_balanced_train_val_test(
            list(range(0, len(names))), smiles,
            frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed,
            balanced=True)

    name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[
        test_idx], labels[train_idx], labels[val_idx], labels[test_idx]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args, test_smiles_list=test_smiles)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### load model #####################################
    # 加载模型状态字典
    best_model_path = f'best_contrastive_model_{args.dataset}.ckpt'
    state_dict = torch.load(best_model_path, map_location=args.device)
    text_dim = 128
    img_dim = 128
    embedding_dim = 512
    model = SimpleCLIP(text_dim, img_dim, embedding_dim, device).to(device)

    # 加载状态字典
    model.load_state_dict(state_dict)


    model = inject_adaptation_to_linear_layer(
        model,
        efficient_finetune='lora',
        lora_r=16,
        lora_alpha=4,
        filter=['q_proj', 'k_proj', 'v_proj', 'fc', 'fc1', 'fc2', 'fc3'],
        module_filter=None,
        extra_trainable_params=['q_proj', 'k_proj', 'v_proj', 'fc', 'fc1', 'fc2', 'fc3']
    )

    if args.resume != 'None':
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")


    data_lora_path = f'lora_checkpoint/finetune_lora_{args.dataset}.pt'
    model.load_state_dict(torch.load(data_lora_path), strict=False)


    model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))

    ##################################### evaluation #####################################
    test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                                    criterion=criterion, device=device, epoch=-1,
                                                                    task_type=args.task_type, return_data_dict=True)
    test_result = test_results[eval_metric.upper()]

    print("[test] {}: {:.1f}%".format(eval_metric, test_result * 100))


if __name__ == "__main__":
    args = parse_args()
    main(args)
