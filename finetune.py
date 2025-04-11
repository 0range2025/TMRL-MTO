import argparse
import os
from collections import Counter
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from ruamel.yaml import YAML
import torch.nn.functional as F
import numpy as np
from torch import optim
from typing import Dict
from adaptation_layers import inject_adaptation_to_linear_layer
from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets

from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt, \
    get_support_model_names
from model.train_utils import fix_train_random_seed, load_smiles
from utils.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


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

    # basic
    parser.add_argument('--dataset', type=str, default="Human_fibroblast_toxicity",
                        help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # optimizer
    parser.add_argument('--lr', default=0.06, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2024, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="scaffold_balanced", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced'],
                        help='regularization of classification loss')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    # parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--weighted_CE', action='store_true', default=False,
                        help='whether to use global imbalanced weight')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune/', help='path to log')
    parser.add_argument('--device', default='cuda')

    return parser.parse_args()


def main(args):
    global train_smiles, val_smiles, test_smiles, val_loss
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
    args.verbose = True

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    ##################################### initialize some parameters #####################################
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ##################################### load data #####################################
    if args.image_aug:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                                 transforms.ToTensor()]
    else:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    if args.split == "random":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx, train_smiles, val_smiles, test_smiles = split_train_val_test_idx(
            list(range(0, len(names))), smiles, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
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
    train_dataset = ImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train),
                                 normalize=normalize, args=args, train_smiles_list=train_smiles)
    val_dataset = ImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, args=args, val_smiles_list=val_smiles)
    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args, test_smiles_list=test_smiles)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### load model #####################################

    best_model_path = f'best_contrastive_model_{args.dataset}.ckpt'
    state_dict = torch.load(best_model_path, map_location=args.device)
    text_dim = 128
    img_dim = 128
    embedding_dim = 512
    model = SimpleCLIP(text_dim, img_dim, embedding_dim, device).to(device)


    model.load_state_dict(state_dict)


    model = inject_adaptation_to_linear_layer(
        model,
        efficient_finetune='lora',
        lora_r=16,
        lora_alpha=4,
        filter=['q_proj', 'k_proj', 'v_proj', 'fc', 'fc1', 'fc2', 'fc3'],
        module_filter=None,
        extra_trainable_params=['q_proj', 'k_proj', 'v_proj', 'fc', 'fc1', 'fc2', 'fc3'],
        conv_lora_expert_num=8
    )

    for name, param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_A', 'lora_B', 'lora_a', 'lora_b']:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print(model)
    print("params: {}".format(cal_torch_model_params(model)))
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    ##################################### initialize optimizer #####################################
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=10 ** args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.008,
                                                     verbose=True)

    weights = None
    if args.task_type == "classification":
        if args.weighted_CE:
            labels_train_list = labels_train[labels_train != -1].flatten().tolist()
            count_labels_train = Counter(labels_train_list)
            imbalance_weight = {key: 1 - count_labels_train[key] / len(labels_train_list) for key in
                                count_labels_train.keys()}
            weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))

    ##################################### train #####################################
    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value,
               'highest_valid_desc': None,
               "final_train_desc": None,
               "final_test_desc": None}

    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                                  weights=weights, device=device, epoch=epoch, task_type=args.task_type,
                                  num_tasks=num_tasks)
        # evaluate
        train_loss, train_results, train_data_dict = evaluate_on_multitask(model=model, data_loader=train_dataloader,
                                                                           criterion=criterion, device=device,
                                                                           epoch=epoch, num_tasks=num_tasks,
                                                                           task_type=args.task_type,
                                                                           return_data_dict=True)
        val_loss, val_results, val_data_dict = evaluate_on_multitask(model=model, data_loader=val_dataloader,
                                                                     criterion=criterion, device=device,
                                                                     epoch=epoch, num_tasks=num_tasks,
                                                                     task_type=args.task_type,
                                                                     return_data_dict=True)
        test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                                        criterion=criterion, device=device, epoch=epoch,
                                                                        num_tasks=num_tasks, task_type=args.task_type,
                                                                        return_data_dict=True)

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result}, optimizer.param_groups[0]['lr'])

        # 调整学习率
        scheduler.step(val_loss)
        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Learning Rate = {current_lr}")

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            results['highest_valid_desc'] = val_results
            results['final_train_desc'] = train_results
            results['final_test_desc'] = test_results

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(train_loss, 4), epoch, args.log_dir,
                                   f"{args.dataset}_valid_best",
                                   lr_scheduler=None, result_dict=results)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print("final results: highest_valid: {:.3f}, final_train: {:.3f}, final_test: {:.3f}"
          .format(results["highest_valid"], results["final_train"], results["final_test"]))


    data_lora_path = f'lora_checkpoint/finetune_lora_{args.dataset}.pt'
    torch.save(lora_state_dict(model), data_lora_path)


if __name__ == "__main__":
    args = parse_args()

    main(args)
