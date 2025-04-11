import logging
import os
import sys

import numpy as np
import torch
import torchvision
from sklearn import metrics
from torch import nn
from tqdm import tqdm
import re
import dataloader.image_dataloader
from Models.ImageMol import ImageMol
from Models.TransformerModel import TransformerModel
from model.evaluate import metric as utils_evaluate_metric
from model.evaluate import metric_multitask as utils_evaluate_metric_multitask
from model.evaluate import metric_reg as utils_evaluate_metric_reg
from model.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask
from Models.mamba import MambaConfig, Mamba
import math
from smiles_to_graph_deal import load_model_smiles_to_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

char_list = [
    '[nH]', 'F', 'n', '=', '#', '[O]', 'o', 's', '[NH2+]', ')', 'c', '1', '3',
    '[n+]', 'S', '[NH3+]', '[N+]', 'C', 'L', '(', 'R', '5', 'N', '-', '[S+]',
    'O', '7', '[n-]', '6', '[N-]', '8', '4', '[O-]', '[o+]', '2', 'Br', 'Cl'
]
char_to_id = {ch: idx + 1 for idx, ch in enumerate(char_list)}
char_to_id['<pad>'] = 0


def get_max_len(smiles):
    return max(len(re.findall(r'(\[[^\[\]]*\]|Br|Cl|[a-z]|\d|[A-Z]|[^a-zA-Z0-9])', s)) for s in smiles)


class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mamba_config):
        super(MambaModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.mamba = Mamba(mamba_config)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)  # (64,95,128)
        output = self.mamba(src)
        output = self.decoder(output)
        return output

    def encode(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        output = self.mamba(src)
        output = output.mean(dim=1)  # (batch_size, hidden_dim)
        return output


def load_model_and_decode(smiles_list):
    # 模型参数
    input_dim = 38
    hidden_dim = 128
    output_dim = 38
    n_layers = 3
    char_to_id = {'<pad>': 0}  #

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
        encoded_vectors = smiles_model.encode(input_tensor)  # (batch_size, hidden_dim)
        return encoded_vectors


def extract_image_features(images):  # images（128,3,224,224）

    jigsaw_classes = 100 + 1
    label1_classes = 100
    label2_classes = 1000
    label3_classes = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Imagemodel = ImageMol("ResNet18", jigsaw_classes, label1_classes, label2_classes, label3_classes).to(device)

    # 加载 checkpoint 文件
    checkpoint = torch.load("data/ImageMol.pth.tar", map_location=device)
    Imagemodel.load_state_dict(checkpoint['state_dict'])

    Imagemodel.eval()
    with torch.no_grad():
        features = Imagemodel(images)[0].to(device)
    return features


def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    return model


# evaluation for classification
def metric(y_true, y_pred, y_prob):
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    f1 = metrics.f1_score(y_true, y_pred)
    precision_list, recall_list, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall_list, precision_list)
    precision = metrics.precision_score(y_true, y_pred, zero_division=1)
    recall = metrics.recall_score(y_true, y_pred, zero_division=1)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    matthews = metrics.matthews_corrcoef(y_true, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob)
    return {
        "accuracy": acc,
        "ROCAUC": auc,
        "f1": f1,
        "AUPR": aupr,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
        "matthews": matthews,
        "fpr": fpr,  # list
        "tpr": tpr,  # list
        "precision_list": precision_list,
        "recall_list": recall_list
    }


def train_one_epoch_multitask(model, optimizer, data_loader, criterion, weights, device, epoch, task_type, num_tasks):
    '''
    :param model:
    :param optimizer:
    :param data_loader:
    :param criterion:
    :param device:
    :param epoch:
    :param criterion_lambda:
    :return:
    '''
    assert task_type in ["classification", "regression"]

    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels, train_smiles_list = data
        images, labels = images.to(device), labels.to(device)

        sample_num += images.shape[0]

        # 通过解码获得SMILES向量特征
        smiles_vec = load_model_and_decode(train_smiles_list).to(device)  # （128,128）

        # 提取图像特征
        img_vec = extract_image_features(images)  # （128,512）

        # 图卷积特征
        smiles_graph_vec = load_model_smiles_to_graph(train_smiles_list, device=device).to(device)

        pred = model(smiles_vec, img_vec, smiles_graph_vec)[num_tasks]

        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            if weights is None:
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                cls_weights = labels.clone()
                cls_weights_mask = []
                for i, weight in enumerate(weights):
                    cls_weights_mask.append(cls_weights == i)
                for i, cls_weight_mask in enumerate(cls_weights_mask):
                    cls_weights[cls_weight_mask] = weights[i]
                loss = torch.sum(loss_mat * cls_weights) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion(pred.double(), labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, criterion, device, epoch, num_tasks, task_type="classification",
                          return_data_dict=False):
    assert task_type in ["classification", "regression"]

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, all_features = [], [], []

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels, train_smiles_list = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        smiles_vec = load_model_and_decode(train_smiles_list).to(device)  # （128,128）

        img_vec = extract_image_features(images)  # （128,512）

        smiles_graph_vec = load_model_smiles_to_graph(train_smiles_list, device=device).to(device)

        with torch.no_grad():

            pred = model(smiles_vec, img_vec, smiles_graph_vec)[num_tasks]

            labels = labels.view(pred.shape).to(torch.float64)

            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat,
                                       torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(pred.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        y_true.append(labels.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            # print(y_true.shape, y_pred.shape, y_pro.shape)
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro,
                                                                                      num_tasks=y_true.shape[1],
                                                                                      empty=-1), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro,
                                                                                      num_tasks=y_true.shape[1],
                                                                                      empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores,
                                                                                          num_tasks=y_true.shape[
                                                                                              1]), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores,
                                                                                          num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")


def save_finetune_ckpt(model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None,
                       logger=None):
    log = logger if logger is not None else logging
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'loss': loss,
        'result_dict': result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log.info("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    log.info('model has been saved as {}'.format(filename))
