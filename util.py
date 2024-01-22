# -*- coding: utf-8 -*-

import os
import csv
import math
import logging
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as tg
from sklearn.manifold import TSNE
from termcolor import colored
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score

from torch.optim.lr_scheduler import _LRScheduler


# -----------------------------------------------------------------------------
# Calculation utils
# -----------------------------------------------------------------------------
def update_mean(mean: float, data: float, step: int):
    return (mean * step + data) / (step + 1)


def pairwise_distances(x, y, p=2):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return (x - y).abs().pow(p).sum(-1)  # square of distances
    # return (x - y).abs().pow(p).sum(-1).pow(1 / p)


def standard_scaler(x):
    """
    Args:
        x: 2-D matrix
    Returns:
        x: standardized input matrix
        mu: mean of dim 0
        std: std of dim 0
    """
    mu = x.mean(dim=0)
    std = x.std(dim=0)
    eps = 1e-12

    return (x - mu) / (std + eps), mu, std


def standard_scaler2(x):
    mu = x.mean(dim=1).unsqueeze(1)
    std = x.std(dim=1).unsqueeze(1)
    eps = 1e-12
    return (x - mu) / (std + eps), mu, std
    # x_max = x.max(dim=1)[0].unsqueeze(1)
    # x_min = x.min(dim=1)[0].unsqueeze(1)
    # eps = 1e-15
    # return (x - x_min) / (x_max - x_min + eps), 0, 0


def pairwise_cos_sim(a, b):
    """
    Args:
        a: 2d
        b: 2d
    """
    eps = 1e-12
    sim_matrix = torch.einsum('ik, jk->ij', a, b) / torch.einsum('i, j->ij', (a + eps).norm(dim=1),
                                                                 (b + eps).norm(dim=1))
    return sim_matrix


def batched_pairwise_cos_sim(a, b):
    """
    Args:
        a: 3d
        b: 3d
    """
    eps = 1e-12
    sim_matrix = torch.einsum('bik, bjk->bij', a, b) / torch.einsum('bi, bj->bij', (a + eps).norm(dim=-1),
                                                                    (b + eps).norm(dim=-1))
    return sim_matrix


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out


def sinkhorn_knopp(out, n_iter=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    Q /= sum_Q
    for _ in range(n_iter):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


# -----------------------------------------------------------------------------
# Metric utils
# -----------------------------------------------------------------------------
def evaluate_cluster(X, y):
    sscore = silhouette_score(X.detach().cpu().numpy(), y.detach().cpu().numpy())
    chi = calinski_harabasz_score(X.detach().cpu().numpy(), y.detach().cpu().numpy())
    dbi = davies_bouldin_score(X.detach().cpu().numpy(), y.detach().cpu().numpy())
    return sscore, chi, dbi


def evaluate_mod(A, W_norm):
    D = torch.sum(A, dim=2).unsqueeze(dim=2)  # (batch, n_nodes, 1): degree matrix
    m = torch.sum(D, dim=1).unsqueeze(dim=1)  # (batch, 1, 1): 2 * sum of degrees
    B = A - torch.bmm(D, D.permute(0, 2, 1)) / m  # (batch, n_nodes, n_nodes)

    Q = torch.bmm(torch.bmm(W_norm.permute(0, 2, 1), B), W_norm)  # (batch, n_comm, n_comm)
    diag = torch.eye(Q.shape[-1]).type_as(Q)
    Q = torch.sum(torch.sum(Q * diag, dim=2), dim=1) / m.squeeze(dim=1)  # (batch, 1)

    return torch.mean(Q)


def eval_acc(eval_dict):
    """
    Args:
        eval_dict: dict of 'y_true' and 'y_pred'
    Returns:
        dict of accuracy
    """
    y_pred = eval_dict['y_pred']
    y_true = eval_dict['y_true']

    # multiple classification
    y_pred = y_pred.argmax(dim=1)
    acc = (y_pred == y_true).float().sum().item()
    acc = acc / len(y_true)
    return {'acc': acc}


def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets, preds):
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets, preds):
    return mean_squared_error(targets, preds)


def get_metric_func(metric):
    if metric == 'rocauc':
        return roc_auc_score

    if metric == 'prc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    raise ValueError(f'Metric "{metric}" not supported.')


# -----------------------------------------------------------------------------
# Data utils
# -----------------------------------------------------------------------------
def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path, use_compound_names=False):
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


# -----------------------------------------------------------------------------
# Set global seed
# -----------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def get_logger(filename, verbosity=1, name=None):
    while len(logging.root.handlers) > 0:
        logging.root.handlers.pop()

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def create_logger(args):
    # pop redundant loggers
    while len(logging.root.handlers) > 0:
        logging.root.handlers.pop()

    # log name
    time_str = time.strftime("%Y-%m-%d")
    log_name = "{}_{}.log".format(args.test_name, time_str)

    # log dir
    log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = \
        colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d): ', 'yellow') + \
        colored('%(levelname)-5s', 'magenta') + ' %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Model resuming & checkpoint loading and saving.
# -----------------------------------------------------------------------------
def load_checkpoint(cfg, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {cfg.TRAIN.RESUME}....................")

    checkpoint = torch.load(cfg.TRAIN.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    best_epoch, best_auc = 0, 0.0
    if not cfg.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.defrost()
        cfg.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        cfg.freeze()
        logger.info(f"=> loaded successfully '{cfg.TRAIN.RESUME}' (epoch {checkpoint['epoch']})")
        if 'best_auc' in checkpoint:
            best_auc = checkpoint['best_auc']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']

    del checkpoint
    torch.cuda.empty_cache()
    return best_epoch, best_auc


def save_best_checkpoint(args, epoch, model, best_auc, best_epoch, optimizer, scheduler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'best_auc': best_auc,
                  'best_epoch': best_epoch,
                  'epoch': epoch,
                  }

    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, f'{args.test_name}_best.pth')
    torch.save(save_state, save_path)


def load_best_result(args, model, logger):
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
    best_ckpt_path = os.path.join(ckpt_dir, f'{args.test_name}_best.pth')
    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Ckpt loading: {best_ckpt_path}')
    best_epoch = ckpt['best_epoch']

    return model, best_epoch


# -----------------------------------------------------------------------------
# Lr_scheduler
# -----------------------------------------------------------------------------
class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                 init_lr, max_lr, final_lr):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):

        return list(self.lr)

    def step(self, current_step=None):

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def tsne(xs, us):
    """
    Args:
        us: shape (n_clusters, dim)
        xs: shape (n_sub_graphs, dim)

    Returns:
        xs_embed: shape (n_clusters, 2)
        us_embed: shape (n_sub_graphs, 2)
    """
    xs_us = np.concatenate((xs, us), axis=0)
    print("Shape of input xs + us: " + str(xs_us.shape))
    # dist_mat = distance_matrix(xs_us, xs_us)
    print("Reducing dimensions...")

    xs_us_embed = TSNE(n_components=2, metric='euclidean', learning_rate='auto', init='pca').fit_transform(xs_us)
    xs_embed = xs_us_embed[:xs.shape[0], :]
    us_embed = xs_us_embed[xs.shape[0]:, :]
    print("Shape of embedded sub graphs: " + str(xs_embed.shape))

    return xs_embed, us_embed


def show_graph(data, A, X, W):
    W_mask = W == 0.
    non_zero_rows = W.abs().sum(dim=1) > 0
    W = W.masked_fill(W_mask, 1e5)
    row_min = W.min(dim=1)[0].unsqueeze(dim=-1)
    W = W.masked_fill(W_mask, -1e5)
    row_max = W.max(dim=1)[0].unsqueeze(dim=-1)
    W = (W - row_min) / (row_max - row_min)
    W = W.masked_fill(W_mask, -1e5)
    W_non_zero = W[non_zero_rows]
    W_non_zero = F.softmax(W_non_zero / 0.1, dim=1)
    W[non_zero_rows] = W_non_zero
    W = W.masked_fill(W_mask, 0)

    # row_min = W.min(dim=1)[0].unsqueeze(dim=-1)
    # row_max = W.max(dim=1)[0].unsqueeze(dim=-1)
    # W = (W - row_min) / (row_max - row_min)
    # W = F.softmax(W / 0.1, dim=1)
    # W = torch.nan_to_num(W, nan=0.)

    print_MUTAG_feats(data.x)
    print(f'y = {data.y.item()}')

    assign = W.argmax(dim=1)

    G = tg.utils.to_networkx(data, to_undirected=True)
    color_map = []
    for idx, node in enumerate(G):
        color_map.append(assign[idx].item())
    nx.draw_networkx(G, with_labels=True, font_weight='bold', node_color=color_map)
    plt.show()
    print(1)
    # plt.savefig('path.png')


def smiles_to_graph(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(150, 150), kekulize=True)
    plt.imshow(img)
    plt.show()
    plt.close()


# ==================================
# Print dataset features
# ==================================
def print_MUTAG_feats(x):
    elements = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    for i, e in enumerate(elements):
        print(f'{e}:', end=' ')
        for j, node in enumerate(x[:, i]):
            if node.item() == 1.:
                print(j, end=' ')
        print()


def print_PTC_feats(x):
    elements = ['In', 'P', 'O', 'N', 'Na', 'C', 'Cl', 'S', 'Br', 'F', 'K', 'Cu', 'Zn', 'I', 'Ba', 'Sn', 'Pb', 'Ca']
    for i, e in enumerate(elements):
        print(f'{e}:', end=' ')
        for j, node in enumerate(x[:, i]):
            if node.item() == 1.:
                print(j, end=' ')
        print()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    """return None when non-exist access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
