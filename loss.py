# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn.functional as F
from util import sinkhorn_knopp, pairwise_cos_sim, batched_pairwise_cos_sim
from torch_geometric.utils import to_dense_batch, to_dense_adj


def target_distribution(q):
    p = torch.pow(q, 2) / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1).unsqueeze(-1)
    return p


def get_cluster_loss(q):
    p = target_distribution(q.detach())
    kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
    return kl_loss


def get_compact_loss(sim_matrix, labels):
    # return torch.zeros(1).type_as(sim_matrix)
    positives = sim_matrix[labels.bool()].view(labels.size(0), -1)
    logits = 1 - positives
    loss = logits.pow(2).mean()
    return loss


def get_assign_loss(self, q, q_hat=None):
    if q_hat is None:
        q_hat = sinkhorn_knopp(q)
        q_hat = F.one_hot(q_hat.argmax(dim=1), num_classes=self.n_clusters)

    loss = -(q_hat * torch.log(q + 1e-10)).sum(dim=1).mean()
    return loss


# DMoN loss
def get_collapse_loss(I_norm):
    cluster_sizes = I_norm.sum(dim=0)  # (n_cluster)
    n_nodes = I_norm.size(0)
    loss = (
            torch.norm(cluster_sizes, dim=-1)
            / n_nodes
            * (I_norm.size(-1) ** 0.5)
            - 1
    )
    return loss


def get_entro_loss(s):
    loss = (-s * torch.log(s + 1e-12)).sum(dim=-1).mean()
    return loss


# todo: del select
def get_cut_loss(edge_index, batch, s):
    S, _ = to_dense_batch(s, batch)
    A = to_dense_adj(edge_index, batch)  # (batch, n, n)
    select = A.sum(dim=2).sum(dim=1) > 0.
    A = A[select]
    S = S[select]
    M = S.transpose(1, 2).bmm(A).bmm(S)  # (batch, K, K)
    loss = mod_cut(S, M, A)
    # loss = vol_cut_loss(S, M, A)
    return loss


def get_ortho_loss(I_norm):
    I_o = I_norm.t().matmul(I_norm)  # (K, K)
    I_o = I_o / torch.norm(I_o)
    I_eye = torch.eye(I_o.shape[0]).type_as(I_o)
    I_eye = I_eye / torch.norm(I_eye)
    orth_loss = torch.norm(I_o - I_eye)
    return orth_loss


def mod_cut(W_norm, M, A):
    d = A.sum(dim=2).unsqueeze(dim=2)  # (batch, n, 1) degree vector
    m = d.sum(dim=1)  # (batch, 1): 2 * sum of degrees
    Q = M - W_norm.permute(0, 2, 1).bmm(d).bmm(d.permute(0, 2, 1)).bmm(W_norm) / m.unsqueeze(dim=1)
    Q = Q.diagonal(dim1=-2, dim2=-1)  # (batch, k)
    Q = Q.sum(dim=1) / m
    return -Q.mean()


def vol_cut(W_norm, M, A):
    D = A.sum(dim=2).unsqueeze(dim=2)  # (batch, n, 1) degree matrix

    vol = (W_norm * D).sum(dim=1).unsqueeze(dim=2)  # (batch, k, 1)
    eps = 1e-12

    diag = torch.eye(M.shape[-1]).type_as(M)
    loss = ((M - M * diag) / (vol + eps)).sum(dim=2).sum(dim=1).mean()
    return loss


def get_contrastive_loss(sim_matrix, labels, t2):
    # return torch.zeros(1).type_as(sim_matrix)
    positives = sim_matrix[labels.bool()].view(labels.size(0), -1)
    negatives = sim_matrix[~labels.bool()].view(labels.size(0), -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / t2
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=sim_matrix.device)

    contra_loss = F.cross_entropy(logits, labels)
    return contra_loss


class FocalLoss(torch.nn.Module):
    # For imbalanced data.
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        target = target.float()
        pt = torch.softmax(inputs, dim=1)
        p = pt[:, 1]
        eps = 1e-12
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p + eps)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p + eps))
        return loss.mean()


def get_sup_loss(args, criterion, output, y):
    loss = 0.
    # y_pred_list = {}
    # y_label_list = {}

    for i in range(len(args.task_name)):
        if args.task_type == 'classification':
            y_pred = output[:, i * 2:(i + 1) * 2]
            y_label = y[:, i].squeeze()
            validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]

            if len(validId) == 0:
                continue
            if y_label.dim() == 0:
                y_label = y_label.unsqueeze(0)

            y_pred = y_pred[torch.tensor(validId).to(args.device)]
            y_label = y_label[torch.tensor(validId).to(args.device)]

            loss += criterion[i](y_pred, y_label)
            # y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
        else:
            y_pred = output[:, i]
            y_label = y[:, i].float()
            loss += criterion(y_pred, y_label)
            # y_pred = y_pred.detach().cpu().numpy()

        # try:
        #     y_label_list[i].extend(y_label.cpu().numpy())
        #     y_pred_list[i].extend(y_pred)
        # except:
        #     y_label_list[i] = []
        #     y_pred_list[i] = []
        #     y_label_list[i].extend(y_label.cpu().numpy())
        #     y_pred_list[i].extend(y_pred)

    return loss


# ---------------------------------
# JS MI estimator
# ---------------------------------
def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq


# from MVGRL: https://github.com/mtang724/mvgrl/blob/main/graph/utils.py
def local_global_loss(l_enc, g_enc, h_dense, K, sub_mask):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    device = g_enc.device

    batch_size = h_dense.size(0)
    pos_mask = torch.arange(0, batch_size, dtype=torch.long, device=device)
    pos_mask = pos_mask.expand(K, batch_size).t()  # (batch, K)
    pos_mask = pos_mask[sub_mask]  # (J)
    pos_mask = F.one_hot(pos_mask, num_classes=batch_size)  # (J, K)
    neg_mask = 1 - pos_mask

    # pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    # neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    # for nodeidx, graphidx in enumerate(graph_id):
    #     pos_mask[nodeidx][graphidx] = 1.
    #     neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


# ---------------------------------
# sub-sub contrast
# ---------------------------------
def local_local_loss(s, h_dense, sub_mask, edge_index, batch, t):
    h = h_dense[sub_mask]
    # h_dense = h_dense.view(-1, h_dense.size(-1))
    s_dense, _ = to_dense_batch(s, batch)
    batch_size = s_dense.size(0)

    # 1 2 3 4 5 6  7  8  9
    # 1 2 4 5 8 9 10 11 12
    # 0 0 1 1 3 3  3  3  3

    A = to_dense_adj(edge_index, batch)  # (batch, n, n)
    AK = s_dense.transpose(1, 2).bmm(A).bmm(s_dense)  # (batch, Ki, Ki)
    AK = AK.view(-1, AK.size(-1))  # (batch * Ki, Ki)
    # add = torch.arange(0, batch_size, dtype=torch.long, device=s.device)  # (batch)
    # add = add.expand(AK.size(-1), batch_size).t().reshape(-1)  # (batch * K)

    AK = torch.argmax(AK, dim=-1)  # (batch * Ki)
    h_dense = h_dense.view(-1, h_dense.size(-1))  # (batch * Ki, d)
    pos = torch.index_select(h_dense, 0, AK)[sub_mask.view(-1)]  # (J, d)
    pos = torch.cosine_similarity(pos, h, dim=-1).unsqueeze(-1)  # (J, 1)

    neg = pairwise_cos_sim(h, h)  # (J, J)

    # pos = reciprocal_weight(pos)
    # neg = reciprocal_weight(neg)

    logits = torch.cat([pos, neg], dim=1)  # (J, J + 1)
    logits = logits / t
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=pos.device)

    loss = F.cross_entropy(logits, labels)

    # AK = F.one_hot(AK, num_classes=AK.size(-1))  # (batch * Ki, Ki)
    # pos = h_dense[AK.view()bool()]

    # a = torch.arange(0, batch_size * AK.size(-1), dtype=torch.long, device=torch.device('cuda'))
    # a = a[sub_mask.view(-1)]  # (J)
    # b = torch.arange(0, a.size(0), dtype=torch.long, device=torch.device('cuda'))  # (J)
    # b = a - b
    # label = torch.argmax(AK, dim=1)  # (batch * Ki)
    # label += add * AK.size(-1)
    # label = label[sub_mask.view(-1)]  # (J)
    # label = F.one_hot(label, num_classes=h.size(0))

    # sim = pairwise_cos_sim(h, h)  # (J, J)
    # sim = sim[sub_mask.view(-1)]
    # sim = sim.t()[sub_mask.view(-1)].t()
    # loss = get_contrastive_loss(sim, label, t)
    return loss


def gaussian_weight(sim):
    tau = 1.5
    rou = -1
    sim = -0.5 * ((sim - (rou + tau ** 2)) / tau) ** 2
    sim = torch.exp(sim)
    return sim


def reciprocal_weight(sim):
    rou = 2
    lam = -1
    return (sim + rou) ** (-lam)
