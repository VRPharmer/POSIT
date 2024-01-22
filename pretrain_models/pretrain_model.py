# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch
# from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder

# from encoders import AttentiveFP
# from cluster import get_collapse_loss, sinkhorn_knopp
# from partition import vol_cut_loss, mod_cut_loss, calc_mod, evaluate_clu
from util import pairwise_cos_sim, batched_pairwise_cos_sim, evaluate_mod, evaluate_cluster
from encoders import GNN, AttentiveFP


class ProtoMiner(nn.Module):
    def __init__(self, args):
        super(ProtoMiner, self).__init__()
        self.n_clusters = args.n_clusters
        self.t1 = args.t1
        self.t2 = args.t2
        self.dropout = args.dropout
        self.gamma = args.gamma
        self.use_edge_attr = args.use_edge_attr

        self.encoder = AttentiveFP(args.in_dim, args.hidden_dim, args.hidden_dim, 10, 3, 3, args.dropout)

        n_dim = args.hidden_dim
        self.node_classifier = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.BatchNorm1d(n_dim),
            nn.LeakyReLU(),
            nn.Linear(n_dim, self.n_clusters)
        )

        # random initialize centers
        # non-trainable
        # self.U = torch.empty(self.n_clusters, 2 * hidden_dim + args.in_dim)
        self.U = torch.empty(self.n_clusters, n_dim)
        # trainable
        # self.U = nn.Parameter(torch.empty(self.n_clusters, 2 * hidden_dim + args.in_dim))

        self.proj_head = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.BatchNorm1d(n_dim),
            nn.LeakyReLU(),
            nn.Linear(n_dim, n_dim)
        )

        self.init_weights()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.use_edge_attr else None
        z, g = self.encoder(x, edge_index, batch, edge_attr)

        s = self.node_classifier(z)
        s = F.softmax(s / self.t1, dim=-1)

        s_hat = sim = 0.

        z_dense, _ = to_dense_batch(z, batch)
        h_dense, p_hat, sub_mask = self.represent_subgraph(z_dense, s, batch, edge_index)
        self.infer_centroids(h_dense, sub_mask)  # todo: validate

        h = h_dense[sub_mask]  # (J, d)

        p = pairwise_cos_sim(h, self.U)
        gc = lc = 0.

        return s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask

    def represent_subgraph(self, z_dense, s_hat, batch, edge_index=None):
        s_hat_dense, mask = to_dense_batch(s_hat, batch)
        sub_filter = s_hat_dense > 1 * s_hat_dense.std(dim=-1, keepdim=True) + 1 / self.n_clusters
        s_hat_dense = s_hat_dense * sub_filter.float()

        p_hat = torch.sum(s_hat_dense, dim=1)
        p_hat = p_hat / (p_hat + 1e-12)
        p_hat = torch.diag_embed(p_hat)

        # mean pooling
        pooling_weights = s_hat_dense.transpose(1, 2)
        pooling_weights = F.normalize(pooling_weights, p=1, dim=-1)
        h_dense = torch.bmm(pooling_weights, z_dense)

        sub_mask = h_dense.abs().sum(dim=-1) > 0.
        p_hat = p_hat[sub_mask]

        return h_dense, p_hat, sub_mask

    def infer_centroids(self, h_dense, mask):
        h_dense = h_dense.permute(1, 0, 2)
        mask = mask.view(h_dense.size(0), h_dense.size(1))
        sizes = torch.sum(mask.float(), dim=-1).unsqueeze(-1)

        centroids = torch.sum(h_dense, dim=1)
        centroids = centroids / (sizes + 1e-10)
        centroids = centroids.detach()
        centroids = self.gamma * self.U.type_as(h_dense) + (1 - self.gamma) * centroids
        self.U = centroids

    @staticmethod
    def evaluate(edge_index, batch, s, h, labels):
        S, mask = to_dense_batch(s, batch)
        A = to_dense_adj(edge_index, batch)
        select = A.sum(dim=2).sum(dim=1) > 0.
        A = A[select]
        S = S[select]
        mod = evaluate_mod(A, S)
        sscore, chi, dbi = evaluate_cluster(h, labels)
        return mod, sscore, chi, dbi

    def init_weights(self):
        nn.init.uniform_(self.U, -1.5, 1.5)


# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(MLP, self).__init__()
#         self.fcs = nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.PReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.PReLU(),
#             nn.Linear(out_dim, out_dim),
#             nn.PReLU()
#         )
#         self.linear_shortcut = nn.Linear(in_dim, out_dim)
#
#     def forward(self, x):
#         return self.fcs(x) + self.linear_shortcut(x)
#
#
# class Encoder2(nn.Module):
#     def __init__(self, in_dim, hidden_dim, dropout=0.2):
#         super(Encoder2, self).__init__()
#         self.transform = nn.Sequential(
#             nn.Linear(3 * in_dim, hidden_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(hidden_dim, 2 * hidden_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#
#     def forward(self, x, edge_index, batch, edge_attr=None):
#         # todo: weights
#         # weights = [1, 0.75, 0.5]
#         weights = [1, 0.5]
#         X, mask = to_dense_batch(x, batch)  # (batch, n, d)
#         X = X.float()
#         z_list = [X]
#         A = to_dense_adj(edge_index, batch)  # (batch, n, n)
#         A2 = A.bmm(A.transpose(1, 2))
#         A2 = A2 - A2 * torch.eye(A2.shape[-1]).type_as(A2)
#         # A3 = A2.bmm(A.transpose(1, 2))
#         # A3 = A3 - A3 * torch.eye(A3.shape[-1]).type_as(A3)
#         z1 = A.bmm(X) * weights[0]
#         z_list.append(z1)
#         z2 = A2.bmm(X) * weights[1]
#         z_list.append(z2)
#         # z3 = A3.bmm(X) * weights[2]
#         # z_list.append(z3)
#
#         z_list = torch.cat(z_list, dim=2)  # (batch, n, 3d)
#         z_embed = self.transform(z_list)  # (batch, n, d)
#         z_embed = z_embed[mask]
#         return z_embed
#
#
# class PPRMiner(nn.Module):
#     def __init__(self, args):
#         super(PPRMiner, self).__init__()
#         self.t1 = args.t1
#         self.encoder = Encoder(args.in_dim, args.hidden_dim, 2)  # GIN
#         hidden_dim = args.hidden_dim
#         self.proj_head = nn.Sequential(
#             nn.Linear(args.in_dim + 2 * hidden_dim, 2 * hidden_dim),
#             nn.BatchNorm1d(2 * hidden_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(2 * hidden_dim, 2 * hidden_dim),
#         )
#
#     def forward(self, x, edge_index, batch, edge_attr=None):
#         z = self.encoder(x, edge_index, batch, edge_attr)  # (N, d)
#
#         h = self.proj_head(z)  # (N, d')
#         h_dense, mask = to_dense_batch(h, batch)  # (batch, n, d')
#
#         p = batched_pairwise_cos_sim(h_dense, h_dense)  # (batch, n, n)
#
#         return z, h, p
#
#     def contrastive_loss(self, p, w, batch):
#         mx = torch.max(p, dim=-1)[0]  # prevent overflow: (batch, n)
#
#         w, mask = to_dense_batch(w, batch)
#         w = w[:, :, :p.size(-1)]
#         # todo: where to * w, order of sum & exp
#         # w = p * w  # (batch, n, n)
#         # w = w.sum(dim=-1)  # (batch, n)
#         # w = torch.exp((w - mx) / self.t1)
#
#         p = torch.exp((p - mx.unsqueeze(-1)) / self.t1)  # (batch, n, n)
#
#         w = p * w
#         w = w.sum(dim=-1)  # (batch, n)
#
#         p = p.sum(dim=-1)  # (batch, n)
#
#         # todo: del self
#         p = (w / p)[mask]
#         non_zero = p != 0
#         p = p[non_zero]
#         loss = -torch.log(p)  # (batch, n)
#         loss = loss.mean()
#
#         return loss
