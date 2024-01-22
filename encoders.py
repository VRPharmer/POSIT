from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dropout_adj, degree
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GATEConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_dim: int,
            dropout: float = 0.0,
            # args
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout
        # self.dropout = args.dropout
        # in_channels = args.in_dim
        # out_channels = args.hidden_dim
        # edge_dim = args.edge_dim

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.global_pool = global_add_pool

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        # g = self.global_pool(out, batch)
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


# -----------------------------------------------------------------------------
# GIN and GCN implementation adapted from
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
# -----------------------------------------------------------------------------
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        Note: additional BatchNorm in the message passing compared with regular GINConv!
        """
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                                       nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_embedding = 0
        else:
            edge_embedding = edge_attr

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# todo: handle GCN embed
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if edge_attr is not None:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = 0

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    GIN or GCN model with edge attributes
    """

    def __init__(self, args):
        super(GNN_node, self).__init__()
        self.num_layers = args.depth
        self.drop_ratio = args.dropout
        self.JK = args.JK
        self.normalization = args.norm
        emb_dim = args.hidden_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embed = nn.Linear(args.in_dim, args.hidden_dim)
        if args.use_edge_attr:
            self.e_embed = nn.Linear(args.edge_dim, args.hidden_dim)

        # List of MLPs
        self.convs = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if args.gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif args.gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))

        self.norms = torch.nn.ModuleList()

        if self.normalization == 'batch':
            for layer in range(self.num_layers):
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif self.normalization == 'layer':
            for layer in range(self.num_layers):
                self.norms.append(torch.nn.LayerNorm(emb_dim))

        if self.JK == 'concat':
            self.proj = nn.Linear(emb_dim * (self.num_layers + 1), emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.x_embed(x))  # (N, hidden_dim)
        if edge_attr is not None:
            edge_attr = F.relu(self.e_embed(edge_attr))  # (N', hidden_dim)

        h_list = [x]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            if self.normalization != 'none':
                h = self.norms[layer](h)

            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)  # remove relu for the last layer
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        node_emb = h
        # Different implementations of Jk-concat
        # todo: GRU JK
        if self.JK == "concat":
            h_concat = torch.cat(h_list, dim=1)
            # todo: recover
            # node_emb = self.proj(h_concat)
            node_emb = F.relu(self.proj(h_concat))
        elif self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_emb


class GNN(torch.nn.Module):
    """
    Wrapper of different GNN models

    """

    def __init__(self, args):
        super(GNN, self).__init__()
        emb_dim = args.hidden_dim

        if args.gnn_type == "gin":
            self.gnn_node = GNN_node(args)
        elif args.gnn_type == "gcn":
            self.gnn_node = GNN_node(args)
        # elif args.model_type == "dgcn":
        #     self.gnn_node = DeeperGCN(args)

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.global_pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.global_pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.global_pool = global_max_pool
        elif args.graph_pooling == 'set2set':
            self.global_pool = Set2Set(emb_dim, processing_steps=3)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, batch, edge_attr = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 3:
            x, edge_index, batch = argv[0], argv[1], argv[2]
            edge_attr = None
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        node_emb = self.gnn_node(x, edge_index, edge_attr)
        graph_emb = self.global_pool(node_emb, batch)

        # return node_emb
        return node_emb, graph_emb


# -----------------------------------------------------------------------------
# Attentive FP
# -----------------------------------------------------------------------------
class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            edge_dim: int,
            num_layers: int,
            num_timesteps: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)  # atom_fc

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        # -----------------------------------------------------------
        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        # -----------------------------------------------------------
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor):
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return x, self.lin2(out)
        # return x

    def jittable(self) -> 'AttentiveFP':
        self.gate_conv = self.gate_conv.jittable()
        self.atom_convs = torch.nn.ModuleList(
            [conv.jittable() for conv in self.atom_convs])
        # -----------------------------------------------------------
        self.mol_conv = self.mol_conv.jittable()
        return self

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')
