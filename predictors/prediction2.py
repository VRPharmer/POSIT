# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import global_add_pool, GATConv

from util import pairwise_cos_sim
from encoders import AttentiveFP


class PredictionProto(nn.Module):
    def __init__(self, pre_model, args):
        super(PredictionProto, self).__init__()

        self.pre_model = pre_model
        self.n_tasks = args.n_tasks
        self.out_dim = args.n_tasks * 2
        self.predict = args.predict
        self.use_edge_attr = args.use_edge_attr
        self.K = pre_model.U.size(0)
        self.d = pre_model.U.size(1)

        if args.predict == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.K, 32),
                nn.BatchNorm1d(32),
                nn.CELU(),
                # nn.LeakyReLU(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.CELU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(16, self.out_dim)
            )

        elif args.predict == 'pair':
            self.classifier = nn.Sequential(
                nn.Linear(self.K * (self.K + 1) // 2, 64),
                # nn.BatchNorm1d(64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )

        elif args.predict == 'dim':
            self.classifier = nn.Sequential(
                nn.Linear(self.K * self.d, self.d),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(128, self.out_dim)
            )

        elif args.predict == 'avg':
            self.classifier1 = nn.Sequential(
                nn.Linear(self.K, 32),
                nn.LayerNorm(32),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=args.dropout),
                nn.Linear(32, self.out_dim)
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(self.K * (self.K + 1) // 2 - self.K, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )

        elif args.predict == 'add' or args.predict == 'only_G':
            self.classifier = nn.Sequential(
                nn.Linear(self.d, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )
            # self.classifier = nn.Linear(self.d, self.out_dim)

        elif args.predict == 'pair_U':
            self.classifier = nn.Sequential(
                nn.Linear(self.K * (self.K + 1) // 2 + self.d, 64),
                # nn.BatchNorm1d(64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )

        elif args.predict == 'vote':
            self.g_classifier = nn.Sequential(
                nn.Linear(self.d, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )
            self.sub_classifier = nn.Sequential(
                nn.Linear(self.d, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )

        elif args.predict == 'K_g':
            self.classifier = nn.Sequential(
                nn.Linear(self.K + self.d, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(64, self.out_dim)
            )

        elif args.predict == 'fuse':
            self.heads = args.heads
            self.Ws = nn.ModuleList()
            self.a = nn.ModuleList()
            for i in range(self.heads):
                self.Ws.append(nn.Linear(self.d, self.d))
                self.a.append(nn.Linear(self.d * 2, 1))

            # self.W6 = nn.Linear(self.d, self.d)
            # self.a = nn.Linear(self.d * 2, 1)

            self.classifier = nn.Sequential(
                nn.Linear(self.d * 2, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(128, self.out_dim)
            )

        elif self.predict == 'inter':
            self.heads = args.heads
            self.Ws = nn.ModuleList()
            self.Is = nn.ModuleList()
            for i in range(self.heads):
                self.Ws.append(nn.Linear(self.d, self.d))
                self.Is.append(nn.Linear(self.d, self.d))

            self.classifier = nn.Sequential(
                nn.Linear(self.d * 2, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(128, self.out_dim)
            )

        elif args.predict == 'encode':
            self.g_encoder = AttentiveFP(46, 256, 256, 10, 3, 3, 0.2)
            self.W6 = nn.Linear(self.d, self.d)
            self.a = nn.Linear(self.d * 2, 1)

            self.classifier = nn.Sequential(
                nn.Linear(self.d * 2, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(128, self.out_dim)
            )

        elif args.predict == 'attentive_fp':
            self.g_encoder = AttentiveFP(46, 256, 256, 10, 3, 3, 0.2)
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(128, self.out_dim)
            )

        elif args.predict == 'sum':
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(128, self.out_dim)
            )

        else:
            raise NotImplementedError(f'Unknown predict type: {args.predict}')

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask = self.pre_model(data)
        s_dense, mask = to_dense_batch(s, batch)

        k = s_dense.sum(dim=1)

        # for aggregation
        if self.predict == 'mlp':
            out = self.classifier(k)

        elif self.predict == 'pair':
            A = to_dense_adj(edge_index, batch)
            AK = s_dense.transpose(1, 2).bmm(A).bmm(s_dense)
            indices = torch.triu_indices(AK.shape[-1], AK.shape[-1], offset=1)
            AK = AK[:, indices[0], indices[1]]
            inp = torch.cat([k, AK], dim=1)
            out = self.classifier(inp)

        elif self.predict == 'dim':
            # 1.
            # U = self.pre_model.U.unsqueeze(0)
            # inp = k.unsqueeze(-1) * U
            # 2.
            inp = torch.einsum('bnk,bnd->bkd', s_dense, z_dense)
            out = self.classifier(inp.view(inp.shape[0], -1))

        elif self.predict == 'avg':
            out1 = self.classifier1(k)
            A = to_dense_adj(edge_index, batch)
            AK = s_dense.transpose(1, 2).bmm(A).bmm(s_dense)
            indices = torch.triu_indices(AK.shape[-1], AK.shape[-1], offset=1)
            AK = AK[:, indices[0], indices[1]]
            out2 = self.classifier2(AK)
            out = 0.5 * (out1 + out2)

        elif self.predict == 'add':
            z = z_dense[mask]
            pool = global_add_pool(z, batch)
            out = self.classifier(pool)

        elif self.predict == 'pair_U':
            pooling_weights = s_dense.transpose(1, 2)
            pooling_weights = F.normalize(pooling_weights, p=1, dim=-1)
            h_dense = torch.bmm(pooling_weights, z_dense)
            sim = pairwise_cos_sim(h_dense.view(-1, h_dense.size(-1)), self.pre_model.U)
            sim = sim.view(h_dense.size(0), -1, sim.size(-1))
            K = sim.sum(dim=1)

            A = to_dense_adj(edge_index, batch)
            AK = s_dense.transpose(1, 2).bmm(A).bmm(s_dense)
            AK = sim.transpose(1, 2).bmm(AK).bmm(sim)
            indices = torch.triu_indices(AK.shape[-1], AK.shape[-1], offset=1)
            AK = AK[:, indices[0], indices[1]]
            inp = torch.cat([K, AK, g], dim=1)
            out = self.classifier(inp)

        elif self.predict == 'only_G':
            out = self.classifier(g)

        elif self.predict == 'vote':
            out = s_dense.transpose(1, 2).bmm(z_dense)
            out = self.sub_classifier(out)
            out = out.sum(dim=1)
            out2 = self.g_classifier(g)
            out = 0.5 * (out + out2)

        elif self.predict == 'K_g':
            pooling_weights = s_dense.transpose(1, 2)
            pooling_weights = F.normalize(pooling_weights, p=1, dim=-1)
            h_dense = torch.bmm(pooling_weights, z_dense)
            sim = pairwise_cos_sim(h_dense.view(-1, h_dense.size(-1)), self.pre_model.U)
            sim = sim.view(h_dense.size(0), -1, sim.size(-1))
            K = sim.sum(dim=1)
            inp = torch.cat([K, g], dim=1)
            out = self.classifier(inp)

        elif self.predict == 'fuse':
            batch_size = h_dense.size(0)
            s_Gs = torch.zeros(batch_size, h_dense.size(-1)).type_as(h_dense)
            for i in range(self.heads):
                s_t = self.Ws[i](h_dense)
                h_G = self.Ws[i](g).unsqueeze(1).expand_as(s_t)
                s_G = torch.cat([h_G, s_t], dim=2)
                att = self.a[i](s_G).squeeze(-1)
                att = F.leaky_relu(att)
                att[~sub_mask] = -9999
                att = F.softmax(att, dim=-1).unsqueeze(-1)
                s_G = s_t * att
                s_G = s_G.sum(dim=1)
                s_Gs += s_G

            s_Gs /= self.heads

            g = torch.cat([s_Gs, g], dim=-1)
            out = self.classifier(g)

        elif self.predict == 'inter':
            batch_size = h_dense.size(0)
            s_Gs = torch.zeros(batch_size, h_dense.size(-1)).type_as(h_dense)
            for i in range(self.heads):
                s_t = self.Ws[i](h_dense)
                s_t = F.leaky_relu(s_t)
                h_G = F.leaky_relu(g)

                att = self.Is[i](h_G).unsqueeze(1).bmm(s_t.transpose(1, 2)).squeeze(1)
                att = F.leaky_relu(att)
                att[~sub_mask] = -9999
                att = F.softmax(att, dim=-1).unsqueeze(-1)
                s_G = s_t * att
                s_G = s_G.sum(dim=1)
                s_Gs += s_G

            s_Gs /= self.heads

            g = torch.cat([s_Gs, g], dim=-1)
            out = self.classifier(g)

        elif self.predict == 'encode':
            edge_attr = data.edge_attr
            _, glob = self.g_encoder(x, edge_index, batch, edge_attr)

            batch_size = h_dense.size(0)
            h_dense = h_dense.view(-1, h_dense.size(-1))
            s_t = self.W6(h_dense)
            s_t = s_t.view(batch_size, -1, s_t.size(-1))
            h_G = self.W6(glob).unsqueeze(1).expand_as(s_t)
            s_G = torch.cat([h_G, s_t], dim=2)

            att = self.a(s_G).squeeze(-1)
            att = F.leaky_relu(att)
            att[~sub_mask] = -9999
            att = F.softmax(att, dim=-1).unsqueeze(-1)
            s_G = s_t * att
            s_G = s_G.sum(dim=1)

            glob = torch.cat([s_G, glob], dim=-1)
            out = self.classifier(glob)

        elif self.predict == 'attentive_fp':
            _, g = self.g_encoder(x, edge_index, batch, data.edge_attr)
            out = self.classifier(g)

        elif self.predict == 'sum':
            out = self.classifier(g)

        else:
            raise Exception('wrong value of self.predict')

        return out, s, s_hat, p, p_hat, h_dense, gc, lc, sub_mask, h


class PredictionGIN(nn.Module):
    def __init__(self, args):
        super(PredictionGIN, self).__init__()
        self.n_tasks = args.n_tasks
        from pretrain_models.pretrain_model3 import Encoder
        self.encoder = Encoder(46, 128, 2)

        self.classifier = nn.Sequential(
            nn.Linear(302, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(64, 2 * self.n_tasks)
        )

        # criterion
        if args.focal_loss and args.weights is not None:
            self.criterion = [FocalLoss(alpha=1 / w[0]) for w in args.weights]
        elif args.weights is not None:
            self.criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).cuda(), reduction='mean') for
                              w in args.weights]
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch, edge_attr=None):
        z = self.encoder(x, edge_index, batch, edge_attr)
        pool = global_add_pool(z, batch)
        out = self.classifier(pool)

        return out

    def get_loss(self, out, y):
        sup_loss = 0.
        for i in range(self.n_tasks):
            y_pred = out[:, i * 2:(i + 1) * 2]
            y_label = y[:, i].squeeze()
            valid_id = ~torch.isnan(y_label)
            y_pred = y_pred[valid_id]
            y_label = y_label[valid_id]
            sup_loss += self.criterion[i](y_pred, y_label)
        return sup_loss


class PredictionPPR(nn.Module):
    def __init__(self, pre_model, args):
        super(PredictionPPR, self).__init__()
        self.pre_model = pre_model
        self.n_tasks = args.n_tasks

        self.classifier = nn.Sequential(
            nn.Linear(302, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(64, 2 * self.n_tasks)
        )

        # criterion
        if args.focal_loss and args.weights is not None:
            self.criterion = [FocalLoss(alpha=1 / w[0]) for w in args.weights]
        elif args.weights is not None:
            self.criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).cuda(), reduction='mean') for
                              w in args.weights]
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch, edge_attr=None):
        z, h, p = self.pre_model(x, edge_index, batch, edge_attr)
        pool = global_add_pool(z, batch)
        out = self.classifier(pool)

        return out, z, h, p

    def get_loss(self, out, y):
        sup_loss = 0.
        for i in range(self.n_tasks):
            y_pred = out[:, i * 2:(i + 1) * 2]
            y_label = y[:, i].squeeze()  # (batch)
            valid_id = ~torch.isnan(y_label)
            y_pred = y_pred[valid_id]
            y_label = y_label[valid_id]
            sup_loss += self.criterion[i](y_pred, y_label)
        return sup_loss
