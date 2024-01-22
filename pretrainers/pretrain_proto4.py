# -*- coding: utf-8 -*-
import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import torch_geometric as tg
from torch.utils.tensorboard import SummaryWriter

from pretrain_models.pretrain_model import ProtoMiner
from util import update_mean, seed_everything, tsne, DotDict, create_logger, smiles_to_graph
from dataset import build_loader
from loss import get_cut_loss, get_collapse_loss, get_contrastive_loss, get_compact_loss, local_global_loss, \
    local_local_loss


class PreTrainer:
    def __init__(self, dataloader, logger, args):
        self.args = args
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device: {args.device}')

        self.logger = logger
        self.dataloader = dataloader
        self.model = ProtoMiner(args).to(args.device)

        if args.use_tensorboard:
            tensorboard_dir = os.path.join(args.log_dir, args.dataset, 'tensorboard', 'pretrain')
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            self.writer = None

        self.fig_dir = os.path.join(args.fig_dir, args.dataset)
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.checkpoint = os.path.join(ckpt_dir, args.test_name + '.pth')

        if args.verbose:
            logger.info(f'Model: {self.model}')
            logger.info(f'Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            logger.info('-' * 50)
            logger.info('Trainable parameters:')
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
            logger.info('-' * 50)

        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()

    def set_optimizer(self):
        # todo: modify
        params = [{'params': self.model.parameters(), 'lr': self.args.lr}]
        return torch.optim.Adam(params, weight_decay=self.args.l2)

    def set_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        return scheduler

    def fit(self):
        args = self.args
        self.optimizer = self.set_optimizer()
        self.logger.info("Start pretraining")
        start_time = time.time()

        for epoch in range(1, args.n_epochs + 1):
            self.model.train()
            if args.detect_anomaly:
                torch.autograd.set_detect_anomaly(True)
                with torch.autograd.detect_anomaly():
                    loss_map = self.train_epoch()
            else:
                loss_map = self.train_epoch()

            self.scheduler.step(loss_map['loss'])

            if args.use_tensorboard:
                self.writer.add_scalars("scalar/loss", loss_map, epoch)

            log_info = "epoch [{}/{}]".format(epoch, args.n_epochs)
            for loss, val in loss_map.items():
                log_info += " || {}: {:.4f}".format(loss, val)
            self.logger.info(log_info)

            # evaluate performance every valid_step epochs
            if epoch % args.evaluate_step == 0 or epoch == args.n_epochs:
                self.evaluate(epoch)

        if args.save_model:
            torch.save(self.model, self.checkpoint)
            self.logger.info(f'Model saved at: {self.checkpoint}')

        if args.use_tensorboard:
            self.writer.close()

        self.logger.info(f'Elapsed time: {time.time() - start_time}s')

    def train_epoch(self):
        loss_mean = 0.
        cut_loss_mean = 0.
        ortho_loss_mean = 0.
        contra_loss_mean = 0.
        compact_loss_mean = 0.
        mi_loss_mean = 0.

        for batch_idx, data in enumerate(self.dataloader):
            device = self.args.device
            data = data.to(device)
            outputs = self.model(data)
            s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask = outputs

            cut_loss = get_cut_loss(data.edge_index, data.batch, s)
            ortho_loss = get_collapse_loss(s)
            contra_loss = get_contrastive_loss(p, p_hat, self.args.t2)
            compact_loss = get_compact_loss(p, p_hat)

            loss = self.args.alpha * cut_loss
            loss += self.args.yeta * ortho_loss
            loss += 1 * contra_loss
            loss += 1 * compact_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_mean = update_mean(loss_mean, loss.item(), batch_idx)
            cut_loss_mean = update_mean(cut_loss_mean, cut_loss.item(), batch_idx)
            ortho_loss_mean = update_mean(ortho_loss_mean, ortho_loss.item(), batch_idx)
            contra_loss_mean = update_mean(contra_loss_mean, contra_loss.item(), batch_idx)
            compact_loss_mean = update_mean(compact_loss_mean, compact_loss.item(), batch_idx)

        return {'loss': loss_mean, 'cut_loss': cut_loss_mean, 'ortho_loss': ortho_loss_mean,
                'contra_loss': contra_loss_mean, 'compact_loss': compact_loss_mean, 'mi_loss': mi_loss_mean}

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        mod_mean = sscore_mean = chi_mean = dbi_mean = 0.

        for batch_idx, data in enumerate(self.dataloader):
            device = self.args.device
            data = data.to(device)
            outputs = self.model(data)
            s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask = outputs
            labels = torch.argmax(p, dim=-1)

            mod, sscore, chi, dbi = self.model.evaluate(data.edge_index, data.batch, s, h, labels)
            mod_mean = update_mean(mod_mean, mod, batch_idx)
            sscore_mean = update_mean(sscore_mean, sscore, batch_idx)
            chi_mean = update_mean(chi_mean, chi, batch_idx)
            dbi_mean = update_mean(dbi_mean, dbi, batch_idx)

        metric_map = {'mod': mod_mean, 'sscore': sscore_mean, 'chi': chi_mean, 'dbi': dbi_mean}

        if self.args.use_tensorboard:
            self.writer.add_scalars('scalar/metric', metric_map, epoch)

        log_info = "epoch [{}/{}]".format(epoch, self.args.n_epochs)
        for metric, val in metric_map.items():
            log_info += " || {}: {:.4f}".format(metric, val)
        self.logger.info(log_info)

    def plot_self(self):
        self._plot('./fig/{}/'.format(self.args.dataset), self.model.to('cpu'))

    def plot_model(self, fig_path, model):
        self._plot(fig_path, model.to('cpu'))

    @torch.no_grad()
    def _plot(self, fig_path, model):
        Xs, Ys = [], []
        for batch_idx, data in enumerate(self.dataloader):
            outputs = model(data)
            s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask = outputs
            y_hat = torch.argmax(p, dim=-1)
            Xs.append(h)  # H
            Ys.append(y_hat)  # y_hat

        Xs = torch.cat(Xs, dim=0).detach().cpu().numpy()
        Ys = torch.cat(Ys, dim=0).detach().cpu().numpy()
        Us = model.U.detach().cpu().numpy()

        Xs_embed, Us_embed = tsne(Xs, Us)

        _ = plt.figure()
        colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for _ in range(Us.shape[0])])
        # colors = ((0.020763084316730684, 0.3392633772201282, 0.3250325707640547),
        #           (0.9987346279280792, 0.21373423234771183, 0.5415462670112116),
        #           (0.8241956813883989, 0.6041832739012899, 0.3177968617209419),
        #           (0.662895880527682, 0.4213214904767877, 0.7077275950429015),
        #           (0.02759961040632186, 0.09997892518676765, 0.10842725661063468),
        #           (0.1712013327126196, 0.14835956369807424, 0.18659038515697313),
        #           (0.552110614786549, 0.7457108331409652, 0.5243742344791907),
        #           (0.9381972647005745, 0.5894039722764554, 0.8898353703665299),
        #           (0.525429816776611, 0.16797837363612578, 0.2967935071162503),
        #           (0.7620557320070429, 0.197690839989197, 0.9863327356244473),
        #           (0.41325143445805756, 0.10930374677326893, 0.9465519456026892),
        #           (0.10698629273768379, 0.33930634697575, 0.1797811784357508),
        #           (0.6275826458722844, 0.6626754672583846, 0.3817066486492088),
        #           (0.1487282561902743, 0.871220842455088, 0.7840339489967384),
        #           (0.4416025463791019, 0.6742122265791213, 0.37829356054210717),
        #           (0.6230828057686211, 0.09873068265563623, 0.6122588219847629),
        #           (0.88741251909064, 0.9206715005629298, 0.24929222643447413),
        #           (0.5828845643584099, 0.5909396232444835, 0.2788678845160417),
        #           (0.46074915129795924, 0.9434240871122579, 0.775502629370766),
        #           (0.02969658669081232, 0.8814945660353604, 0.8866417339413124),
        #           (0.5261219881308883, 0.48060721485564684, 0.4854202598080225),
        #           (0.6840838534710899, 0.8213529927751597, 0.11915654649082352),
        #           (0.8297353958541973, 0.16750838410761748, 0.5657155256472977),
        #           (0.218739741944186, 0.7426282505959713, 0.7112977899765991),
        #           (0.48960412421332533, 0.09954455194084755, 0.5209298169395392),
        #           (0.050736825604706826, 0.7345412216028596, 0.8808530522385168),
        #           (0.7423161994311442, 0.5026815978659492, 0.4737183437611815),
        #           (0.5325976940497347, 0.4473311468709963, 0.7308086804400351),
        #           (0.8422512594359389, 0.23450449951044217, 0.07271820020625974),
        #           (0.5754344556679859, 0.05393521602808249, 0.13650524176574086))
        colors = [rgb2hex(x) for x in colors]

        for center, color in enumerate(colors):
            index = (Ys == center).nonzero()
            plt.scatter(Xs_embed[index, 0], Xs_embed[index, 1], s=10, marker='.', alpha=0.5, c=color)
            plt.scatter(Us_embed[center, 0], Us_embed[center, 1], s=10, c='black', marker='x', alpha=0.9)

        plt.title('Clustering Result')
        if self.args.save_fig:
            plt.savefig(fig_path + 'all-clustering-results0', dpi=1000)
            plt.show()
        else:
            plt.show()
        plt.clf()
        plt.close()

    def show_graphs(self, model):
        model.eval()
        model = model.cuda()
        for idx, data in enumerate(self.dataloader):
            device = self.args.device
            outputs = model(data.to(device))
            s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask = outputs
            from torch_geometric.utils import to_dense_batch
            batch = data.batch
            S, mask = to_dense_batch(s, batch)
            for i, d in enumerate(data):
                self._show_graph(data[i], S[i])

    @staticmethod
    def _show_graph(data, S):
        import networkx as nx
        assign = S.argmax(dim=-1).squeeze(0)
        G = tg.utils.to_networkx(data, to_undirected=True)
        color_map = []
        for idx, node in enumerate(G):
            color_map.append(assign[idx].item())
        nx.draw_networkx(G, with_labels=True, font_weight='bold', node_color=color_map, width=2.0)
        plt.show()
        plt.clf()
        print(1)

    def count_sub(self, model):
        tot = torch.zeros(30)
        for batch_idx, data in enumerate(self.dataloader):
            outputs = model(data)
            s, s_hat, h, h_dense, p, p_hat, z_dense, g, gc, lc, sub_mask = outputs
            ss = s.sum(dim=0)
            tot += ss
        print(tot)


def pretrain_proto(config_path: str):
    # set config
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)['Pretrain']
    global_config = yaml.load(open('./config/global_config.yaml'), Loader=yaml.FullLoader)['Pretrain']
    for k, v in global_config.items():
        config[k] = v
    args = DotDict(config)

    seed_everything(args.seed)
    logger = create_logger(args)

    dataloader = build_loader(args, logger)
    trainer = PreTrainer(dataloader=dataloader, logger=logger, args=args)
    trainer.fit()


if __name__ == '__main__':
    print(1)
