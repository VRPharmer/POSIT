# -*- coding: utf-8 -*-
import os
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from dataset import build_loader
from loss import FocalLoss
from predictors.prediction2 import PredictionProto
from util import update_mean, seed_everything, create_logger, DotDict, save_best_checkpoint, get_metric_func, \
    load_best_result, NoamLR
from loss import get_sup_loss, get_cut_loss, get_collapse_loss, get_compact_loss, get_contrastive_loss, \
    local_global_loss, local_local_loss


class Trainer:
    def __init__(self, train_loader, valid_loader, test_loader, logger, args):
        self.args = args
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.args.device = torch.device('cpu')
        self.logger = logger
        logger.info(f'Device: {self.args.device}')

        if args.use_tensorboard:
            tensorboard_dir = os.path.join(args.log_dir, args.dataset, 'tensorboard', str(args.nth_run))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            self.writer = None

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.model = self.init_model()

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
        self.criterion = self.set_criterion()

        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    def set_optimizer(self):
        params = [{'params': self.model.parameters(), 'lr': self.args.lr}]
        return torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.l2)

    def set_scheduler(self):
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=75, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-5,
        )
        return scheduler

    def set_criterion(self):
        if self.args.task_type == 'classification':
            if self.args.focal_loss and self.args.weights is not None:
                criterion = [FocalLoss(alpha=1 / w[0]) for w in self.args.weights]
            elif self.args.weights is not None:
                criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).cuda(), reduction='mean') for
                             w in self.args.weights]
            else:
                criterion = torch.nn.CrossEntropyLoss()
        else:
            # regression
            criterion = torch.nn.MSELoss()

        return criterion

    def init_model(self):
        args = self.args
        if not args.use_extra_pretrain:
            print('no pre model')
            pre_model_path = os.path.join(args.ckpt_dir, args.dataset, args.pre_test_name + '.pth')
        else:
            pre_model_path = './model/hiv/hiv0.pth'
        pre_model = torch.load(pre_model_path)
        if args.fix_pretrain:
            for p in pre_model.parameters():
                p.requires_grad = False
        return PredictionProto(pre_model, args).to(args.device)

    def fit(self):
        args = self.args
        start_time = time.time()
        early_stop_cnt = 0
        best_metric = 0. if self.args.task_type == 'classification' else 100.

        for epoch in range(1, args.n_epochs + 1):
            if args.detect_anomaly:
                torch.autograd.set_detect_anomaly(True)
                with torch.autograd.detect_anomaly():
                    loss_map = self.train_epoch()
            else:
                loss_map = self.train_epoch()

            # logging
            log_info = f'epoch [{epoch}/{args.n_epochs}]'
            for name, val in loss_map.items():
                log_info += f' || {name}: {val:.4f}'

            # validate train metric
            train_loss, train_metric = self.validate(self.train_loader, epoch)
            log_info += f' || train_{args.metric}: {train_metric:.4f}'

            # validate performance
            # if epoch % args.valid_step == 0:
            valid_loss, valid_metric = self.validate(self.valid_loader, epoch)
            log_info += f' || valid_{args.metric}: {valid_metric:.4f}'

            self.scheduler.step(valid_loss)

            # test performance
            # if epoch % args.valid_step == 0 and self.test_loader is not None:
            test_loss, test_metric = self.validate(self.test_loader, epoch)
            log_info += f' || test_{args.metric}: {test_metric:.4f}'

            self.logger.info(log_info)

            if self.args.use_tensorboard:
                self.writer.add_scalar(f'scalar/train_{args.metric}', train_metric, epoch)
                self.writer.add_scalar(f'scalar/valid_{args.metric}', valid_metric, epoch)
                self.writer.add_scalar(f'scalar/test_{args.metric}', test_metric, epoch)

            # todo: update learning rate

            if args.task_type == 'classification' and valid_metric > best_metric or \
                    args.task_type == 'regression' and valid_metric < best_metric:
                best_metric, best_epoch = valid_metric, epoch
                save_best_checkpoint(args, epoch, self.model, best_metric, best_epoch, self.optimizer, self.scheduler)

                self.logger.info(f'Best ckpt saved at epoch {epoch}')
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            # Early stopping.
            if early_stop_cnt > args.early_stop > 0:
                self.logger.info(f'Early stop at epoch {epoch}')
                break

        if args.use_tensorboard:
            self.writer.close()

        # todo: save
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
        torch.save(self.model.pre_model, os.path.join(ckpt_dir, f'{args.test_name}_pre.pth'))

        self.logger.info(f'Elapsed time: {time.time() - start_time} seconds')

        self.model, best_epoch = load_best_result(args, self.model, self.logger)
        scores = self.validate(self.test_loader, best_epoch, eval_mode=True)

        return scores

    def train_epoch(self):
        self.model.train()
        self.model.pre_model.train()

        loss_mean = 0.
        cut_loss_mean = 0.
        ortho_loss_mean = 0.
        contra_loss_mean = 0.
        compact_loss_mean = 0.
        sup_loss_mean = 0.
        mi_loss_mean = 0.

        for batch_idx, data in enumerate(self.train_loader):
            device = self.args.device
            data = data.to(device)
            out, s, s_hat, p, p_hat, h_dense, gc, lc, sub_mask, h = self.model(data)

            sup_loss = get_sup_loss(self.args, self.criterion, out, data.y)
            cut_loss = get_cut_loss(data.edge_index, data.batch, s)
            ortho_loss = get_collapse_loss(s)
            contra_loss = get_contrastive_loss(p, p_hat, self.model.pre_model.t2)
            compact_loss = get_compact_loss(p, p_hat)

            # todo: recover
            loss = self.get_loss(cut_loss, 0, ortho_loss, compact_loss, contra_loss, sup_loss, 0)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_mean = update_mean(loss_mean, loss.item(), batch_idx)
            cut_loss_mean = update_mean(cut_loss_mean, cut_loss.item(), batch_idx)
            ortho_loss_mean = update_mean(ortho_loss_mean, ortho_loss.item(), batch_idx)
            contra_loss_mean = update_mean(contra_loss_mean, contra_loss.item(), batch_idx)
            compact_loss_mean = update_mean(compact_loss_mean, compact_loss.item(), batch_idx)
            sup_loss_mean = update_mean(sup_loss_mean, sup_loss.item(), batch_idx)

        return {'loss': loss_mean, 'cut_loss': cut_loss_mean, 'ortho_loss': ortho_loss_mean,
                'contra_loss': contra_loss_mean, 'compact_loss': compact_loss_mean,
                'sup_loss': sup_loss_mean, 'mi_loss': mi_loss_mean}

    @staticmethod
    def get_loss(cut_loss, entro_loss, ortho_loss, compact_loss, contra_loss, sup_loss, mi_loss):
        loss = 1 * cut_loss + 1 * ortho_loss + 1 * compact_loss + 1 * contra_loss + 1 * sup_loss + 1 * mi_loss
        return loss

    @torch.no_grad()
    def validate(self, dataloader, epoch, eval_mode=False):
        self.model.pre_model.eval()
        self.model.eval()

        losses = []
        y_pred_list = {}
        y_label_list = {}
        device = self.args.device

        for data in dataloader:
            data = data.to(device)
            output = self.model(data)[0]
            loss = 0.

            for i in range(len(self.args.task_name)):
                if self.args.task_type == 'classification':
                    y_pred = output[:, i * 2:(i + 1) * 2]
                    y_label = data.y[:, i].squeeze()
                    validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                    if len(validId) == 0:
                        continue
                    if y_label.dim() == 0:
                        y_label = y_label.unsqueeze(0)

                    y_pred = y_pred[torch.tensor(validId).to(device)]
                    y_label = y_label[torch.tensor(validId).to(device)]

                    loss += self.criterion[i](y_pred, y_label)
                    y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                else:
                    y_pred = output[:, i]
                    y_label = data.y[:, i]
                    loss += self.criterion(y_pred, y_label)
                    y_pred = y_pred.detach().cpu().numpy()

                try:
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                losses.append(loss.item())

        # Compute metric
        val_results = []
        metric_func = get_metric_func(metric=self.args.metric)
        for i, task in enumerate(self.args.task_name):
            if self.args.task_type == 'classification':
                nan = False
                if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                    nan = True
                    self.logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while validating')

                if nan:
                    val_results.append(float('nan'))
                    continue

            if len(y_label_list[i]) == 0:
                continue

            val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

        avg_val_results = np.nanmean(val_results)
        val_loss = np.array(losses).mean()
        if eval_mode:
            self.logger.info(f'Seed {2021 + self.args.nth_run} Dataset {self.args.dataset} ==> '
                             f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_score:{avg_val_results:.3f}')
            return val_results

        return val_loss, avg_val_results


def train_proto(config_path: str):
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)['Train']
    global_config = yaml.load(open('./config/global_config.yaml'), Loader=yaml.FullLoader)['Train']
    for k, v in global_config.items():
        config[k] = v
    args = DotDict(config)

    if args.dataset == 'bbbp' or args.dataset == 'bace' or args.dataset == 'hiv':
        args.split_type = 'scaffold'

    logger = create_logger(args)

    seeds = [args.seed + i for i in range(args.n_run)]
    acc_folds = []
    for i, seed in enumerate(seeds):
        seed_everything(seed)  # reset seed per run
        args.seed = seed
        train_loader, val_loader, test_loader, weights = build_loader(args, logger)
        args.weights = weights
        args.nth_run = i
        if i > 0:
            args.verbose = False
        logger.info('------------------- ' + 'run ' + str(i) + ' -------------------')
        acc = train(train_loader, val_loader, test_loader, logger, args)
        acc_folds.append(acc)

    for acc in acc_folds:
        logger.info(f'{acc:.4f}')
    mean_acc = statistics.fmean(acc_folds)
    dev_acc = statistics.stdev(acc_folds)
    logger.info(f'avg acc of {args.n_run} run: {mean_acc:.4f}')
    logger.info(f'std dev of {args.n_run} run: {dev_acc:.4f}')


def train(train_loader, valid_loader, test_loader, logger, args):
    trainer = Trainer(train_loader, valid_loader, test_loader, logger, args)
    scores = trainer.fit()
    return np.nanmean(scores)
