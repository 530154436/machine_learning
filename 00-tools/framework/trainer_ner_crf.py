#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import tqdm
from typing import Callable
from torch import nn
from transformers import get_linear_schedule_with_warmup

from src.framework.basic.callbacks import EarlyStopper


class Trainer4NerCrf(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        early_stop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn: Callable = torch.optim.AdamW,
        bert_optimizer_params: dict = None,
        optimizer_params: dict = None,
        scheduler_fn: Callable = get_linear_schedule_with_warmup,
        scheduler_params=None,
        n_epoch=10,
        early_stop_patience=10,
        device="cpu",
        gpus=None,
        model_path="model.pth",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)
        self.model.to(self.device)

        # 配置：优化器
        self.bert_optimizer_params = {"lr": 3e-5, "weight_decay": 1e-5} if bert_optimizer_params is None else bert_optimizer_params
        self.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5} if optimizer_params is None else optimizer_params
        self.optimizer_fn = optimizer_fn
        self.optimizer = None
        self.build_optimizer()

        # 配置：学习率调度器(差分学习率)
        # 注意：num_training_steps = len(train_loader) * n_epochs
        self.scheduler_params = {"num_warmup_steps": 10, "num_training_steps": 1000} if scheduler_params is None else scheduler_params
        self.scheduler_fn = scheduler_fn
        self.scheduler = scheduler_fn(self.optimizer, **self.scheduler_params)

        self.evaluate_fn = None  # default evaluate function
        self.n_epoch = n_epoch
        # self.early_stopper = EarlyStopper(patience=early_stop_patience)
        self.early_stopper = EarlyStopper(patience=early_stop_patience)
        self.model_path = model_path

    def build_optimizer(self):
        """配置优化器（optimizer）"""
        if self.optimizer is not None:
            return self.optimizer

        module = (self.model.module if hasattr(self.model, "module") else self.model)

        # 差分学习率
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []
        for name, para in model_param:
            space = name.split('.')
            if space[0] in ('bert_module', "bert"):
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], **self.bert_optimizer_params},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": self.bert_optimizer_params.get("lr")},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)], **self.optimizer_params},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": self.optimizer_params.get("lr")},
        ]

        self.optimizer = self.optimizer_fn(optimizer_grouped_parameters)
        return self.optimizer

    def train_one_epoch(self, data_loader, log_interval=10) -> float:
        self.model.train()
        batch, total_loss = 1, 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for batch, xy in enumerate(tk0, start=1):
            xy_tuple = tuple(x.to(self.device) for x in xy)
            # [bs, max_seq_length]
            loss, y_pred = self.model(*xy_tuple)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (batch + 1) % log_interval == 0:
                tk0.set_postfix(loss=round(total_loss/batch, 5))
        return total_loss/batch

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            epoch_i += 1
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            train_loss = self.train_one_epoch(train_dataloader)
            self.scheduler.step()  # update lr in epoch level by scheduler

            if val_dataloader:
                val_loss = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, "Current lr : {}".format(lr), 'train_loss:', train_loss, 'val_loss:', val_loss)

                if self.early_stopper.stop_training(val_loss, self.model.state_dict(), mode='min'):
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    print('Current loss: %.6f, Best Value: %.6f\n' % (val_loss, self.early_stopper.best_value))
                    break
        torch.save(self.model.state_dict(), self.model_path)  # save best auc model
        print('Saved model\'s loss: %.6f' % self.early_stopper.best_value)

    @torch.no_grad()
    def evaluate(self, model, data_loader):
        model.eval()
        batch, total_loss = 1, 0
        for batch, xy in enumerate(data_loader, start=1):
            xy_tuple = tuple(x.to(self.device) for x in xy)
            loss, y_pred = self.model(*xy_tuple)
            total_loss += loss
        val_loss = float(total_loss / batch)
        return val_loss
