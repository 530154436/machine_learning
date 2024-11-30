#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from typing import Callable, List
from src.framework.basic.callbacks import EarlyStopper


class CTRTrainer(object):
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
        model,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn: Callable = torch.optim.lr_scheduler.StepLR,
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
        self.device = torch.device(device)  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer

        if scheduler_params is None:
            scheduler_params = {"step_size": 1, "gamma": 0.8}
        self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)

        self.loss_fn = torch.nn.BCELoss()  # default loss cross_entropy
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.n_epoch = n_epoch
        # self.early_stopper = EarlyStopper(patience=early_stop_patience)
        self.early_stopper = EarlyStopper(patience=early_stop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10) -> float:
        self.model.train()
        targets, predicts = list(), list()
        batch, total_loss = 1, 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for batch, xy in enumerate(tk0, start=1):
            x_tuple = tuple(x.to(self.device) for x in xy[:-1])
            y = xy[-1].to(self.device)
            y_pred = self.model(*x_tuple)
            loss = self.loss_fn(y_pred, y.float())

            targets.extend(y.tolist())
            predicts.extend(y_pred.tolist())

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (batch + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / batch)
                # total_loss = 0
        train_auc = self.evaluate_fn(targets, predicts)
        return total_loss/batch, train_auc

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            epoch_i += 1
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            train_loss, train_auc = self.train_one_epoch(train_dataloader)
            self.scheduler.step()  # update lr in epoch level by scheduler

            if val_dataloader:
                val_loss, val_auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, "Current lr : {}".format(lr),
                      'train_loss:', train_loss, 'train_auc:', train_auc,
                      'val_loss:', val_loss, 'val_auc:', val_auc)

                # if self.early_stopper.stop_training(val_auc, self.model.state_dict(), mode='max'):
                if self.early_stopper.stop_training(val_loss, self.model.state_dict(), mode='min'):
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    print('Current AUC: %.6f, Best Value: %.6f\n' % (val_auc, self.early_stopper.best_value))
                    break
        torch.save(self.model.state_dict(), self.model_path)  # save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        batch, total_loss = 1, 0
        with torch.no_grad():
            for batch, xy in enumerate(data_loader, start=1):
                x_tuple = tuple(x.to(self.device) for x in xy[:-1])
                y = xy[-1].to(self.device)
                y_pred = self.model(*x_tuple)

                total_loss += self.loss_fn(y_pred, y.float()).float()
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        val_loss = float(total_loss / batch)
        val_auc = self.evaluate_fn(targets, predicts)
        return val_loss, val_auc

    @classmethod
    def predict_prob(cls, model, data_loader, device='cpu'):
        # 加载模型
        model.eval()
        predicts = list()
        with torch.no_grad():
            for i, x in enumerate(data_loader, start=1):
                x_tuple = tuple(x.to(device) for x in x)
                y_pred = model(*x_tuple)
                predicts.extend(y_pred.tolist())
        return predicts

    @classmethod
    def predict(cls, model, data_loader) -> List[int]:
        predicts = cls.predict_prob(model, data_loader)
        predicts = np.where(np.array(predicts) > 0.5, 1, 0).reshape(-1)
        return predicts
