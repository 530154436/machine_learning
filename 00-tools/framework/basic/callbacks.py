#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import copy


class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_value = None
        self.best_weights = None

    def stop_training(self, value, weights, mode: str = 'max'):
        """whether to stop training.

        Args:
            value (float): auc score in val data. auc、loss
            weights (tensor): the weights of model
            mode:
        """
        if self.best_value is None:
            self.best_value = 0 if mode == 'max' else 999999

        if (value > self.best_value and mode == 'max') or \
                (value < self.best_value and mode == 'min'):
            self.best_value = value
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True
