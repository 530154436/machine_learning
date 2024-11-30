#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from torch import nn


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif act_name.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif act_name.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif act_name.lower() == 'softmax':
            return nn.Softmax()
    elif issubclass(act_name, nn.Module):
        return act_name()
    else:
        raise NotImplementedError
