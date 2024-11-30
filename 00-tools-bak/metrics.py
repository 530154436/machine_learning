# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

def MSE(y_true, y_hat):
    '''
    MSE(Mean Square Error 平均平方误差) = \frac 1 N \sum_{i=1}^{N} (y_i - \hat{y_i})^2
    :param y_true:  真实值
    :param y_hat:   预测值
    :return:
    '''
    return np.sum(np.square(y_true - y_hat)) / y_true.size

def RMSE(y_true, y_hat):
    '''
    RMSE(Root Mean Square Error 平均平方误差的平方根) = \frac 1 N \sum_{i=1}^{N} (y_i - \hat{y_i})^2
    :param y_true:  真实值
    :param y_hat:   预测值
    :return:
    '''
    return np.sqrt(MSE(y_true, y_hat))

def MAE(y_true, y_hat):
    '''
    MAE(Mean Absolute Error 平均绝对误差) = \frac 1 N \sum_{i=1}^{N} |y_i - \hat{y_i}|
    :param y_true:  真实值
    :param y_hat:   预测值
    :return:
    '''
    return np.absolute(y_true-y_hat) / y_true.size