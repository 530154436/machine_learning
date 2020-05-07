# /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    BiasSVD
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
    https://yijunsu.github.io/2018/06/09/2018-06-09-Basic,Regularized%20and%20BiasSVD%20Matrix%20Factorization/#BiasSVD-Matrix-Factorization
    Numpy实现库 https://github.com/lxmly/recsyspy/tree/master/
'''
import numpy as np
from matrix_factorization.funk_svd import FunkSVD

class BiasSVD(FunkSVD):

    def __init__(self, learning_rate=0.001, _lambda=0.02, n_epochs=100, epsilon=1000, n_factors=10):
        super(BiasSVD, self).__init__( learning_rate=learning_rate, _lambda=_lambda,
                                       n_epochs=n_epochs, epsilon=epsilon, n_factors=n_factors)
        self.mu = None  # 评分系统的平均分
        self.bu = None  # bu 表示的是用户u的偏置
        self.bj = None  # bj表示的是物品j的偏置

    def init_weights(self, m, n):
        ''' 初始化参数 '''
        self.P = np.random.normal(size=(m, self.n_factors))
        self.Q = np.random.normal(size=(n, self.n_factors))
        self.mu = self.global_mean
        self.bu = np.zeros(m, np.double)
        self.bj = np.zeros(n, np.double)

    def sgd(self, u, j, y_true, y_hat):
        '''
        梯度下降更新参数
        :param u:       用户u
        :param j:       物品j
        :param y_true:  评分
        '''
        e_uj = y_true - y_hat
        self.P[u] += self.learning_rate * (e_uj * self.Q[j] - self._lambda * self.P[u])
        self.Q[j] += self.learning_rate * (e_uj * self.P[u] - self._lambda * self.Q[j])
        self.bu[u] += self.learning_rate * (e_uj - self._lambda * self.bu[u])
        self.bj[j] += self.learning_rate * (e_uj - self._lambda * self.bj[j])

    def predict(self, u:int, j:int):
        '''
        预测u用户对物品i的评分
        :param u:   用户
        :param j:   物品
        '''
        if u==None and j==None:
            return
        return np.dot(self.P[u, :], self.Q[j, :]) + self.mu + self.bu[u] + self.bj[j]
