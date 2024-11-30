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

    def init_weights(self, train_set):
        ''' 初始化参数 '''
        n_users = train_set.n_users
        n_items = train_set.n_items

        self.mu = train_set.global_mean
        self.P = np.random.normal(size=(n_users, self.n_factors))
        self.Q = np.random.normal(size=(n_items, self.n_factors))
        self.bu = np.zeros(n_users, np.double)
        self.bj = np.zeros(n_items, np.double)

    def sgd(self, u, j, y_true, train_set):
        '''
        梯度下降更新参数
        :param u:       用户u
        :param j:       物品j
        :param y_true:  评分
        '''
        e_uj = y_true - self.predict(u, j)
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
        rating = self.mu
        know_u = self._know_u(u)
        know_j = self._know_j(j)

        if know_u:
            rating += self.bu[u]

        if know_j:
            rating += self.bj[j]

        if know_u and know_j:
            rating += np.dot(self.P[u, :], self.Q[j, :])

        return rating
