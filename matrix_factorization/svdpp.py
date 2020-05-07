# /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    SVD++
    理论讲解比较好的 http://xtf615.com/2018/05/03/recommender-system-survey/
    Surprise https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
'''
import numpy as np
from matrix_factorization.bias_svd import BiasSVD

class SVDpp(BiasSVD):

    def __init__(self, learning_rate=0.001, _lambda=0.02, n_epochs=100, epsilon=1000, n_factors=10):
        super(SVDpp, self).__init__( learning_rate=learning_rate, _lambda=_lambda,
                                     n_epochs=n_epochs, epsilon=epsilon, n_factors=n_factors)
        self.yi = None              # 用户u对商品i的隐式偏好 => 体现在每个物品上
        self.u_implicit_fb = None   # 用户u对商品i(特征)的隐式偏好 => 发生互动的物品偏好求和归一

    def init_weights(self, m, n):
        ''' 初始化参数 '''
        self.mu = self.global_mean
        self.bu = np.zeros(m, np.double)
        self.bj = np.zeros(n, np.double)

        self.P = np.random.normal(size=(m, self.n_factors))
        self.Q = np.random.normal(size=(n, self.n_factors))

        # 隐式反馈
        self.yi = np.random.normal(size=(n, self.n_factors))
        self.u_implicit_fb = np.random.normal(size=(m, self.n_factors))

        # 用户对物品i的偏好集合、并计算用户的隐式反馈、注意:ui是物品的id列表
        for u in range(m):
            ui = self.ui[u]
            ui_sqrt = np.sqrt(len(ui))
            self.u_implicit_fb[u] = np.sum(self.yi[ui], axis=0) / ui_sqrt

    def sgd(self, u, j, y_true):
        '''
        梯度下降更新参数
        :param u:       用户u
        :param j:       物品j
        :param y_true:  评分
        '''
        # 残差
        e_uj = y_true - self.predict(u, j)

        # 更新显示因子
        self.P[u] += self.learning_rate * (e_uj * self.Q[j] - self._lambda * self.P[u])
        self.Q[j] += self.learning_rate * (e_uj * self.P[u] - self._lambda * self.Q[j])

        # 更新偏置
        self.bu[u] += self.learning_rate * (e_uj - self._lambda * self.bu[u])
        self.bj[j] += self.learning_rate * (e_uj - self._lambda * self.bj[j])

        # 更新隐式因子
        ui = self.ui[u]
        ui_sqrt = np.sqrt(len(ui))
        self.yi[ui] = self.learning_rate * (e_uj * self.Q[j] / ui_sqrt - self._lambda * self.yi[ui])
        # for i in ui:
        #     self.yi[i] = self.learning_rate * (e_uj * self.Q[j] / ui_sqrt - self._lambda * self.yi[i])
        self.u_implicit_fb[u] = np.sum(self.yi[ui], axis=0) / ui_sqrt

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
            rating += np.dot(self.P[u, :] + self.u_implicit_fb[u], self.Q[j, :])

        return rating