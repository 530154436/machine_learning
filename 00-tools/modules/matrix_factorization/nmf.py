# /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    NMF(非负矩阵分解)
    可基于Funk-SVD、也可基于Bias-SVD

    http://xtf615.com/2018/05/03/recommender-system-survey/
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF
'''
import numpy as np
from matrix_factorization.funk_svd import FunkSVD,DataSet

class NMF(FunkSVD):

    def __init__(self, learning_rate=0.001, _lambda=0.02, n_epochs=100, epsilon=1000, n_factors=10,
                 init_low=0, init_high=1):
        super(NMF, self).__init__(learning_rate=learning_rate, _lambda=_lambda,
                                  n_epochs=n_epochs, epsilon=epsilon, n_factors=n_factors)

        # P、Q矩阵初始化的权值范围
        self.init_low = init_low
        self.init_high = init_high

        self.mu = None  # 评分系统的平均分
        self.bu = None  # bu 表示的是用户u的偏置
        self.bj = None  # bj表示的是物品j的偏置

    def init_weights(self, train_set:DataSet):
        '''初始化参数'''
        n_users = train_set.n_users
        n_items = train_set.n_items

        self.mu = train_set.global_mean
        self.P = np.random.uniform(self.init_low, self.init_high, size=(n_users, self.n_factors))
        self.Q = np.random.uniform(self.init_low, self.init_high, size=(n_items, self.n_factors))
        self.bu = np.zeros(n_users, np.double)
        self.bj = np.zeros(n_items, np.double)

    def fit(self, train_set:DataSet):
        '''
        训练
        :param data:        原始数据
        :param is_triple:   True/三元组(uid, jid, rating)
                            False/矩阵(rating = data[uid, jid])
        '''
        # 输入数据、参数初始化
        self.init_weights(train_set)

        # 开始训练
        epoch = 0
        while epoch < self.n_epochs and self.loss > self.epsilon:
            loss = 0
            user_num = np.zeros((train_set.n_users, self.n_factors))
            user_denom = np.zeros((train_set.n_users, self.n_factors))
            item_num = np.zeros((train_set.n_items, self.n_factors))
            item_denom = np.zeros((train_set.n_items, self.n_factors))

            for u, j, y_true in train_set.all_ratings():
                # 预测
                y_hat = self.predict(u, j)

                # 残差
                err = y_true - y_hat

                # 更新偏置
                self.bu[u] += self.learning_rate * (err - self._lambda * self.bu[u])
                self.bj[j] += self.learning_rate * (err - self._lambda * self.bj[j])

                # compute numerators and denominators
                user_num[u] += self.Q[j] * y_true
                user_denom[u] += self.Q[j] * y_hat
                item_num[j] += self.P[u] * y_true
                item_denom[j] += self.P[u] * y_hat

                loss += np.square(y_true - self.predict(u, j))

            # 更新用户矩阵
            for u in range(train_set.n_users):
                n_rating = len(train_set.ui[u])
                user_denom[u] += n_rating * self._lambda * self.P[u]
                self.P[u] *= user_num[u] / user_denom[u]

            # 更新物品矩阵
            for j in range(train_set.n_items):
                n_rating = len(train_set.iu[j])
                item_denom[j] += n_rating * self._lambda * self.Q[j]
                self.Q[j] *= item_num[j] / item_denom[j]

            epoch += 1
            self.loss = loss
            mse = self.loss/train_set.N
            print(f'Epoch {epoch}, loss={self.loss}, MSE={mse}, RMSE={np.sqrt(mse)}')
        print(f'Train Done, Q.shape={self.Q.shape}, P.shape={self.P.shape}')

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