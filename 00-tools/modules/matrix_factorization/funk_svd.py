# /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    Funk-SVD
    simon-funk博客       https://sifter.org/~simon/journal/20061211.html
    博客                 http://freewill.top/2017/03/07/机器学习算法系列（13）：推荐系统（3）——矩阵分解技术/
    surprise            https://surprise.readthedocs.io/en/stable/matrix_factorization.html
    基于矩阵分解的协同过滤  https://mathpretty.com/11495.html
'''
import numpy as np

class DataSet(object):

    def __init__(self, is_triple=True):
        # 样本相关的统计
        self.global_mean = None     # y_true的均值
        self.N = 0                  # 样本容量
        self.n_users = 0            # 用户数
        self.n_items = 0            # 物品数
        self.ui = {}                # 用户-物品集合 <uid, list[iid1,iid2]>
        self.iu = {}                # 物品-用户集合 <iid, list[ud1,uid2]>
        self.data = None            # 原始的数据
        self.is_triple = is_triple  # 是否为三元组

    def create(self, data: np.array):
        ''' 初始化参数、输入 '''
        self.data = data
        if self.is_triple:
            uids = np.unique(data[:, 0])
            jids = np.unique(data[:, 1])
            m, n = uids.size, jids.size

            # 统计评分均值
            self.global_mean = np.mean(data[:, 2])

            # 用户-物品集合 (隐式反馈)
            for u, i, r in self.all_ratings():
                if u not in self.ui:
                    self.ui[u] = set()
                self.ui[u].add(i)
            for k, v in self.ui.items():
                self.ui[k] = list(v)

            # 物品-用户集合
            for u, i, r in self.all_ratings():
                if i not in self.iu:
                    self.iu[i] = set()
                self.iu[i].add(u)
            for k, v in self.iu.items():
                self.iu[k] = list(v)

            # 样本容量
            self.N = data.shape[0]
        else:
            self.global_mean = np.mean(data)
            m, n = data.shape

            # 样本容量
            self.N = m * n

        self.n_users = m
        self.n_items = n

        return self

    def all_ratings(self):
        for i in range(self.data.shape[0]):
            # 三元组
            if self.is_triple:
                u, j, y_true = (self.data[i, xi] for xi in range(3))
                yield u, j, y_true

            # 评分矩阵
            else:
                for j in range(self.data.shape[1]):
                    u, j, y_true = i, j, self.data[i,j]
                    yield u, j, y_true

class FunkSVD(object):

    def __init__(self, learning_rate=0.001, _lambda=0.02, n_epochs=100, epsilon=1000, n_factors=10):
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.n_factors = n_factors

        self.loss = np.inf
        self.P = None           # (u,k): 每一行代表某个人对于不同特征的热爱程度.   =>  X(m,n) = P(m,k)* Q^T(k,n)
        self.Q = None           # (j,k): 每一行代表某个事物拥有这个特征的程度.

    def init_weights(self, train_set:DataSet):
        '''初始化参数'''
        n_users = train_set.n_users
        n_items = train_set.n_items

        self.P = np.random.normal(size=(n_users, self.n_factors))
        self.Q = np.random.normal(size=(n_items, self.n_factors))

    def sgd(self, u, j, y_true, train_set:DataSet):
        '''
        梯度下降更新参数
        :param u:       用户u
        :param j:       物品j
        :param y_true:  评分
        '''
        err = y_true - self.predict(u, j)
        self.P[u] += self.learning_rate * (err * self.Q[j] - self._lambda * self.P[u])
        self.Q[j] += self.learning_rate * (err * self.P[u] - self._lambda * self.Q[j])

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
            for u, j, y_true in train_set.all_ratings():
                self.sgd(u, j, y_true, train_set)
                loss += np.square(y_true - self.predict(u, j))
            epoch += 1
            self.loss = loss
            mse = self.loss/train_set.N
            print(f'Epoch {epoch}, loss={self.loss}, MSE={mse}, RMSE={np.sqrt(mse)}')
        print(f'Train Done, Q.shape={self.Q.shape}, P.shape={self.P.shape}')

    def _know_u(self, u:int):
        return u!=None and u>=0 and u<self.P.shape[0]

    def _know_j(self, j:int):
        return j!=None and j>=0 and j<self.Q.shape[0]

    def predict(self, u:int, j:int):
        '''
        预测u用户对物品i的评分
        :param u:   用户
        :param j:   物品
        '''
        if self._know_u(u) and self._know_j(j):
            return np.dot(self.P[u, :], self.Q[j, :])
        else:
            return None