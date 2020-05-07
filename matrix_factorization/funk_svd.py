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
        self.u_mapping = None   # 用户ID映射
        self.j_mapping = None   # 物品ID映射

        self.global_mean = None # y_true的均值
        self.N = 0              # 样本容量

    def prepare_data(self, data:np.array, is_triple):
        ''' 初始化参数、输入 '''
        if is_triple:
            uids = np.unique(data[:, 0])
            jids = np.unique(data[:, 1])
            self.global_mean = np.mean(data[:, 2])
            self.u_mapping = {uid: idx for idx, uid in enumerate(uids)}
            self.j_mapping = {jid: idx for idx, jid in enumerate(jids)}
            m, n = uids.size, jids.size
        else:
            self.global_mean = np.mean(data)
            m, n = data.shape

        self.N = data.shape[0]
        return m,n

    def init_weights(self, m, n):
        '''初始化参数'''
        self.P = np.random.normal(size=(m, self.n_factors))
        self.Q = np.random.normal(size=(n, self.n_factors))

    def sgd(self, u, j, y_true, y_hat):
        '''
        梯度下降更新参数
        :param u:       用户u
        :param j:       物品j
        :param y_true:  评分
        '''
        err = y_true - y_hat
        self.P[u] += self.learning_rate * (err * self.Q[j] - self._lambda * self.P[u])
        self.Q[j] += self.learning_rate * (err * self.P[u] - self._lambda * self.Q[j])

    def _loss(self, u, j, y_true):
        '''
        拟合一个样本并返回损失值
        :param u:       用户u
        :param j:       物品j
        :param y_true:  真实评分
        :return: 损失值
        '''
        y_hat = self.predict(u, j)
        self.sgd(u, j, y_true, y_hat)
        return np.square(y_true - y_hat)

    def fit(self, data:np.array, is_triple=True):
        '''
        训练
        :param data:        原始数据
        :param is_triple:   True/三元组(uid, jid, rating)
                            False/矩阵(rating = data[uid, jid])
        '''
        # 输入数据、参数初始化
        m, n = self.prepare_data(data, is_triple)
        self.init_weights(m, n)

        # 开始训练
        epoch = 0
        while epoch < self.n_epochs and self.loss > self.epsilon:
            loss = 0
            for i in range(data.shape[0]):
                if is_triple:
                    # ID映射
                    u, j, y_true = (data[i, xi] for xi in range(3))
                    u = self.u_mapping.get(u)
                    j = self.j_mapping.get(j)

                    loss += self._loss(u, j, y_true)
                else:
                    for j in range(data.shape[1]):
                        u, y_true = i, data[i,j]
                        loss += self._loss(u, j, y_true)
            epoch += 1
            self.loss = loss
            print(f'Epoch {epoch}, loss={self.loss}')
        print(f'Train Done, Q.shape={self.Q.shape}, P.shape={self.P.shape}')

    def predict(self, u:int, j:int):
        '''
        预测u用户对物品i的评分
        :param u:   用户
        :param j:   物品
        '''
        if u==None and j==None:
            return
        return np.dot(self.P[u, :], self.Q[j, :])