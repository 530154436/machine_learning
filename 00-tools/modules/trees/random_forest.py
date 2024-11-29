# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from tools import sampling

'''
    周志华 8.3 Bagging与随机森林 p178
    随机森林的Python源码实现 https://zhuanlan.zhihu.com/p/32179140
'''

class MyRandomForestClassifier(object):

    def __init__(self, n_estimators:int=10, max_features:int=None, max_depth:int=None):

        # 训练n_estimators棵树
        self.n_estimators = n_estimators

        # 每棵树选用数据集中的最大的特征数
        self.max_features = max_features

        # 随机特征索引 a[i]~第i颗树随机特征对应的索引列表
        self.features = []

        # 每棵树的最大深度
        self.max_depth = max_depth

        # 训练的决策树集合
        self.trees = []

    def fit(self, x:np.array, y:np.array):

        n_features = x.shape[1]
        if not self.max_features or self.max_features>n_features:
            self.max_features = n_features

        for i in range(self.n_estimators):

            # 自助法采样样本
            x_bs, y_bs = sampling.bootstrap(x, y, seed=i)

            # 自助采样特征
            self.features.append(np.random.RandomState(i).randint(0, self.max_features, self.max_features))
            x_bs = x_bs[:, self.features[i]]

            # 训练绝测
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_bs, y_bs)

            self.trees.append(tree)

    def predict(self, x:np.array):
        '''
        绝对多数投票
        '''
        m = x.shape[0]

        # 投票矩阵 (i,j): 第i个样本第j颗树的预测结果
        votes = np.zeros((m, self.n_estimators)).astype(int)
        for j in range(self.n_estimators):
            # 随机特征
            x_bs = x[:, self.features[j]]
            y_pre = self.trees[j].predict(x_bs)
            votes[:,j] = y_pre

        # 每个样本最多投票的类别作为最终结果
        y_pred = np.zeros(m).astype(int)
        for i in range(m):
            y_pred[i] = np.bincount(votes[i, :]).argmax() # 统计每个类别得票总数、并求出最大票数对应的索引

        return y_pred

def test_classification():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
    print(f"X_train.shape={X_train.shape},\ty_train.shape={y_train.shape}")
    print(f"X_test.shape={X_test.shape},\ty_test.shape={y_test.shape}")

    mrfc = MyRandomForestClassifier(n_estimators=10)
    mrfc.fit(X_train, y_train)
    y_pred = mrfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("MyRF Accuracy:", accuracy)

    # Sk-learn
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print(f'Sklearn Accuray = {accuracy_score(y_test, y_pred)})')

if __name__ == "__main__":
    test_classification()

