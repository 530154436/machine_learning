# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd

class NaiveBayes():
    '''
    朴素贝叶斯 (基于Pandas)
    '''
    def __init__(self):
        self._lambda = 1
        self.y_count = None   # (类型:数量)
        self.y_prob = None    # (类型:概率)
        self.features = dict()

    def fit(self, x_train, y_train):
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)

        # 先验概率
        self.y_count = y_train[0].value_counts()
        self.y_prob = (self.y_count+self._lambda)/ (self.y_count.sum(0)+self.y_count.size*self._lambda)

        # 条件概率
        for j in x_train.columns:  # 维度j
            for yk,yk_count in self.y_count.items():
                X_j_count = x_train[(y_train == yk).values][j].value_counts()   # (xj, count)
                Sj = X_j_count.size
                for xj, xj_count in X_j_count.items():
                    self.features[(j, xj, yk)] = (xj_count+self._lambda) / (yk_count+Sj*self._lambda)

    def predict(self, x_test, with_pro=True):
        y_pred = []
        for sample in x_test:
            # 后验概率
            max_pro = 0
            max_yk = 0
            for yk, yk_pro in self.y_prob.items():
                prob = yk_pro
                for j,xj in enumerate(sample):
                    xj_prob = self.features.get((j, xj, yk))
                    if not xj_prob: xj_prob = self._lambda/ (self.y_count.sum(0)+self.y_count.size*self._lambda)
                    prob *= xj_prob
                if prob>max_pro:
                    max_pro = prob
                    max_yk = yk
            if with_pro:
                y_pred.append((max_yk, max_pro))
            else:
                y_pred.append(max_yk)
        return y_pred

if __name__ == '__main__':
    x_train = [[1, 'S'],
               [1, 'M'],
               [1, 'M'],
               [1, 'S'],
               [1, 'S'],
               [2, 'S'],
               [2, 'M'],
               [2, 'M'],
               [2, 'L'],
               [2, 'L'],
               [3, 'L'],
               [3, 'M'],
               [3, 'M'],
               [3, 'L'],
               [3, 'L']]
    y_train = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    x_test = [[2, 'S']]

    # bayes = Bayes()
    # bayes.fit(x_train, y_train)
    # y_pre = bayes.predict(x_test, with_pro=True)
    # print('预测结果',y_pre)

    nb = NaiveBayes()
    nb.fit(x_train,y_train)
    y_pre = nb.predict(x_test, with_pro=True)
    print('预测结果',y_pre)
