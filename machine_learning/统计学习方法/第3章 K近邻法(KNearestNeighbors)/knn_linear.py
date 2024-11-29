# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from collections import Counter
class KNN():
    '''
    K近邻 线性扫描算法
    '''
    def __init__(self, x, y):
        self.x_train = x
        self.y_train = y

    def __calEuclideanDist(self, x1, x2, axis=1):
        '''
        计算欧拉距离(L2范数)
        :param x1: matrix
        :param x2: matrix
        :return: sqrt(||x1-x2||^2)
        '''
        return np.linalg.norm(x=(x1-x2), ord=2, axis=axis)

    def predict(self, x, k=1):
        '''
        训练 KNN
        :param x: 测试集
        :param k: N_k
        '''
        y = []
        if isinstance(x, list): x = np.array(x)

        for i,sample in enumerate(x):

            # 广播机制计算欧拉距离
            dist = self.__calEuclideanDist(sample, self.x_train)
            print('测试样本{}={}, 距离={}'.format(i,sample, dist))

            # 快排
            dist_idx = np.argsort(dist, kind='quicksort')[:k]

            # 多数表决
            classes = np.array([self.y_train[i] for i in dist_idx])

            # 投票统计[(label, count)]
            y_pred = Counter(classes).most_common()

            # 投票最多
            y.append(y_pred[0][0])

        return y

def test():
    x_train = [[5,4],[9,6],[4,7],[2,3],[8,1],[7,2]]
    y_train = [1,1,1,0,0,0]

    x_test = [[5,3]]

    knn = KNN(x_train, y_train)
    y_pre = knn.predict(x_test)
    print("y_pre = {}\n".format(y_pre))

if __name__ == '__main__':
    test()
