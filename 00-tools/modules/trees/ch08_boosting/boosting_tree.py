# /usr/bin/env python3
# -*- coding:utf-8 -*-
import functools
import numpy as np

'''
    回归问题的提升树算法
    李航-统计学习方法2 p168
'''
class BoostingTree(object):

    def __init__(self, M=10, epsilon=0.1):
        '''
        :param M:       最大迭代次数
        :param epsilon: 允许的最大误差
        '''
        self.M = M
        self.epsilon = epsilon

        # 决策树列表
        self.funcs = []

    def gen_s(self, x:np.array):
        '''
        生成切分点列表: 由排序数组相邻元素的平均值组成
        '''
        x = np.unique(x)                        # 去重并排序
        thresholds = np.zeros(x.size - 1)       # 阈值列表
        for i in range(x.size - 1):
            thresholds[i] = x[i:i + 2].mean()   # 取相邻元素的均值
        return thresholds

    def bin_split(self, x:np.array, y:np.array, val):
        '''
        切分点 s 切分数据 => R1、R2
        :param x:       特征集
        :param y:       标签集
        :param feature: 特征变量j
        :param val:     切分点s
        :return:
        '''
        mask1 = x <= val
        mask2 = x > val
        return x[mask1],y[mask1], x[mask2], y[mask2]

    def loss(self, y:np.array):
        '''
        损失函数: 总体平方误差
        使得数据集内部平方损失函数误差达到最小的 c1=y1.mean()、c2=y2.mean()
        方差(numpy有偏估计): var = mean(abs(x - x.mean())**2)
        '''
        return y.var() * y.size

    def cal_ms(self, ss, x, y):
        '''
        计算最小损失，并求得最优切分点、c1、c2
        :param ss:  切分点集合
        :param x:   训练集
        :param y:   标签集
        '''
        min_ms = np.inf
        best_s = None
        best_c1 = None
        best_c2 = None

        for s in ss:

            # 切分数据集 R1、R2
            x1, y1, x2, y2 = self.bin_split(x, y, s)

            # min ∑(yi-c1)^2 +  min ∑(yi-c2)^2
            m = self.loss(y1)  + self.loss(y2)
            if m<min_ms:
                min_ms = m
                best_s = s
                best_c1 = y1.mean()
                best_c2 = y2.mean()

        return min_ms, best_s, best_c1, best_c2

    def _func(self, x:np.array, s:float, c1:float, c2:float):
        '''
        二分类函数(二叉回归树)
        :param x:   输入变量
        :param s:   切分点
        :param c1:  使数据集 R1 平方损失函数误差达到最小的 c1
        :param c2:  使数据集 R2 平方损失函数误差达到最小的 c2
        '''
        y = np.zeros(x.size)
        y[x < s] = c1
        y[x >= s] = c2
        return y

    def predict(self, x):
        '''
        预测(多个分类器线性组合)
        '''
        res = 0
        for fx in self.funcs:
            res += fx(x)
        return res

    def fit(self, x:np.array, y:np.array):

        # 生成切分点列表
        ss = self.gen_s(x)

        for m in range(self.M):

            # (1) 计算残差
            rm = y if len(self.funcs)==0 else y-self.predict(x)

            # (2) 拟合残差学习一个回归树(计算最小损失，并求得最优切分点)
            min_ms, s, c1, c2 = self.cal_ms(ss, x, rm)

            # (3) 更新提升树
            self.funcs.append(functools.partial(self._func, s=s, c1=c1, c2=c2))

            # 计算平方损失误差(提前终止)
            loss = self.loss(rm)
            if loss<self.epsilon: break

            print(f'Iter {m+1}, loss={np.round(loss, decimals=4)}, rm={list(np.round(rm, decimals=4))}, '
                  f'm(s)={np.round(min_ms, decimals=4)}, f{m+1}(x)={self.funcs[m].keywords}')

if __name__ == '__main__':

    # p168 例8.2
    train_samples = np.array([[1, 5.56],
                              [2, 5.70],
                              [3, 5.91],
                              [4, 6.40],
                              [5, 6.80],
                              [6, 7.05],
                              [7, 8.90],
                              [8, 8.70],
                              [9, 9.00],
                              [10, 9.05]])
    x, y = train_samples[:,0], train_samples[:,1]
    print(x)
    print(y)

    bt = BoostingTree()
    bt.fit(x, y)