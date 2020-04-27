# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import tree

from trees.loss import SquaresError

'''
    梯度提升树(GBDT)

    李航-统计学习方法2 p171

    博客
    如果你还不了解GBDT，不妨看看这篇文章 https://mp.weixin.qq.com/s/ljC2dYfUzSJ4R5jBN_Mfdg
    机器学习算法GBDT的面试要点总结-上篇 https://www.cnblogs.com/ModifyRong/p/7744987.html
    GBDT原理与Sklearn源码分析-回归篇  https://blog.csdn.net/qq_22238533/article/details/79185969
    【机器学习】GBDT https://www.ershicimi.com/p/e6c1a7c2dca1119292b3e52cd19c27af

    GitHub https://github.com/Freemanzxp/GBDT_Simple_Tutorial/blob/master/GBDT
    pydotplus、graphviz 画图
'''
class GBDT(object):
    ''' 回归 '''

    def __init__(self, loss, M=10):
        self.loss = loss
        self.M = M
        self.f_mx = []

    def fit(self, x:np.array, y:np.array):

        # (1) 初始化
        fx0 = self.loss.init_fx0(x, y)
        self.f_mx.append(fx0)

        # (2) 对 m = 1,2,3,...,M
        for m in range(1, self.M+1):

            # (a) 对 i=1,2,..,N (样本)，计算负梯度(平方损失函数的负梯度为残差)
            y_hat = fx0 if m==1 else self.predict(x, iteration=m-1)
            rmi = -self.loss.grad(y, y_hat)

            # (b) 对 r_mi 拟合一个回归树，得到第 m 棵树的叶子节点区域 R_mj 和 c_mj (其实就是该划分区域y的均值)
            # (c) 对j=1,2,3,..,J，计算 c_mj
            #       argmin_c_{mj} = \argmin_{c} \sum_{x_i \in R_{mj}} L(y_i, f_{m-1}(x_i)+c_{mj})
            #
            #     => f_{m-1}(x_i) 对求偏导 => c_{mj} = \frac {\sum {y_i - f_{m-1}(x_i)}} {N_{mj}})
            dt = tree.DecisionTreeRegressor()
            dt.fit(x, rmi)
            print(f'm={m}, rmi={list(rmi.round(decimals=4))}, loss={self.loss.loss(y, y_hat).round(decimals=4)}')

            # 更新 f_m(x)
            self.f_mx.append(dt)

    def predict(self, x:np.array, iteration=0):

        res = self.f_mx[0]
        if iteration<=0: return res
        for m,fx in enumerate(self.f_mx[1:], start=1):
            if m > iteration:
                break
            # print(fx.predict(x))
            res += fx.predict(x)
        return res

if __name__ == '__main__':
    x = np.array([[5, 20],
                  [7, 30],
                  [21, 70],
                  [30, 60]])
    y = np.array([1.1, 1.3, 1.7, 1.8])
    gbdt = GBDT(SquaresError(), M=5)
    gbdt.fit(x, y)