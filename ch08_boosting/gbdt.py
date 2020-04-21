# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import tree

'''
    梯度提升树(GBDT)

    李航-统计学习方法2 p171

    博客 https://mp.weixin.qq.com/s/ljC2dYfUzSJ4R5jBN_Mfdg
    GitHub https://github.com/Freemanzxp/GBDT_Simple_Tutorial/blob/master/GBDT
    pydotplus、graphviz 画图
'''

class SquaresError(object):
    '''
    平方损失函数(第i个样本):
        L(yi, f(x_i)) =  \frac {1} {2} (yi-f(x_i))^2
        Loss = \sum_{i=1}^N {L(yi, f(x_i))}
    '''

    def __init__(self):
        pass

    def init_fx0(self, x:np.array, y:np.array):
        '''
        f_0(x) = argmin_c \sum {L(yi, c)} =对c求偏导=> c = \frac {\sum y_i} {N}
        '''
        return y.mean()

    def grad(self, y:np.array, y_pred:np.array):
        '''
        损失函数的梯度(第i个样本): yi-f(xi)
        '''
        return -(y-y_pred)

class GBDT(object):
    ''' 回归 '''

    def __init__(self, loss, M=10):
        self.loss = loss
        self.M = M
        self.f_mx = []

    def fit(self, x:np.array, y:np.array):

        # (1) 初始化
        fx0 = self.loss.init_fx0(x, y)

        # (2) 对 m = 1,2,3,...,M
        for m in range(1, self.M+1):

            # (a) 对 i=1,2,..,N (样本)，计算负梯度(平方损失函数的负梯度为残差)
            y_pred = fx0 if m==1 else self.predict(x, iteration=m-1)
            rmi = -self.loss.grad(y, y_pred)

            # (b) 对 r_mi 拟合一个回归树，得到第 m 棵树的叶子节点区域 R_mj 和 c_mj (其实就是该划分区域y的均值)
            # (c) 对j=1,2,3,..,J，计算 c_mj
            #       argmin_c_{mj} = \argmin_{c} \sum_{x_i \in R_{mj}} L(y_i, f_{m-1}(x_i)+c_{mj})
            #
            #     => f_{m-1}(x_i) 对求偏导 => c_{mj} = \frac {\sum {y_i - f_{m-1}(x_i)}} {N_{mj}})
            dt = tree.DecisionTreeRegressor()
            dt.fit(x, rmi)

            print(f'm={m}, rmi={list(rmi.round(decimals=4))}')

            # 更新 f_m(x)
            self.f_mx.append(dt)

    def predict(self, x:np.array, iteration=None):

        if not iteration or iteration<=0:
            iteration = self.M

        res = 0
        for m,fx in enumerate(self.f_mx, start=1):
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