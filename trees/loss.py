# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

'''
    损失函数工具包 (i,j)=(第i个样本,第j个特征), N为样本容量
'''

class LossFunction(object):

    def __init__(self):
        pass

    def arc_base_score(self):
        pass

    def grad(self, y:np.array, y_hat:np.array) -> np.array:
        pass

    def hess(self, y:np.array, y_hat:np.array) -> np.array:
        pass

class SquaresError(LossFunction):
    '''
    平方损失函数:
        \hat{y_i} = F_m(x_i) = \sum_{i=1}^m f_m(x_i)
        L(y_i, \hat{y_i}) =  1/2 (y_i-\hat{y_i})^2

        Loss = \sum_{i=1}^N L(y_i, \hat{y_i})
    '''
    def __init__(self):
        LossFunction.__init__(SquaresError)

    def init_fx0(self, x:np.array, y:np.array):
        '''
        f_0(x) = argmin_c \sum {L(yi, c)} =对c求偏导=> c = \frac {\sum y_i} {N}
        '''
        return y.mean()

    def grad(self, y:np.array, y_hat:np.array):
        '''
        损失函数的梯度
        \frac {\partial{L(y_i, \hat{y_i})}} {\partial{\hat{y_i}}} = y_i-\hat{y_i}
        '''
        return -(y - y_hat)

    def loss(self, y:np.array, y_hat:np.array):
        '''
        损失函数的值
        '''
        return 1/2*np.sum(np.square(y - y_hat))

class CrossEntropy(LossFunction):
    '''
    交叉熵损失函数(二分类)
    z = \sum_j w_j \cdot x_{ij}
    \hat{y_i} = p(Y=1|x_i) = \frac {e^{z}} {1+e^{z} = \frac {1} {1+e^{-z} \\
                p(Y=0|x_i) = \frac {1} {1+e^{z}
    L(y_i, \hat{y_i}) = -[y_i\log{\hat{y_i}} + (1-y_i)\log{1-\hat{y_i}}]

    [X]GBDT
    $$
    \hat{y_i}^{(m)} = \sum_{i=1}^M f_m(x_i) ，  f_m(x_i) \in (-\inf, +\inf) \\ (前m颗决策树的输出，定义域 \in (-inf, +inf))
    y_{pred} = \frac {1} {1+e^{-\hat{y_i}}}
    $$

    $$
    \begin{aligned}
    L(y_i, \hat{y_i})
    &= -[y_i\log{y_{pred}} + (1-y_i)\log(1-y_{pred})] \\
    &= -[y_i\log{\frac {1} {1+e^{-\hat{y_i}}}} + (1-y_i)\log{\frac {1} {1+e^{\hat{y_i}}}}] \\
    \end{aligned}
    $$
    '''
    def __init__(self):
        LossFunction.__init__(CrossEntropy)

    def sigmiod(self, y_hat: np.array):
        '''
        \hat{y_i}^{(m)} = \sum_{i=1}^M f_m(x_i)
        y_{pred} = \frac {1} {1+e^{-\hat{y_i}}}
        '''
        return 1/(1+np.exp(-y_hat))

    def arc_sigmoid(self, y):
        '''
        sigmoid的反函数
        '''
        return np.log(y / (1 - y))

    def grad(self, y: np.array, y_hat: np.array):
        '''
        损失函数的一阶导数

        $$
        \begin{aligned}
        \frac {\partial{L(y_i, \hat{y_i})}} {\partial{\hat{y_i}}}
        &= y_i\frac{-e^{-\hat{y}_i}}{1+e^{-\hat{y}_i}}+(1-y_i)\frac{e^{\hat{y}_i}}{1+e^{\hat{y}_i}} \\
        &= y_i (\frac{1}{1+e^{-\hat{y}_i}}-1) + (1-y_i)\frac{1}{1+e^{-\hat{y}_i}} \\
        &= y_i(y_{\text{pred}}-1) + (1-y_i)y_{\text{pred}} \\
        &= y_{\text{pred}}-y_i
        \end{aligned}
        $$
        '''
        return self.sigmiod(y_hat)-y

    def hess(self, y: np.array, y_hat: np.array):
        '''
        损失函数的二阶导数

        $$
        \begin{aligned}
        \frac {\partial^2{L(y_i, \hat{y_i})}} {\partial{\hat{y_i}^2}}
        &= \frac {\partial} {\partial{\hat{y_i}}} (y_{\text{pred}}-y_i) \\
        &= \frac {\partial} {\partial{\hat{y_i}}} (\frac {1} {1+e^{-\hat{y_i}}} - y_i) \\
        &= (1-y_{\text{pred}}) \cdot y_{\text{pred}}
        \end{aligned}
        $$
        '''
        return (1-self.sigmiod(y_hat)) * self.sigmiod(y_hat)

    def loss(self, y:np.array, y_hat:np.array):
        '''
        损失函数的值

        $$
        \begin{aligned}
        Loss
        &= \sum_{i=1}^N L(y_i, y_{\text{pred}}) \\
        &= -\sum_{i=1}^N [y_i\log{y_{\text{pred}}} + (1-y_i)\log(1-y_{\text{pred}})] \\
        \end{aligned}
        $$
        '''
        return np.sum(-(np.multiply(y, np.log(self.sigmiod(y_hat))) + np.multiply(1-y, np.log(1-self.sigmiod(y_hat)))))