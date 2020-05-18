# /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    激活函数
'''
from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x):
        raise NotImplementedError

    @abstractmethod
    def hess(self):
        pass

class Sigmoid(Activation):
    '''
    逻辑斯递归激活函数
    '''
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, x):
        '''
        函数值
        \sigma(x_i) = \frac{1} {1 + e^{-x_i}}
        '''
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        """
        一阶导数
        \\frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        fn_x = self.fn(x)
        return  fn_x * (1 - fn_x)

    def hess(self):
        pass
