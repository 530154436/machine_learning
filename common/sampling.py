# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

'''
集成所有采样算法
    [x] 自助采样法(bootstrap)
'''

def bootstrap(x: np.array, y: np.array, seed=10):
    '''
    自助采样法:
    一种从给定训练集中有放回的均匀抽样，即每当选中一个样本，它等可能地被再次选中并被再次添加到训练集中。

    https://wzdnzd.github.io/articles/20190427/%E8%87%AA%E5%8A%A9%E6%B3%95/
    '''
    m = x.shape[0]
    indices = np.random.RandomState(seed).randint(low=0, high=m, size=m) # 有放回采样(可重复) 0~m-1
    # indices = np.random.choice(m, size=m, replace=True)
    return x[indices,:], y[indices]