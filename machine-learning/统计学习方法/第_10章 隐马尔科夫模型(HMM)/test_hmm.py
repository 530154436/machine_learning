# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from hmm import HMM

def test01():
    # 状态集合 (病情)
    STATES = ('Healthy', 'Fever')
    # 观测集合 (身体感受)
    OBSERVATIONS = ('normal', 'cold', 'dizzy')
    # 状态转移概率矩阵
    A = np.array([[0.7, 0.3],
                  [0.4, 0.6]])
    # 发射概率矩阵
    B = np.array([[0.5, 0.4, 0.1],
                  [0.1, 0.3, 0.6]])
    # 初始状态概率向量
    pi = np.array([0.6, 0.4])
    hmm = HMM(A, B, pi)
    observations, states = hmm.simulate(5)
    print([OBSERVATIONS[index] for index in observations])
    print([STATES[index] for index in states])

def test02():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    hmm = HMM(A, B, pi)

    pro = hmm.forward([0,1,0])
    # 0.130218
    print('前向算法：',pro)
    pro = hmm.backward([0, 1, 0])
    print('后向算法：',pro)

if __name__ == '__main__':
    # test01()
    test02()