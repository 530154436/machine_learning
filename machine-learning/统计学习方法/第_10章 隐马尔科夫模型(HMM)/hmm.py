# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

class HMM():
    """
    Hidden Markov Model
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    """
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def simulate(self, T):
        '''
        观测序列生成算法
        :param T: 观测序列长度
        :return:
        '''
        def drawFrom(probs):
            ''' X～B(N，p)，根据概率分布进行多项分布抽样 '''
            return np.where(np.random.multinomial(1,probs)==1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        # 根据初始状态分布生成状态 it
        states[0] = drawFrom(self.pi)
        # 安装状态it的观测概率分布生成ot，取第it行
        observations[0] = drawFrom(self.B[states[0], :])
        for t in range(1,T):
            states[t] = drawFrom(self.A[states[t-1], :])
            observations[t] = drawFrom(self.B[states[t], :])
        return observations,states

    def forward(self, obs_seq):
        '''
        前向算法
        :param obs_seq: 观测序列
        :return:
        '''
        N = self.A.shape[0]
        T = len(obs_seq)

        alpha = np.zeros((N,T))
        alpha[:, 0] = self.pi * self.B[:, obs_seq[0]]
        for t in range(1, T):
            for i in range(N):
                alpha[i, t] = np.dot(alpha[:, t-1], self.A[:, i]) * self.B[i, obs_seq[t]]
        return np.sum(alpha[:, -1])

    def backward(self, obs_seq):
        '''
        后向算法
        :param obs_seq: 观测序列
        :return:
        '''
        N = self.A.shape[0]
        T = len(obs_seq)

        beta = np.zeros((N,T))
        beta[:,-1] = 1
        for t in reversed(range(T-1)):
            for i in range(N):
                beta[i, t] = np.sum(self.A[i, :] * beta[:,t+1 ] * self.B[:, obs_seq[t+1]])

        return np.sum(self.pi * self.B[:, obs_seq[0]] * beta[0])
