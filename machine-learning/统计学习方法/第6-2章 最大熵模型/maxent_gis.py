# /usr/bin/env python3
# -*- coding:utf-8 -*-
import math
from collections import OrderedDict

# 数据集：标签-特征集合
class MaxEnt():
    def __init__(self, eps=0.01):
        self.samples = []             # 所有训练样本
        self.x_y_count=OrderedDict()  # 特征值-标签：count(xi, yi)
        self.x_y_i = OrderedDict()    # 特征值-标签：fi
        self.n = 0                    # 最大特征数 n
        self.C = 0                    # 样本最大的特征数量
                                      # 用于求参数时的迭代，见IIS原理说明
        self.N = 0                    # 样本容量
        self.w = []                   # 权值
        self.EPS = eps                # 判断是否收敛的阈值 esp
        self.lastW = []               # 上一次迭代的权值
        self.modelExp = []            # 模型分布的特征期望值
        self.empiricalExp =[]         # 经验分布的特征期望值
        self.labels = set()           # 标签集合

    def _loadData(self, filePath, separator='\t'):
        ''' 加载数据集 '''
        file = open(filePath, 'r', encoding='utf-8')
        for line in file:
            if len(line)<2:
                continue
            sample = line.strip().split(separator)
            self.samples.append(sample)
            label = sample[0]
            self.labels.add(label)
            if len(sample[1:])>self.C:
                self.C = len(sample[1:])
            for field in sample[1:]:
                key = (field, label)
                if key not in self.x_y_count:
                    self.x_y_count[key] = 0
                self.x_y_count[key] += 1
        # 初始化
        self.init()
        for k,v in self.x_y_count.items():
            print(k,v)

    def init(self):
        '''初始化'''
        self.N = len(self.samples)
        self.n = len(self.x_y_count)
        self.w = [0.0] * self.n
        self._empirical_exp()

    def _indicator(self, xy):
        ''' 指示函数 fi = {0,1} '''
        if xy in self.x_y_count:
            return 1
        else:
            return 0

    def _cal(self, X, label):
        '''
        计算 exp(∑\lambda(i) * fi(x,y))
        '''
        sum = 0.0
        for x in X:
            xy = (x, label)
            if self._indicator(xy) == 1:
                i = self.x_y_i[xy]
                sum += self.w[i] * 1
        return math.exp(sum)

    def _pyx(self, X):
        '''
        计算 p(y|x) = 1/z(x) * exp(∑\lambda(i) * fi(x,y))
        '''
        zx = 0.0
        pyx = {}
        for label in self.labels:
            exp = self._cal(X, label)
            pyx[label] = exp
            zx += exp                        # z(x)
        for k,v in pyx.items():
            pyx[k] = v/zx                    # 归一化
        return pyx

    def _model_exp(self):
        ''' 模型分布期望 '''
        self.modelExp = [0.0] * self.n
        for sample in self.samples:
            X = sample[1:]
            pyx = self._pyx(X)
            for label, p in pyx.items():
                for x in X:
                    xy = (x, label)
                    if self._indicator(xy) == 1:
                        i = self.x_y_i[xy]
                        self.modelExp[i] += p * 1 / self.N

    def _empirical_exp(self):
        ''' 经验分布期望 '''
        self.empiricalExp = [0.0] * self.n
        for i,xy in enumerate(self.x_y_count.keys()):
            self.empiricalExp[i] = self.x_y_count[xy] / self.N
            self.x_y_i[xy] = i

    def _convergence(self):
        ''' 是否达到阈值 '''
        for w,_w in zip(self.w, self.lastW):
            if math.fabs(w - _w) >= self.EPS:
                return False
        return True

    def train(self, dataPath, maxIter=1000):
        ''' 训练 '''
        self._loadData(dataPath)
        for iter in range(1, maxIter):
            self.lastW = self.w[:]  # 复制w
            self._model_exp()       # 上次迭代的模型分布期望
            for i in range(len(self.w)):
                self.w[i] += 1.0/self.C * math.log(self.empiricalExp[i] / self.modelExp[i])
            print('iter {}, w={}'.format(iter, self.w))
            if self._convergence():
                break

    def predict(self, input):
        X = input.strip().split('\t')
        prob = self._pyx(X)
        return prob

if __name__ == "__main__":
    maxEnt = MaxEnt()
    # maxEnt._loadData('data/data.txt')
    maxEnt.train('data/data.txt')
    # play outlook temperature humidity windy
    print(maxEnt.predict("sunny\thot\thigh\tFALSE"))
    print(maxEnt.predict("overcast\thot\thigh\tFALSE"))
    print(maxEnt.predict("sunny\tcool\thigh\tTRUE"))
