# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
log_pattern = '选取误分类点x{}={}, y{}={}; 更新后w={}, b={}'
class Perceptron():
    '''
    感知机学习算法的原始形式
    '''
    def __init__(self, l_rate=0.3):
        self.w = None
        self.b = 0
        self.l_rate = l_rate

    def cal(self, xi):
        '''
        计算 样本i z=x*w+b
        :param xi:  样本i
        :return: z=x*w+b
        '''
        return np.dot(xi, self.w) + self.b

    def check(self, x_train, y_train):
        '''
        检查 数据集 是否含有误分类点
        :param x_train: 训练集
        :param y_train: 标签集
        :return: 误分类点i
        '''
        for i,sample in enumerate(x_train):
            x_i = np.array(x_train[i])
            y_i = y_train[i]
            res = y_i * self.cal(x_i)
            if res<= 0: return i
        return -1

    def update(self, xi, yi):
        '''
        梯度下降 更新参数
        :param xi:  样本i
        :param yi:  标签i
        :return:
        '''
        self.w = self.w + self.l_rate*yi*xi
        self.b = self.b + self.l_rate*yi

    def fit(self, x_train, y_train):
        '''
        训练感知机模型
        :param x_train: 数据集
        :param y_train: 标签集
        :return:
        '''

        # 初始化w
        if len(x_train)<1: return
        self.w = np.array([0 for i in range(len(x_train[0]))])

        # 检查数据集中是否存在误分类点，直至训练集中没有误分类点
        i = self.check(x_train, y_train)
        while i!=-1:
            # i为误分类点，更新参数
            x_i = np.array(x_train[i])
            y_i = y_train[i]
            self.update(x_i, y_i)

            print(log_pattern.format(i, x_i, i, y_i, self.w, self.b))
            i = self.check(x_train, y_train)

    def predict(self, x_test):
        '''
        对测试集进行预测
        :param x_test: 测试集
        :return:
        '''
        y_pred = []
        for sample_i in x_test:
            z = self.cal(sample_i)
            y_pred.append(np.sign(z))
        return y_pred

if __name__ == '__main__':
    print('感知机学习算法的原始形式')
    # 训练集
    x_train = [[3, 3], [4, 3], [1, 1]]
    y_train = [1,1,-1]
    perceptron = Perceptron(l_rate=1)
    perceptron.fit(x_train, y_train)