# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

print_pattern = 'Iteration {}, 误分类点: x{}, alphas={}, b={}'

class Perceptron():
    def __init__(self, max_iteration=1000, learning_rate=1):
        # 参数向量
        self.alphas = None
        # Gram 矩阵
        self.gram = None
        # 偏置
        self.b = 0
        # 迭代次数
        self.iteration = 0
        # 最大迭代次数
        self.max_iteration = max_iteration
        # 学习率
        self.learning_rate = learning_rate

    def __init(self, data_set):
        '''
        计算 Gram 矩阵、初始化 alphas
          gram =[[18, 21 ,6],
                 [21, 25 ,7],
                 [ 6, 7 , 2]]
        :param data_set:
        :return:
        '''
        N = len(data_set)
        self.alphas = np.zeros(N, dtype=np.float)
        self.gram = np.zeros(shape=(N, N), dtype=int)
        for i in range(len(data_set)):
            for j in range(len(data_set)):
                # 求内积 -> np.dot 点乘或矩阵乘法
                self.gram[i][j] = np.dot(data_set[i][0], data_set[j][0])
        return self.gram

    def __cal(self, i, y):
        '''
        计算距离
        :param i:
        :param y:
        :return:
        '''
        res = np.dot(self.alphas * y, self.gram[i]) + self.b
        res *= y[i]
        return res

    def __update(self, i, yi):
        '''
        更新权值、偏置
        :param item: 实例点及类别 [(x1,x2),y]
        :return:
        '''
        self.alphas[i] += self.learning_rate
        self.b += self.learning_rate * yi

    def __check(self, y):
        '''
        标识是否存在误分类点
        :param data_set:
        :return:
        '''
        flag = False
        for i in range(len(y)):
            if self.__cal(i, y) <= 0:
                flag = True
                self.__update(i, y[i])
                self.iteration += 1
                print(print_pattern.format(self.iteration, i+1, self.alphas, self.b))
        # 误分类点不存在，迭代结束
        return flag

    def train(self, data_set):
        '''
        感知机学习算法的对偶形式
        :param data_set:
        :return:
        '''
        self.__init(data_set)

        if type(data_set) is not np.ndarray:
            data_set = np.array(data_set)

        # 分类标签
        y = data_set[:, 1]

        while True:
            # 标识是否存在误分类点或是否大于最大迭代次数
            if not self.__check(y) or self.iteration > self.max_iteration:
                break

if __name__ == '__main__':
    perceptron = Perceptron()
    # 训练集
    data_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
    perceptron.train(data_set)








