# /usr/bin/env python3
# -*- coding:utf-8 -*-
from plot_all import *

TRAIN_DATA_SET = './data/testSet.txt'
PRINT_PATTERN = 'Iteration {}, Error: {}'

weights_history = []

def loadDataSet(path, separator='\t'):
    '''
    加载数据集
    :param path:
    :return: [(x0, x1, x2)], [yi], yi={0,1}
    '''
    file = open(path, mode='r', encoding='utf-8')
    data_set = []; label_set = []
    for line in file:
        line = line.strip().split(separator)
        data_set.append([1, float(line[0]), float(line[1])])
        label_set.append(int(line[2]))
    return data_set,label_set

def sigmoid(X):
    ''' sigmoid 函数 '''
    return 1.0 / (1+np.exp(-X))

def batchGradientDescent(data_mat, class_labels,
                         alpha=0.01, epsilon=0.01, max_iteration=10000):
    '''
    逻辑斯谛回归梯度上升优化算法 (批梯度上升)
    :param data_mat:输入X矩阵
          （100*3的矩阵，每一行代表一个实例，每列分别是X0 X1 X2）
    :param class_labels: 输出Y矩阵
          （类别标签组成的向量）
    :param alpha: 步长
    :return: 权值向量
    '''
    data_matrix = np.mat(data_mat, dtype=np.float32)  #转换为 NumPy 矩阵数据类型
    label_matrix = np.mat(class_labels, dtype=np.float32).transpose() # 转置
    m, n = np.shape(data_matrix)                      #矩阵大小
    weights = np.ones((n, 1))
    iteration = 0
    while True:
        h_i = sigmoid(data_matrix * weights)            # logistic函数sigmoid
        delta = label_matrix -  h_i                     # yi - h(xi)
        weights += alpha*data_matrix.transpose()* delta # w:=w+alpha*∑xi*(yi-h(xi))
        inner = 1/2*delta.transpose()*delta             # 1/2 * ∑(yi-h(xi)^2
        iteration += 1
        if iteration%4==0:
            weights_history.append(np.copy(weights))
        print(PRINT_PATTERN.format(iteration, inner))
        if inner<=epsilon or iteration>=max_iteration:
            break
    return weights

def stochasticGradientDescent(data_mat, class_labels,
                              alpha=0.01, epsilon=0.01, max_iteration=10000):
    '''
    逻辑斯谛回归梯度上升优化算法 (改进的随机梯度上升)
    :param data_mat:输入X矩阵
          （100*3的矩阵，每一行代表一个实例，每列分别是X0 X1 X2）
    :param class_labels: 输出Y矩阵
          （类别标签组成的向量）
    :param alpha: 步长
    :return: 权值向量
    '''
    data_matrix = np.mat(data_mat, dtype=np.float32)  #转换为 NumPy 矩阵数据类型
    label_matrix = np.mat(class_labels, dtype=np.float32).transpose() # 转置
    m, n = np.shape(data_matrix)                      #矩阵大小 m*n
    weights = np.ones((n, 1))
    iteration = 0
    while True:
        data_index = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+iteration+i)*0.01                        # alpha 随迭代次数更新
            randIndex = int(np.random.uniform(0, len(data_index)))  # 随机样本 服从 均匀分布
            h_i = sigmoid(np.sum(data_matrix[randIndex] * weights)) # logistic函数sigmoid
            delta = label_matrix[randIndex] - h_i                   # yi - h(xi)
            weights+=alpha*data_matrix[randIndex].transpose()*delta # w:=w+alpha*xi*(yi-h(xi))
            inner = 1/2*delta**2                                    # 1/2 * (yi-h(xi)^2
            del(data_index[randIndex])                              # 删除已经用过的样本
        if iteration%4==0:
            weights_history.append(np.copy(weights))
        iteration += 1
        print(PRINT_PATTERN.format(iteration, inner))
        if inner<=epsilon or iteration>=max_iteration:
            break
    return weights

def classifyVector(inX, weights):
    '''预测'''
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def test():
    # 加载数据集
    data_mat, class_labels = loadDataSet(TRAIN_DATA_SET)

    # sigmoid 函数
    # plotSigmoid(sigmoid)

    # 散点图
    # plotBestFit(None)

    # 批梯度上升优化算法
    # weights = batchGradientDescent(data_mat, class_labels, alpha=0.001, epsilon=0.5, max_iteration=500)
    # 动画可视化
    # plotAnimation(TRAIN_DATA_SET, weights_history, name='gradAscent.gif')


    # 随机梯度上升优化算法
    weights = stochasticGradientDescent(data_mat, class_labels, alpha=0.001, epsilon=0.0001, max_iteration=500)
    # 动画可视化
    plotAnimation(TRAIN_DATA_SET, weights_history, name='stochasticGradAscent.gif')

if __name__ == '__main__':
    test()


