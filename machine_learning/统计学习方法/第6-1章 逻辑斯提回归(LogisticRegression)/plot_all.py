# /usr/bin/env python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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

def plotSigmoid(sigmoid):
    ''' 画出sigmoid函数 '''
    x = np.linspace(-60, 60, 1000)
    y = sigmoid(x)
    ax = plt.subplot(2,1,1)
    ax.axis([-5,5,0,1])
    plt.xlabel('x')
    plt.ylabel('Sigmoid(x)')
    plt.plot(x, y)

    ax = plt.subplot(2, 1, 2)
    ax.axis([-50, 50, 0, 1])
    plt.xlabel('x')
    plt.ylabel('Sigmoid(x)')
    plt.plot(x, y)
    plt.show()

def plotBestFit(data_set, weights):
    '''
    画出数据集和逻辑斯谛最佳回归直线
    :param weights: None 时为散点图
    :return:
    '''
    data_mat, class_labels = loadDataSet(data_set)
    n, m = np.shape(data_mat)
    x_0=[]; y_0=[]
    x_1=[]; y_1=[]
    for i in range(n):
        if class_labels[i]==0: # 标签为0
            x_0.append(data_mat[i][1]); y_0.append(data_mat[i][2])
        else:    # 标签为1
            x_1.append(data_mat[i][1]); y_1.append(data_mat[i][2])
    # 散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_0, y_0, s=30, c='red', marker='o', label='Y=0')
    ax.scatter(x_1, y_1, s=30, c='green', marker='s', label='Y=1')
    if weights is not None:
        x = np.linspace(-3.0, 3.0, 1000)
        # 令w0*x0 + w1*x1 + w2*x2 = 0，其中x0=1，解出x1和x2的关系
        y = (-weights[0]-weights[1]*x)/weights[2]
        plt.plot(x, y)
    plt.legend()
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def drawLine(weights, line):
    '''画出回归线'''
    x = np.linspace(-5.0, 5.0, 1000)
    # 令w0*x0 + w1*x1 + w2*x2 = 0，其中x0=1，解出x1和x2的关系
    y = (-weights[0]-weights[1]*x)/weights[2]
    line.set_data(x, y)
    return line,

def initWrapper(data_set, line, ax):
    def init():
        '''initialization function: plot the background of each frame'''
        data_mat, class_labels = loadDataSet(data_set)
        n, m = np.shape(data_mat)
        x_0 = [];y_0 = []
        x_1 = [];y_1 = []
        for i in range(n):
            if class_labels[i] == 0:  # 标签为0
                x_0.append(data_mat[i][1]);y_0.append(data_mat[i][2])
            else:  # 标签为1
                x_1.append(data_mat[i][1]);y_1.append(data_mat[i][2])
        # 散点图
        ax.scatter(x_0, y_0, s=30, c='red', marker='o', label='Y=0')
        ax.scatter(x_1, y_1, s=30, c='green', marker='s', label='Y=1')
        plt.legend()
        plt.xlabel('X1'); plt.ylabel('X2')
        return drawLine([0.0000001,0.0000001,0.0000001], line)
    return init

def animateWrapper(weights_history, line):
    def animate(i):
        ''' animation function.  this is called sequentially'''
        print('生成frame: ',i+1)
        return drawLine(weights_history[i], line)
    return animate

def plotAnimation(data_set, weights_history, name='gradAscent.gif'):
    # first set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], 'b', lw=2)
    # call the animator.  blit=true means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animateWrapper(weights_history, line), init_func=initWrapper(data_set,  line, ax),
                                   frames=len(weights_history), interval=50, repeat=False, blit=True)
    # plt.show()
    # 生成时不能show()
    anim.save(name, fps=2, writer='imagemagick')