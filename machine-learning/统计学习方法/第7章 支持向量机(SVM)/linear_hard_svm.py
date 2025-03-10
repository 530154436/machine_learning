# /usr/bin/env python3
# -*- coding:utf-8 -*-
from cvxopt import matrix,solvers
import matplotlib.pyplot as plt
import numpy as np
'''
    硬间隔最大化 (线性可分支持向量机)
    利用 cvxopt 求解二次规划问题(quadratic program，QP)
    注意：cvxopt.matrix([[列1][列2]])
         数值必须为浮点数
'''
def maxMarginMethod():
    P = 1/2 * matrix([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]) # (1/2)*w'*P*w
    q = matrix([0.0, 0.0, 0.0])
    G = matrix([[-3.0,-4.0,1.0], [-3.0,-3.0,1.0], [-1.0,-1.0,1.0]])
    h = matrix([-1.0,-1.0,-1.0])
    sol = solvers.qp(P,q, G=G, h=h)
    print(sol['x'])

def lagrangeDual():
    '''拉格朗日对偶'''
    P = 1 / 2 * matrix([[18.0, 21.0, -6.0], [21.0, 25.0, -7.0], [-6.0, -7.0, 2.0]])
    q = matrix([-1.0, -1.0, -1.0])
    G = matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    h = matrix([0.0, 0.0, 0.0])
    A = matrix([[1.0],[1.0],[-1.0]])
    b = matrix([0.0])
    sol = solvers.qp(P, q, G=G, h=h, A=A, b=b)

    multipliers = sol['x']
    print(multipliers)

def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y)))

def compute_b(w, X, y):
    return np.sum([y[i] - np.dot(w, X[i]) for i in range(len(X))])/len(X)

def wolfeDual():
    '''乌尔夫对偶问题'''
    X = np.array([[3.0, 3.0], [4.0, 3.0], [1.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0])
    N = X.shape[0]

    # K matrix
    K = np.array([np.dot(X[i], X[j])
                  for j in range(N)
                  for i in range(N)]).reshape((N, N))

    # object function
    P = 1/2 * matrix(np.outer(y, y) * K) # 3x3
    q = matrix(-1 * np.ones(N))          # 3x1

    # Equality constraints
    A = matrix(y, (1,N))                 # 1x3
    b = matrix(0.0)                      # 1x1

    # Inequality constraints
    G = matrix(np.diag(-1 * np.ones(N))) # 3x3
    h = matrix(np.zeros(N))              # 3x1

    # Solve the problem
    solution = solvers.qp(P, q, G=G, h=h, A=A, b=b)

    # Lagrange multipliers
    print(solution['x'])
    multipliers = np.ravel(solution['x'])

    # Support vectors have positive multipliers.
    has_positive_multiplier = multipliers > 1e-7
    sv_multipliers = multipliers[has_positive_multiplier]
    support_vectors = X[has_positive_multiplier]
    support_vectors_y = y[has_positive_multiplier]

    w = compute_w(multipliers, X, y)
    w_from_sv=compute_w(sv_multipliers, support_vectors, support_vectors_y)
    print(w)
    print(w_from_sv)

    b = compute_b(w, support_vectors, support_vectors_y)
    print(b)

    plotLine(w,b)

def plotLine(w, b):
    x = np.linspace(-3.0, 3.0, 1000)
    y = (-b - w[0] * x) / w[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([3.0, 4.0], [3.0,3.0], s=30, c='red', marker='o', label='Y=0')
    ax.scatter([1.0],[1.0],  s=30, c='green', marker='s', label='Y=1')
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    # 硬最大间隔最大化
    maxMarginMethod()
    # 对偶算法
    lagrangeDual()
    wolfeDual()