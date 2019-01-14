# /usr/bin/env python3
# -*- coding:utf-8 -*-

# 参数向量
weights = []
# 偏置
b = 0
print_pattern = 'Iteration {}, 误分类点: x{}, w={}, b={}'

def update(item, learning_rate = 1):
    ''' 更新权值、偏置 '''
    global weights, b
    for i in range(len(item)):
        weights[i] += learning_rate * item[1] * item[0][i]
    b += learning_rate * item[1]

def cal(item):
    ''' 计算 yi*(w*xi + b) <= 0 '''
    res = 0
    for i in range(len(item[0])):
        res += item[0][i] * weights[i]
    res += b
    res *= item[1]
    return res

def check(iteration, learning_rate = 1):
    # 标识是否存在误分类点
    flag = False
    for xi in range(len(data_set)):
        if cal(data_set[xi]) <= 0:
            flag = True
            update(data_set[xi], learning_rate=learning_rate)
            print(print_pattern.format(iteration, xi + 1, weights, b))
    # 误分类点不存在，迭代结束
    return flag

def train(data_set, max_iteration=1000, learning_rate=1):
    ''' 感知机学习算法的原始形式 '''
    for i in range(len(data_set)):
        weights.append(0)
    iteration = 0
    print(print_pattern.format(iteration, 1, weights, b))
    while True:
        if not check(iteration) or iteration>max_iteration:
            break
        iteration += 1

if __name__ == '__main__':
    # 训练集
    data_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
    train(data_set)