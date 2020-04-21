# /usr/bin/env python3
# -*- coding:utf-8 -*-
import functools
import numpy as np

'''
    AdaBoost 算法
    (基于单层决策树构建弱分类器)
    李航-统计学习方法2 156
    机器学习实战 p115
'''

def great_than(x:np.array, threshold:float):
    '''
    Gm(x)=1,  x>threshold
         =-1, x<threshold；
    :param x:          输入的
    :param threshold:
    :return:
    '''
    y = np.zeros(x.size)
    y[x > threshold] = 1
    y[x < threshold] = -1
    return y

def less_than(x:np.array, threshold:float):
    '''
    Gm(x)=1,  x<threshold
         =-1, x>threshold；
    :param x:          输入的
    :param threshold:
    :return:
    '''
    y = np.zeros(x.size)
    y[x < threshold] = 1
    y[x > threshold] = -1
    return y

# 基分类器集合
__CLASSIFIERS__ = [great_than, less_than]

class Gm(object):
    '''
    阈值分类器
    统计学习方法 p158 例8.1
    '''

    def __init__(self):
        self.alpha = None      # Gm(x) 的系数
        self.pred_func = None  # 最优分类函数
        self.min_em = np.inf   # 当前迭代中最小的分类误差

    def gen_s(self, x:np.array):
        '''
        生成切分点列表: 由排序数组相邻元素的平均值组成
        '''
        x = np.unique(x)                        # 去重并排序
        thresholds = np.zeros(x.size - 1)       # 阈值列表
        for i in range(x.size - 1):
            thresholds[i] = x[i:i + 2].mean()   # 取相邻元素的均值
        return thresholds

    def cal_em(self, y, y_pred, ws:np.array):
        '''
        计算 Gm(x) 在训练集上的分类误差率:
            em = \sum_1^N P(G_m(x_i)!=y_i)
               = \sum_{G_m(x_i)!=y_i} w_mi
        :param func: 阈值分类函数: great_than、less_than
        :param y:    标签集
        :param ws:   权值分布
        :return:
        '''
        return np.sum(ws[y!=y_pred])

    def cal_alpha(self, em, decimals=4):
        '''
        计算 Gm(x) 的系数(保留4位小数)
            alpha = \frac 1 2 \cdot \log{ \frac{1-e_m} {e_m}}
        '''
        return round(np.log(1.0 / em - 1) / 2, decimals)

    def fit(self, x:np.array, y:np.array, Dm):
        '''
        训练基分类器
        :param x:   特征集
        :param y:   标签集
        :param Dm:  权值分布
        :return:
        '''
        thresholds = self.gen_s(x)     # 生成阈值列表

        for func in __CLASSIFIERS__:            # 遍历阈值分类器集合
            for threshold in thresholds:        # 遍历阈值列表
                y_pred = func(x, threshold)
                em = self.cal_em(y, y_pred, Dm) # 计算分类误差率(损失函数)

                # 最小误差率、最优分类函数
                if em<self.min_em:
                    self.min_em = em
                    self.pred_func = functools.partial(func, threshold=threshold)

        # 计算分类器系数
        self.alpha = self.cal_alpha(self.min_em, decimals=4)

class AdaBoost(object):
    ''' AdaBoost算法 '''

    def __init__(self, M=10):
        self.M = M             # 最大迭代次数
        self.classifiers = []  # 基本分类器列表(M个)

    def fit(self, x:np.array, y:np.array):
        # 初始化权值分布
        Dm = np.ones(x.size) / x.size

        # 前向分步算法
        for m in range(self.M):           # 第m次迭代

            # 第m轮迭代得到 alpha_m、G_m(x)
            # (a) 使用具有权值分布Dm的数据集学习，得到基本分类器 Gm(x)
            # (b) 计算 Gm(x) 的分类误差
            # (c) 计算 Gm(x) 的系数
            gm = Gm()
            gm.fit(x, y, Dm)

            # (d) 更新数据集的权值分布 D_{m+1}
            Dm = Dm * np.exp(-gm.alpha * y * gm.pred_func(x))
            Dm /= sum(Dm)

            # 基本分类器列表
            self.classifiers.append(gm)

            # 当误分类点=0时提前结束
            mask = (self.predict(x)!=y)
            print(f'Iter {m+1}, D_m+1={list(np.round(Dm, decimals=5))}, em={gm.min_em}, alpha={gm.alpha}, 误分类点数(f{m+1}): {sum(mask)}')
            if sum(mask)==0:
                break

    def predict(self, x:np.array) -> np.array:
        '''
        预测: 分类器线性组合
        :param x:   特征集
        '''
        fx = 0
        for cls in self.classifiers:
            fx += cls.alpha * cls.pred_func(x)
        return np.sign(fx)

if __name__ == '__main__':
    # 例8.1
    ab = AdaBoost(M=10)
    x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_train = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    ab.fit(x_train, y_train)
