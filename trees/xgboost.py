# /usr/bin/env python3
# -*- coding:utf-8 -*-
import xgboost
import uuid
import numpy as np
import pandas as pd
from trees.tree import BiNode,BiTree
from trees.loss import LossFunction,CrossEntropy
# xgboost.XGBClassifier

'''
    Introduction to Boosted Trees 陈天奇 slice
    XGBoost超详细推导，终于有人讲明白了！     https://cloud.tencent.com/developer/article/1513111
    xgboost原理分析以及实践                 https://blog.csdn.net/qq_22238533/article/details/79477547
*** xgboost的推导过程及训练方法             https://mzz.pub/2019/12/13/数据挖掘/xgboost/
   【机器学习与算法】用python复现xgboost算法  https://www.pythonf.cn/read/86587
'''
class XGBNode(BiNode):

    def __init__(self, id, left=None, right=None,
                 weight=None, bst_idx=None, bst_val=None, **kwargs):
        '''
        val 格式: 非叶子结点 (特征索引, 特征值)
                 叶子结点 (None, 权重)
        '''
        super(XGBNode, self).__init__(id, left, right, **kwargs)
        # BiNode.__init__(self, id, left, right, **kwargs)

        self.weight = weight
        self.bst_idx = bst_idx
        self.bst_val = bst_val

class MyXGBoost(object):

    def __init__(self, objective:LossFunction, n_estimators=2, gamma=0, reg_lambda=1,
                 max_depth=3, base_score=0.5, min_child_weight=1, min_child_sample=1, learning_rate=0.1):
        '''
        参数说明
        :param objective:           损失函数(目标函数)
        :param n_estimators:        迭代次数(决策树个数)
        :param gamma:               正则项中，叶子节点数T的权重系数
        :param reg_lambda:          L2 正则项的权重系数
        :param max_depth:           单个决策树的最大深度
        :param base_score:          叶子节点权重的初始值，默认0.5 (经过sigmod映射的值，可以理解为一个概率值)
        :param min_child_weight:    每个叶子节点的Hessian矩阵和
        :param min_child_sample:    每个叶子节点的样本数
        :param learning_rate:       Shrinkage技术(缩减): 即学习率eta(防止过拟合)，每棵树要乘的权重系数
        '''
        self.objective = objective
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.max_depth = max_depth
        self.base_score = base_score
        self.min_child_weight = min_child_weight
        self.min_child_sample = min_child_sample
        self.learning_rate = learning_rate

        self.trees = []

    def get_split_points(self, x:np.array, idx) -> np.array:
        '''
        生成切分点集合: 由排序数组相邻元素的平均值组成
        '''
        if x.shape[0]<=1: return x[:,idx]

        f_val = np.unique(x[:, idx])                      # 去重并排序
        # split_points = np.zeros(f_val.shape[0] - 1)     # 切分点列表
        # for i in range(f_val.size - 1):
        #     split_points[i] = x[i:i + 2].mean()         # 取相邻元素的均值
        # return split_points
        return f_val

    def cal_weight(self, G:np.array, H:np.array):
        '''
        计算叶子的权重
        :param Gj: 叶子结点 j 所包含样本的一阶偏导数累加之和，是一个常量；
        :param Hj: 叶子结点 j 所包含样本的二阶偏导数累加之和，是一个常量；
        '''
        return -G/(H+self.reg_lambda)

    def cal_gain(self, G_l, G_r, H_l, H_r):
        '''
        一个叶子结点分裂为左右两个子叶子结点，需要检测这次分裂是否会给损失函数带来增益:
            (1) 增益Gain>0，即分裂为两个叶子节点后，目标函数下降了，那么我们会考虑此次分裂的结果。
            (2) 增益Gain<0时，放弃当前的分裂。
        '''
        l = G_l**2 / (H_l + self.reg_lambda)
        r = G_r**2 / (H_r + self.reg_lambda)
        l_r = (G_l + G_r)**2 / (H_l + H_r + self.reg_lambda)
        # 1.0/2*(l+r - l_r) - self.gamma
        return (l+r - l_r) - self.gamma


    def find_bst_split(self, x:np.array, y:np.array, y_hat:np.array, depth):
        '''
        寻找最佳分裂点(全局扫描法)
        在分裂一个结点时，我们会有很多个候选分割点，寻找最佳分割点的大致步骤如下：
            (1) 遍历每个结点的每个特征；
            (2) 对每个特征，按特征值大小将特征值排序；
            (3) 线性扫描，找出每个特征的最佳分裂特征值；
            (4) 在所有特征中找出最好的分裂点（分裂后增益最大的特征及特征值）

        不满足条件的情况:
            (1) 只有一个节点，则直接返回 (None,叶子权值)
            (2) 切分后数据集的样本数少于最小样本数
        '''
        # 只有一个节点，不需要分裂，直接返回
        if x.shape[0]==0: return None,None
        if x.shape[0]==1 or depth>self.max_depth:
            return None,self.cal_weight(np.sum(self.objective.grad(y, y_hat)),
                                        np.sum(self.objective.hess(y, y_hat)))
        m, n_features = x.shape
        grad = self.objective.grad(y, y_hat)
        hess = self.objective.hess(y, y_hat)

        print(f'g_i({grad.size}) = {list(np.round(grad, decimals=4))}')
        print(f'h_i({hess.size}) = {list(np.round(hess, decimals=4))}')

        # 初始值: 若没找到最佳分裂点，则返回叶子的权值
        bst_gain, bst_idx, bst_val = -np.inf, None, self.cal_weight(np.sum(grad), np.sum(hess))

        # 遍历特征
        for f_idx in range(n_features):

            # 遍历特征值
            for sp in self.get_split_points(x, f_idx):

                # 切分数据集
                mask_l, mask_r = x[:, f_idx]<sp, x[:, f_idx]>=sp

                # 切分后数据集的样本数少于最小样本数  (预剪枝)
                if np.count_nonzero(list(mask_l))<self.min_child_sample \
                        or np.count_nonzero(list(mask_r))<self.min_child_sample:
                    continue

                G_l, G_r = np.sum(grad[mask_l]), np.sum(grad[mask_r])   # 一阶导数和
                H_l, H_r = np.sum(hess[mask_l]), np.sum(hess[mask_r])   # 二阶导数和

                print('find_bst_split left:',x[:, f_idx][mask_l], grad[mask_l], G_l, H_l)
                print('find_bst_split right:', x[:, f_idx][mask_r], grad[mask_r], G_r, H_r)

                # 分裂后增益最大的特征及特征值
                gain = self.cal_gain(G_l, G_r, H_l, H_r)
                print(f'Gain {f_idx+1,sp} = {np.round(gain, decimals=4)}')

                if gain>=bst_gain:
                    bst_gain = gain
                    bst_idx = f_idx
                    bst_val = sp

        print(f'最佳分裂点: {bst_idx+1 if bst_idx!=None else bst_idx, bst_val}\n')
        return bst_idx, bst_val

    def build_tree(self, x:np.array, y:np.array, y_hat:np.array, depth, decimals=6):
        '''
        递归创建决策树
        '''
        bst_idx, bst_val = self.find_bst_split(x, y, y_hat, depth)

        # 数据集为空
        if bst_idx==None and bst_val==None:
            return None

        # 如果该节点不能再切分，将该节点存为叶节点
        if bst_idx==None:
            # shrinkage => 防止过拟合
            bst_val = bst_val * self.learning_rate
            leaf = XGBNode(str(uuid.uuid1()), weight=bst_val)
            leaf.update_kwargs({
                'leaf':np.round(bst_val, decimals=decimals),
                'num_sample': x.shape[0]
            })
            return leaf

        root = XGBNode(str(uuid.uuid1()), bst_idx=bst_idx, bst_val=bst_val)
        root.update_kwargs({
            f'X{bst_idx+1}<{bst_val.round(decimals=decimals)}':None,
            'num_sample': x.shape[0]
        })

        # 分割为左右两个数据集
        mask_l, mask_r = x[:, bst_idx] < bst_val, x[:, bst_idx] >= bst_val
        root.left = self.build_tree(x[mask_l], y[mask_l], y_hat[mask_l],  depth+1)
        root.right = self.build_tree(x[mask_r], y[mask_r], y_hat[mask_r], depth+1)

        return root

    def fit(self, x:np.array, y:np.array):

        if y.shape[0] != y.shape[0]:
            raise ValueError('X and Y must have the same length!')

        y_hat = np.zeros(y.shape)  # base_score=0.5 => \hat{y}=0
        for t in range(1, self.n_estimators+1):

            # 训练了一棵决策树
            root = self.build_tree(x, y, y_hat, depth=1)
            self.trees.append(root)

            # 更新y_hat
            y_hat = self.predict_raw(x, iteration=t)
            print(t, 'y_hat', y_hat)

            y_pred = self.predict(x, iteration=t)
            print(t, 'y_pred', y_pred)

            BiTree.plot(root, f'doc/xgboost_{t}.png')

    def walk_to_leaf(self, root:XGBNode, x_with_index:np.array, y_hat:np.array):
        '''
        单棵决策树的预测方法 f_m(x)
        :param root:            根节点
        :param x_with_index:    带索引的输入，方便定位在原始数组的下标
        :return:                权值
        '''
        if x_with_index.shape[0]<1 or root==None:
            return

        # 叶子节点 => 直接返回权值
        if root.left==None and root.right==None:
            y_hat[x_with_index[:, -1].astype(int)] = root.weight
            return

        # 遍历左右子树
        mask_l, mask_r =  x_with_index[:,root.bst_idx]<root.bst_val, x_with_index[:,root.bst_idx]>=root.bst_val
        self.walk_to_leaf(root.left, x_with_index[mask_l], y_hat)
        self.walk_to_leaf(root.right, x_with_index[mask_r], y_hat)

    def predict_raw(self, x:np.array, iteration=None):
        '''
        \hat{y_i}^{(m)} = \sum_{m=1}^M f_m(x_i)
        '''
        if iteration==None:
            iteration = self.n_estimators
        m, n = x.shape

        # 初始值
        y_hat = np.zeros(m)

        # 标记x每个样本的序号
        idx = np.zeros((m, 1))
        idx[:, 0] = range(m)
        x_with_idx = np.hstack([x, idx])

        for t in range(iteration):
            y_hat_t = np.zeros(m)
            self.walk_to_leaf(self.trees[t], x_with_idx, y_hat_t)
            y_hat += y_hat_t

        return y_hat

    def predict(self, x:np.array, iteration=None):
        '''
        ﻿y_{\text{pred}} =\text{sigmoid}(\hat{y_i})= \frac {1} {1+e^{-\hat{y_i}}}
        '''
        y_hat = self.predict_raw(x, iteration=iteration)
        y_pred = 1/(1+np.exp(-y_hat))
        return y_pred

class MyXGBRegressor(MyXGBoost):
    pass

class MyXGBClassifier(MyXGBoost):
    pass


if __name__ == '__main__':
    x_y = pd.DataFrame(
        [[1, 1, -5, 0],
         [2, 2, 5, 0],
         [3, 3, -2, 1],
         [4, 1, 2, 1],
         [5, 2, 0, 1],
         [6, 6, -5, 1],
         [7, 7, 5, 1],
         [8, 6, -2, 0],
         [9, 7, 2, 0],
         [10, 6, 0, 1],
         [11, 8, -5, 1],
         [12, 9, 5, 1],
         [13, 10, -2, 0],
         [14, 8, 2, 0],
         [15, 9, 0, 1]], columns=['ID', 'x1', 'x2', 'y'])
    x = x_y[['x1', 'x2']].values
    y = x_y['y'].values

    # xgboost.XGBClassifier()

    mxgb = MyXGBoost(objective=CrossEntropy(), n_estimators=2)
    mxgb.fit(x,y)