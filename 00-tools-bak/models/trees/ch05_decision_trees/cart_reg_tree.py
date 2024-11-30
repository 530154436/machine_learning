# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

'''
    最小二乘回归树生成算法(Cart回归树)、模型树、剪枝(预剪枝、后剪枝)
    李航-统计学习方法2 p82
    机器学习实战 p160-167
'''


class CartRegression(object):

    def __init__(self, model='reg'):
        '''
        :param model: reg:回归树、model:模型树
        '''
        self.model = model
        self.tree = None

    def load_data(self, f, sep='\t', target_label='y') -> (pd.DataFrame, pd.Series):
        '''
        加载数据集格式为: x1 ... xn y
        '''
        data = pd.read_csv(f, sep=sep)
        x_cols = ['x%s'%(i+1) for i in range(data.shape[1]-1)]
        data.columns = x_cols+[target_label]
        return data[x_cols],data[target_label]

    def gen_s(self, x:np.array):
        '''
        生成切分列表: 由排序数组相邻元素的平均值组成
        '''
        x = np.unique(x)                        # 去重并排序
        thresholds = np.zeros(x.size - 1)       # 阈值列表
        for i in range(x.size - 1):
            thresholds[i] = x[i:i + 2].mean()   # 取相邻元素的均值
        return thresholds

    # ------------------------------------------------------------------------------------------------
    # 最小二乘(回归树、模型树)生成算法
    # ------------------------------------------------------------------------------------------------
    def linear_solve(self, x:pd.DataFrame, y:pd.Series):
        pass

    def loss(self, x:pd.DataFrame, y:pd.Series):
        '''
        损失函数(误差函数): 总方差
        方差(pandas无偏估计): \delta^2 = \sum \frac{(x - \mu)^2}{n - 1}
        '''
        return y.var() * (y.size-1)

    def leaf(self, x:pd.DataFrame,  y:pd.Series):
        '''
        建立叶子节点的函数: \mu = \frac{1}{n}\sum_1^n x_i
        '''
        return y.mean()

    def bin_split(self, x:pd.DataFrame, y:pd.Series, feature, val):
        '''
        用选定的对 (j,s) 划分区域: R_1(j,s) = {x|x<=s}
                                R_2(j,s) = {x|x>s}
        :param x:       特征集
        :param y:       标签集
        :param feature: 特征变量j
        :param val:     切分点s
        :return:
        '''
        mask1 = x[feature] <= val
        mask2 = x[feature] > val
        return x[mask1],y[mask1], x[mask2], y[mask2]

    def choose_best_split(self, x:pd.DataFrame, y:pd.Series, epsilon=1, min_samples=4):
        '''
        用最佳方式切分数据集、生成相应的叶节点 ( min_{s,j}[min ∑(yi-c1)^2 +  min ∑(yi-c2)^2] )
        对每个特征
            对每个特征值
                将数据集切分为成两份
                计算切分的误差
                如果当前误差小于最小误差，则将当前切分设定为最佳切分并更新最小误差
        返回最佳切分的特征和阈值

        不满足条件的情况:
            (1) y值全部相同，则直接返回 (None,y的均值)
            (2) 切分后数据集的样本数少于最小样本数
            (3) 切分数据集前的误差与切分后的最小误差之差小于阈值

        :param x:           特征集
        :param y:           标签集
        :param epsilon:     可接受的误差范围
        :param min_samples: 最小样本数
        :return:
        '''
        # y值全部相同，则直接返回
        if y.unique().size==1:
            return None, self.leaf(x, y) #exit cond 1

        loss_min = np.inf
        best_feat = None
        best_val = self.leaf(x, y)

        # 遍历切分变量、变量的所有取值
        for feat in x.columns:

            # for val in self.gen_s(x[feat])
            for val in x[feat].unique():

                # 根据val切分数据集
                x1, y1, x2, y2 = self.bin_split(x, y, feat, val)

                # 切分后数据集的样本数少于最小样本数  (预剪枝)
                if y1.size<min_samples or y2.size<min_samples:
                    continue

                # min ∑(yi-c1)^2 +  min ∑(yi-c2)^2
                cur_loss = self.loss(x1, y1) + self.loss(x2, y2)
                if cur_loss<loss_min:
                    loss_min = cur_loss
                    best_val = val
                    best_feat = feat

        # 切分数据集前的误差与切分后的最小误差之差小于阈值，则不需要分裂 (预剪枝)
        if self.loss(x, y)-loss_min<epsilon:
            return None, self.leaf(x, y) #exit cond 2

        return best_feat, best_val

    def create_tree(self, x:pd.DataFrame, y:pd.Series, epsilon=1, min_samples=4):
        '''
        递归创建决策树 (前序遍历)
        找到最佳的带切分特征:
            如果该节点不能再切分，将该节点存为叶节点
            执行二元切分，
            在左子树调用 create_tree()
            在右子树子树调用 create_tree()

        :param x:   特征集
        :param y:   标签集
        '''
        best_feat, best_val = self.choose_best_split(x,y, epsilon=epsilon, min_samples=min_samples)
        if best_feat==None: return best_val

        # 切分数据集
        x1, y1, x2, y2 = self.bin_split(x, y, best_feat, best_val)

        # 递归创建子树
        reg_tree = {'spInd':best_feat, 'spVal':best_val}
        reg_tree['left'] = self.create_tree(x1, y1)
        reg_tree['right'] = self.create_tree(x2, y2)

        self.tree = reg_tree
        return reg_tree

    # ------------------------------------------------------------------------------------------------
    # 后剪枝算法
    # ------------------------------------------------------------------------------------------------
    def is_tree(self, obj):
        '''
        判断是否为决策树
        '''
        return isinstance(obj, dict)

    def get_mean(self, tree):
        '''
        坍塌处理(后序遍历)
        '''
        if self.is_tree(tree['left']):
            tree['left'] = self.get_mean(tree['left'])

        if self.is_tree(tree['right']):
            tree['right'] = self.get_mean(tree['right'])

        return (tree['left']+tree['right'])/2.0

    def prune(self, tree, x:pd.DataFrame, y:pd.Series):
        '''
        后剪枝算法(后序遍历)

        基于已有的树切分测试数据:
            如果存在任一子集是一棵树，则在该子集递归剪枝过程；
            计算将当前两个叶节点合并后的误差；
            计算不合并的误差；
            如果合并会降低误差的话，则将叶节点合并。
        '''
        # 保证测试数据非空
        if x is None or x.shape==0:
            return self.get_mean(tree)

        x1, y1, x2, y2 = self.bin_split(x, y, tree['spInd'], tree['spVal'])

        # 遍历左子树
        if self.is_tree(tree['left']):
            tree['left'] = self.prune(tree['left'], x1, y1)

        # 遍历右子树
        if self.is_tree(tree['right']):
            tree['right'] = self.prune(tree['right'], x2, y2)

        # 左右孩子均为叶子节点
        if not self.is_tree(tree['left']) and not self.is_tree(tree['right']):

            # 合并前的预测误差(残差:实际观察值与估计值（拟合值）之间的差)
            error_no_merge = np.sum(np.power((y1 - tree['left']), 2)) + \
                             np.sum(np.power((y2 - tree['right']), 2))

            # 合并后的预测误差
            mean = (tree['left']+tree['right'])/2.0
            error_merge = np.sum(np.power((y.values-mean), 2))

            if error_merge<error_no_merge:
                print(f'Merging({tree["spInd"], round(tree["spVal"], 7), }): '
                      f'error_no_merge={round(error_no_merge, 7)}, error_merge={round(error_merge, 7)}')
                return mean
        return tree

if __name__ == '__main__':
    import json

    # 最小二乘回归树生成算法
    cart_reg = CartRegression()
    x, y = cart_reg.load_data('ex2.txt')
    print(f'x.shape={x.shape}, y.shape={y.shape}')
    tree = cart_reg.create_tree(x, y, epsilon=1, min_samples=4)
    print(json.dumps(tree))

    # 后剪枝算法
    test_x, test_y = cart_reg.load_data('ex2test.txt')
    tree = cart_reg.prune(tree, test_x, test_y)

    print(json.dumps(tree))

