# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
'''
    熵、条件熵、信息增益(互信息)、ID3算法、C4.5
    李航-统计学习方法2 (p75-78)、参考 ml in action
'''
class DecisionTree(object):

    def __int__(self):
        pass

    def cal_ent(self, y:pd.Series):
        '''
        经验熵（香农熵），即对数据D的经验熵: H(D) = -∑ |C|/|D| * log_2(|C|/|D|)
        '''
        return y.value_counts(normalize=True).map(lambda p: -p * np.log2(p)).sum()

    def cal_conditional_ent(self, x:pd.Series, y:pd.Series):
        '''
        经验条件熵，即特征A对数据集D的经验条件熵: H(D|A) = ∑|Di|/|D|*H(Di)
        '''
        p = x.value_counts(normalize=True).to_dict()                            # 特征A不同特征值的概率: |Di|/|D|
        xy = pd.concat([x, y], axis=1)                                          # 拼接多列
        xy = xy.groupby(by=x.name).agg({y.name: self.cal_ent}).reset_index()    # 在Di中分别计算Y的熵: H(Di)
        return xy.apply(lambda x_ent: p.get(x_ent[0]) * x_ent[1], axis=1).sum() # H(D|X) = ∑|Di|/|D|*H(Di)

    def cal_info_gain(self, x:pd.Series, y:pd.Series, base_ent=None):
        '''
        信息增益 = 数据D的经验熵 - 特征A对数据集D的经验条件熵
        即
            g(D, A) = H(D) - H(D|A)
        '''
        c_ent = self.cal_conditional_ent(x, y)
        return  base_ent-c_ent if base_ent else self.cal_ent(y)-c_ent

    def cal_info_gain_rate(self, x:pd.Series, y:pd.Series, base_ent=None):
        '''
        信息增益率 = 信息增益/数据集D关于特征A的熵
        即
            H_A(D) = -∑ |Di|/|D| * log_2(|Di|/|D|)
            g_R(D, A) = g(D, A) / H_A(D)
        '''
        return  self.cal_info_gain(x, y, base_ent) / self.cal_ent(x)

    def majority_cnt(self, y: pd.Series):
        '''
        返回出现次数最多的分类名称
        '''
        return y.value_counts(ascending=False).index[0]  # 按cnt降序排列，并取最大计数的类别

    def choose_best_feat_to_split(self, x:pd.DataFrame, y:pd.Series, method='ID3'):
        '''
        选择最好的数据集划分方式 (ID3、C4.5算法)
        :param x:
        :param y:
        :param method:
        :return:
        '''
        base_ent = self.cal_ent(y)
        max_val = 0
        best_fea = 'None'

        # 遍历所有维度特征，选择最大的信息增益/信息增益率
        for i,col in enumerate(x.columns):
            val = 0
            if method=='C45':
                val = self.cal_info_gain_rate(x[col], y, base_ent=base_ent)
            elif method=='ID3':
                val = self.cal_info_gain(x[col], y, base_ent=base_ent)

            if val>max_val:
                max_val = val
                best_fea = col

        return best_fea

    def create_tree(self, x:pd.DataFrame, y:pd.Series, method='ID3'):
        '''
        递归创建决策树
        :param x:       特征列
        :param y:       分类标签
        :param method:  生成树算法
        '''
        if x.shape[0]!=y.shape[0]:
            raise Exception('x和y的维度不一致.')

        # (1) 数据集 D 所有实例属于同一类Ck，返回Ck
        if y.unique().size==1:
            return y.values[0]

        # (2)特征集 A 为空，返回实例数最大的类Ck
        if x.shape[1]==0:
            return self.majority_cnt(y)

        # (3)选择信息增益最大的特征 Ag
        best_feat = self.choose_best_feat_to_split(x, y, method=method)

        # 递归创建决策树
        myTree = {best_feat: {}}
        for val in x[best_feat].unique():

            # 根据Ag特征划分子集 Di，不含i列，即子集 Di 对应的特征集 A-{Ag}
            mask = x[best_feat]==val
            sub_set_x = x[mask].drop(best_feat, axis=1)
            sub_set_y = y[mask]

            # 将递归调用的结果作为当前树节点的一个分支
            myTree[best_feat][val] = self.create_tree(sub_set_x, sub_set_y, method=method)

        return myTree

if __name__ == '__main__':
    cols = ['年龄', '有工作', '有房子', '信贷情况', 'y']
    data = pd.DataFrame([
        ['青年', '否', '否', '一般', '拒绝'],
        ['青年', '否', '否', '好', '拒绝'],
        ['青年', '是', '否', '好', '同意'],
        ['青年', '是', '是', '一般', '同意'],
        ['青年', '否', '否', '一般', '拒绝'],
        ['中年', '否', '否', '一般', '拒绝'],
        ['中年', '否', '否', '好', '拒绝'],
        ['中年', '是', '是', '好', '同意'],
        ['中年', '否', '是', '非常好', '同意'],
        ['中年', '否', '是', '非常好', '同意'],
        ['老年', '否', '是', '非常好', '同意'],
        ['老年', '否', '是', '好', '同意'],
        ['老年', '是', '否', '好', '同意'],
        ['老年', '是', '否', '非常好', '同意'],
        ['老年', '否', '否', '一般', '拒绝']], columns=cols)
    mapper = {
        '青年':0, '中年':1, '老年':2,
        '一般':0, '好':1, '非常好':2,
        '是':1, '否':0,
        '同意':1, '拒绝':0
    }
    data = data.applymap(lambda x:mapper.get(x))
    dt = DecisionTree()

    # 熵的计算 p75
    print(f'信息熵 H(D)={dt.cal_ent(data["y"])}')
    for i,col_name in enumerate([x for x in data.columns if x!='y']):
        print(f'条件熵 H(D|{col_name})={dt.cal_conditional_ent(data[col_name], data["y"])},\t'
              f'信息增益 g(D, {col_name})={dt.cal_info_gain(data[col_name], data["y"])}')

    # 创建决策树 p77
    x = data[['年龄', '有工作', '有房子', '信贷情况']]
    y = data['y']
    myTreeID3 = dt.create_tree(x, y, method='ID3')
    myTreeC45 = dt.create_tree(x, y, method='C45')
    print(myTreeID3)
    print(myTreeC45)