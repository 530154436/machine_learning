# /usr/bin/env python3
# -*- coding:utf-8 -*-
from math import log

'''
    熵、条件熵、信息增益(互信息) 李航-统计学习方法(p60-63)(自己编写)
'''

def cal(label_count):
    '''
    熵基本计算公式
    :return: N: 样本容量
    '''
    N = 0
    if isinstance(label_count, dict):
        for count in label_count.values():
            N += count
    shannonEnt = 0
    for label,count in label_count.items():
        prob = count / N                # |C|/|D|
        prob = prob * log(prob, 2)      # |C|/|D| * log(|C|/|D|)
        shannonEnt -= prob              # 求和: -∑ |C|/|D| * log(|C|/|D|)
    return shannonEnt,N

def splitDataSet(data_set, feature_num=-1, feature_names=None):
    '''
    默认统计标签列： {标签1:num, 标签2:num}
    数据集切分成 {特征名1:{特征值1:{标签1:num, 标签2:num}，特征值2:{}}}
    :param data_set:       数据集
    :param feature_names:  特证名(表头)
    :return:
    '''
    feature_sample = {}
    for sample in data_set:
        label = sample[feature_num]
        if feature_names is None:
            if label not in feature_sample:
                feature_sample[label] = 0
            feature_sample[label] += 1
        else:
            for i in range(len(feature_names)):
                feature_name = feature_names[i]
                value = sample[i]
                if feature_name not in feature_sample:
                    feature_sample[feature_name] = {}
                if value not in feature_sample[feature_name]:
                    feature_sample[feature_name][value] = {}
                if label not in feature_sample[feature_name][value]:
                    feature_sample[feature_name][value][label] = 0
                feature_sample[feature_name][value][label] += 1
    for feature, value in feature_sample.items():
        print(feature, value)
    print()
    return feature_sample

def calcShannonEnt(data_set, feature_num=-1):
    '''
    经验熵（香农熵），即对数据D的经验熵: H(D)
    '''
    label_count = splitDataSet(data_set, feature_num)     # 标签-数量
    shannonEnt,N = cal(label_count)
    print('信息熵 H(D) ={}\n'.format(shannonEnt))
    return shannonEnt

def calConditionalEntropy(data_set, feature_names):
    """
    经验条件熵，即特征A对数据集D的经验条件熵: H(D|A)
    :param data_set:      数据集
    :param feature_names: 特证名(表头)
    :return:
    """
    feature_sample = splitDataSet(data_set, feature_names=feature_names)
    N = len(data_set)
    feature_shannonEnt = {}
    for feature, values in feature_sample.items():
        for value,labels in values.items():
            shannonEnt, n = cal(labels)
            if feature not in feature_shannonEnt:
                feature_shannonEnt[feature] = 0
            feature_shannonEnt[feature] += shannonEnt*n/N  # ∑ |D1|/|D|*(-∑H(D))
    print('条件熵 H(Di): {}\n'.format(feature_shannonEnt))
    return feature_shannonEnt

def calcInformationGain(data_set, feature_names):
    '''
    计算信息增益
    :param data_set:       数据集
    :param feature_names:  特证名(表头)
    '''
    shannonEnt = calcShannonEnt(data_set)
    feature_shannonEnt = calConditionalEntropy(data_set, feature_names)
    ig = { feature:shannonEnt-ent for feature, ent in feature_shannonEnt.items()}
    print('信息增益 H(D)-H(Di): {}\n'.format(ig))
    return ig

def calcFeatureEntropy(data_set, feature_names):
    '''
    计算数据集关于特征A的值的熵
    '''
    feature_ent = {}
    for i in range(len(feature_names)):
        ent = calcShannonEnt(data_set, feature_num=i) # -∑|D1|/|D|*log(|D1|/|D|)
        feature = feature_names[i]
        feature_ent[feature] = ent
    print('数据集关于特征A的值的熵 H_A(D): {}'.format(feature_ent))
    return feature_ent

def calcInformationGainRate(data_set, feature_names):
    """
    计算信息增益比
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    ig = calcInformationGain(data_set, feature_names)
    feature_ent = calcFeatureEntropy(data_set, feature_names)
    ig_rate = {}
    for key in feature_names:
        ig_rate[key] = ig.get(key) / feature_ent.get(key)   # H(D)-H(Di))/H_A(D)
    print('信息增益比 H(D)-H(Di))/H_A(D): {}\n'.format(ig_rate))
    return ig_rate


def createDataSet():
    """创建数据集"""
    dataSet = [
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
               ['老年', '否', '否', '一般', '拒绝'],
               ]
    feature_names = ['年龄', '有工作', '有房子', '信贷情况']
    # 返回数据集和每个维度的名称
    return dataSet, feature_names

def test():
    data_set = createDataSet()

    # 熵
    # calcShannonEnt(data_set[0])

    # 条件熵
    # calConditionalEntropy(data_set[0], data_set[1])

    # 信息增益
    calcInformationGain(data_set[0], data_set[1])

    # 信息增益比
    calcInformationGainRate(data_set[0], data_set[1])

if __name__ == '__main__':
    test()