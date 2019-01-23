# /usr/bin/env python3
# -*- coding:utf-8 -*-
from math import log
import pickle

'''
    熵、条件熵、信息增益(互信息) 李航-统计学习方法(p60-63)、参考 ml in action
'''
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

def splitDataSet(data_set, i, value):
    '''
    按照给定特征 i 划分数据集，不含i列
    :param data_set: 数据集(不含表头)
    :param i:        第i个特征
    :param value:    第i个特征的值
    :return:
    '''
    sub_set = []
    for sample in data_set:
        if sample[i] == value:
            new_sample = sample[:i]         # 取 0~i-1 列
            new_sample.extend(sample[i+1:]) # 取 i+1~-1 列
            sub_set.append(new_sample)
    return sub_set

def calcShannonEnt(data_set, feature_num=-1):
    '''
    经验熵（香农熵），即对数据D的经验熵: H(D) = -∑ |C|/|D| * log(|C|/|D|)
    :data_set: [[f1,f2,..,fn,label1],[]]
    :param feature_num:
    :return:
    '''
    N = len(data_set)    # 样本容量
    label_count = {}     # 标签-数量
    for featVec in data_set:
        cur_label = featVec[feature_num]
        if cur_label not in label_count:
            label_count[cur_label] = 0
        label_count[cur_label] += 1
    shannonEnt = 0.0
    for label,count in label_count.items():
        prob = count / N                # |C|/|D|
        prob = prob * log(prob, 2)      # |C|/|D| * log(|C|/|D|)
        shannonEnt -= prob              # 求和: -∑ |C|/|D| * log(|C|/|D|)
    return shannonEnt

def calConditionalEntropy(data_set, i):
    """
    经验条件熵，即特征A对数据集D的经验条件熵: H(D|A) = ∑|Di|/|D|*H(Di)
    :param data_set: 数据集
    :param i:        第i个特征
    :return:
    """
    feature_list = [example[i] for example in data_set]  # 取出第i列的所有特征值
    unique_values = set(feature_list)                    # 特征值去重
    conditional_ent = 0.0
    for value in unique_values:
        sub_set = splitDataSet(data_set, i, value)       # Di
        prob = len(sub_set) / float(len(data_set))       # |Di|/|D|
        prob *= calcShannonEnt(sub_set)                  # H(Di)
        conditional_ent += prob                          # 期望 ∑|Di|/|D|*H(Di)
    return conditional_ent

def calcInformationGain(data_set, base_entropy, i):
    '''
    计算信息增益
    :param data_set:       数据集
    :param base_entropy:   数据集中Y的信息熵
    :param i:              特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    '''
    return base_entropy-calConditionalEntropy(data_set, i)

def calcInformationGainRate(data_set, base_entropy, i):
    '''
    计算信息增益率
    :param data_set:       数据集
    :param base_entropy:   数据集中Y的信息熵 H(D)
    :param i:              特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    '''
    ent_A = calcShannonEnt(data_set, i)     # H_A(D)，数据集D关于特征A的值的熵
    return calcInformationGain(data_set, base_entropy, i) / ent_A

def chooseBestFeatureToSplitByID3(data_set):
    """
    选择最好的数据集划分方式 (ID3算法)
    :param data_set:
    :return:
    """
    numFeatures = len(data_set[0]) - 1   # 最后一列是分类
    info_gain_max = 0.0
    best_feature = -1
    base_ent = calcShannonEnt(data_set)
    for i in range(numFeatures):        # 遍历所有维度特征
        ig = calcInformationGain(data_set, base_ent, i)
        if ig>info_gain_max:            # 选择最大的信息增益
            info_gain_max = ig
            best_feature = i
    return best_feature                 # 返回最佳特征对应的维度

def chooseBestFeatureToSplitByC45(data_set):
    """
    选择最好的数据集划分方式 (C4.5算法)
    :param data_set:
    :return:
    """
    numFeatures = len(data_set[0]) - 1   # 最后一列是分类
    info_gain_max = 0.0
    best_feature = -1
    base_ent = calcShannonEnt(data_set)
    for i in range(numFeatures):        # 遍历所有维度特征
        ig = calcInformationGainRate(data_set, base_ent, i)
        if ig>info_gain_max:            # 选择最大的信息增益
            info_gain_max = ig
            best_feature = i
    return best_feature                 # 返回最佳特征对应的维度

def majorityCount(label_list):
    '''
    返回出现次数最多的分类名称
    :param class_list:  类列表
    :return: 出现次数最多的类名称
    '''
    label_count = {}
    for label in label_count:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    sorted_label_count \
        = sorted(label_count.items(), key=lambda x : x[1], reverse=True)
    return sorted_label_count[0][0]

def createTree( data_set, features,
                choseBestFeatureFunc=chooseBestFeatureToSplitByID3):
    '''
    递归创建决策树
    :param data_set:                数据集
    :param features:                  分类标签
    :param choseBestFeatureFunc:    选择函数(ID3、C4.5)
    :return:
    '''
    sample_labels = [ example[-1] for example in data_set ]
    sample_labels_unique = set(sample_labels)

    # (1)数据集 D 所有实例属于同一类Ck，返回Ck
    if len(sample_labels_unique) == 1:
        return sample_labels[0]

    # (2)特征集 A 为空，返回实例数最大的类Ck
    if len(data_set[0]) == 1:
        return majorityCount(sample_labels)

    # (3)选择信息增益最大的特征 Ag
    best_feature_num = choseBestFeatureFunc(data_set)
    best_feature = features[best_feature_num]

    myTree = {best_feature: {}}
    sample_features = [sample[best_feature_num] for sample in data_set]
    sample_features_unique = set(sample_features)

    for value in sample_features_unique:
        # 根据Ag特征划分子集 Di，不含i列
        sub_set = splitDataSet(data_set, best_feature_num, value)

        # 子集 Di 对应的特征集 A-{Ag}
        sub_features = features[:best_feature_num]
        sub_features.extend(features[best_feature_num+1:])
        myTree[best_feature][value] \
            = createTree(sub_set, sub_features, choseBestFeatureFunc)
    return myTree

def classify(input_tree, features, test_vec):
    '''
    预测算法，主要是根据 input_tree->特征维度i->树的遍历
    :param input_tree:  决策树
    :param featLabels:    特征标签列表
    :param test_vec:   测试数据
    :return:  分类标签
    '''
    item = [ (k,v) for k,v in input_tree.items()][0]
    first_str = item[0]     # 取特征 Ai (根节点)
    featIndex = features.index(first_str)  # 特征 Ai 对应列 i
    second_dict = item[1]   # 取孩子节点(dict)
    value = second_dict.get(test_vec[featIndex])  # 特征 Ai 对应的值
    if isinstance(value, dict) :  # 递归
        class_label = classify(value, features, test_vec)
    else:
        class_label = value
    return class_label

def saveTree(input_tree, file_path=''):
    '''
    储存决策树模型
    :param input_tree:
    :param file_path:
    :return:
    '''
    fw = open(file_path, 'wb')
    print(input_tree)
    pickle.dump(input_tree, fw)
    fw.close()
    return True


def loadTree(input_tree, file_path=''):
    '''
    加载决策树模型
    :param input_tree:
    :param file_path:
    :return:
    '''
    fr = open(file_path, 'rb')
    tree = pickle.load(fr)
    fr.close()
    return tree

def test():
    data_set, feature_names = createDataSet()
    sub_set = splitDataSet(data_set, 0, '中年')
    print(sub_set)

    shannonEnt = calcShannonEnt(data_set)
    print('信息熵 H(D) ={}\n'.format(shannonEnt))

    c0 = calConditionalEntropy(data_set, 0)
    c1 = calConditionalEntropy(data_set, 1)
    c2 = calConditionalEntropy(data_set, 2)
    c3 = calConditionalEntropy(data_set, 3)
    print('条件熵', c0, c1, c2, c3)

    g0 = calcInformationGain(data_set, shannonEnt, 0)
    g1 = calcInformationGain(data_set, shannonEnt, 1)
    g2 = calcInformationGain(data_set, shannonEnt, 2)
    g3 = calcInformationGain(data_set, shannonEnt, 3)
    print('信息增益', g0, g1, g2, g3)

    tree_id3 = createTree(data_set, feature_names, choseBestFeatureFunc=chooseBestFeatureToSplitByID3)
    print(tree_id3)

    tree_c45 = createTree(data_set, feature_names, choseBestFeatureFunc=chooseBestFeatureToSplitByC45)
    print(tree_c45)

    class_label = classify(tree_id3, feature_names, ['青年', '是', '否', '一般'])
    print(class_label)

    print(saveTree(tree_id3, file_path='decision_tree.pkl'))
    print(loadTree(tree_id3, file_path='decision_tree.pkl'))

if __name__ == '__main__':
    test()