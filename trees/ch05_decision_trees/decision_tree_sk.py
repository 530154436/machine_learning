# /usr/bin/env python3
# -*- coding:utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
import numpy as np

dataSet = [
    ['青年', '否', '否', '一般'],
    ['青年', '否', '否', '好'],
    ['青年', '是', '否', '好'],
    ['青年', '是', '是', '一般'],
    ['青年', '否', '否', '一般'],
    ['中年', '否', '否', '一般'],
    ['中年', '否', '否', '好'],
    ['中年', '是', '是', '好'],
    ['中年', '否', '是', '非常好'],
    ['中年', '否', '是', '非常好'],
    ['老年', '否', '是', '非常好'],
    ['老年', '否', '是', '好'],
    ['老年', '是', '否', '好'],
    ['老年', '是', '否', '非常好'],
    ['老年', '否', '否', '一般'],
]
x_train = []
fea = {'青年':1, '中年':2, '老年':3, '是':1, '否':0, '一般':1,'好':2,'非常好':3}
for sample in dataSet:
    features = []
    for i in sample:
        features.append(fea[i])
    x_train.append(features)
print('x_train:',x_train)

y_train = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
print('y_train:',y_train )

dt = DecisionTreeClassifier()
dt.fit(np.array(x_train), np.array(y_train))

print('青年A={}, 结果:{}'.format([1,0,1,1], dt.predict(np.array([[1,0,1,1]]))))
print('青年B={}, 结果:{}'.format([2,1,0,2], dt.predict(np.array([[2,1,0,2]]))))
print('青年C={}, 结果:{}'.format([3,0,1,1], dt.predict(np.array([[3,0,1,1]]))))