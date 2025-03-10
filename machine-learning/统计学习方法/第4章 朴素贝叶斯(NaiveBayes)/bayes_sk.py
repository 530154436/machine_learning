# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

x2 = {'S':1, 'M':2, 'L':3}
x_train = [[1, 'S'],
           [1, 'M'],
           [1, 'M'],
           [1, 'S'],
           [1, 'S'],
           [2, 'S'],
           [2, 'M'],
           [2, 'M'],
           [2, 'L'],
           [2, 'L'],
           [3, 'L'],
           [3, 'M'],
           [3, 'M'],
           [3, 'L'],
           [3, 'L']]

x_train_trans = []
for x in x_train:
    new_x = []
    new_x.append(x[0])
    new_x.append(x2.get(x[1]))
    x_train_trans.append(new_x)
x_train_trans = np.array(x_train_trans)

y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
x_test = [[2, 1]]

for bayes in [GaussianNB(), BernoulliNB(), MultinomialNB()]:
    bayes.fit(x_train_trans, y_train)
    y_pred = bayes.predict(x_test)
    print(y_pred)
