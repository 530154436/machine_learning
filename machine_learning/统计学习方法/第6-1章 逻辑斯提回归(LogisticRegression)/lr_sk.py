# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression

x_train = [ [3,3,3],
            [4,3,2],
            [2,1,2],
            [1,1,1],
            [-1,0,1],
            [2,-2,1] ]
y_train = [1,1,1,0,0,0]

for alg in ['newton-cg', 'lbfgs', 'liblinear']:
    lr = LogisticRegression(solver=alg)
    lr.fit(x_train, y_train)
    print(alg, lr.predict([[1,2,-2]]))

