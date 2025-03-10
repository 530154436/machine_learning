# /usr/bin/env python3
# -*- coding:utf-8 -*-
import time
import numpy as np
from sklearn.linear_model import Perceptron

def train():
    X = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    print('sklearn.linear_model.Perceptron')
    print('train_x={}'.format(X))
    print('train_y={}'.format(y))
    for i in range(1,11):
        start = time.time()
        learning_rate = i*0.1
        p = Perceptron(eta0=learning_rate,max_iter=5, tol=-np.infty)
        model = p.fit(X,y)
        print("learning_rate eta={}%, cost={} s".format(format(learning_rate*100, '.2f'), time.time()-start))

train()