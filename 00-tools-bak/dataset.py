# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import sklearn.datasets as data_sets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_boston(print_data_info=False, normalization=False,test_size=0.3, random_state=1):
    '''
    the boston house-prices dataset (regression)
    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features         real, positive
    Targets           real 5. - 50.
    ==============   ==============
    '''
    x,y = data_sets.load_boston(return_X_y=True)

    if normalization:
        x = MinMaxScaler().fit_transform(x)

    if print_data_info:
        data = data_sets.load_boston()
        frame = pd.DataFrame(data.data, columns=data.feature_names)
        print(frame.head(10))
        print(frame.describe())

    x_tr,x_te,y_tr,y_te = train_test_split(x,y, test_size=test_size, random_state=random_state)
    print(f'x_train{x_tr.shape},y_train{y_tr.shape}'
          f'x_test {x_te.shape},y_test {y_te.shape}')
    return x_tr,x_te,y_tr,y_te

def load_iris(print_data_info=False, normalization=False, test_size=0.3, random_state=1):
    '''
    the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.
    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    '''
    x,y = data_sets.load_iris(return_X_y=True)

    if print_data_info:
        data = data_sets.load_iris()
        frame = pd.DataFrame(data.data, columns=data.feature_names)
        print(frame.head(10))
        print(frame.describe())

    x_tr, x_te, y_tr, y_te = train_test_split(x,y, test_size=test_size, random_state=random_state)
    print(f'x_train{x_tr.shape},y_train{y_tr.shape}'
          f'x_test {x_te.shape},y_test {y_te.shape}')
    return x_tr,x_te,y_tr,y_te

def load_breast_cancer(print_data_info=False, normalization=False, test_size=0.3, random_state=1):
    '''
    The breast cancer dataset is a classic and very easy binary classification dataset.
    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============
    '''
    x,y = data_sets.load_breast_cancer(return_X_y=True)

    if normalization:
        x = MinMaxScaler().fit_transform(x)

    if print_data_info:
        data = data_sets.load_breast_cancer()
        frame = pd.DataFrame(data.data, columns=data.feature_names)
        print(frame.head(10))
        print(frame.describe())

    x_tr, x_te, y_tr, y_te = train_test_split(x,y, test_size=test_size, random_state=random_state)
    print(f'x_train{x_tr.shape},y_train{y_tr.shape}'
          f'x_test {x_te.shape},y_test {y_te.shape}')
    return x_tr,x_te,y_tr,y_te