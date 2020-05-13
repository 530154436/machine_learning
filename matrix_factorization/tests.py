# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pathlib
import pandas as pd
import numpy as np
from matrix_factorization.funk_svd import FunkSVD, DataSet
from matrix_factorization.bias_svd import BiasSVD
from matrix_factorization.svdpp import SVDpp
from matrix_factorization.nmf import NMF
from tools import metrics
from tools.encoder import MyLabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

base = pathlib.Path(__file__).parent

uid_le, jid_le = MyLabelEncoder(ignore_unknown=True), MyLabelEncoder(ignore_unknown=True)

def get_train_set():
    # 读取、预处理数据
    data = pd.read_csv(base.joinpath('data/ml-100k/ua.base'), sep='\t')
    data.columns = ['uid','jid','rating', 'timestamp']

    uid_le.fit(data['uid'])
    jid_le.fit(data['jid'])

    data['uid'] = uid_le.transform(data['uid'])
    data['jid'] = jid_le.transform(data['jid'])

    x = data[['uid','jid', 'rating']].values
    train_set = DataSet(is_triple=True).create(x)

    return train_set

def test_svd(_type, train_set):

    # 训练
    n_epochs, n_factors,lr = 20, 25, 0.005

    svd = FunkSVD(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)
    if _type=='bias_svd':
        svd = BiasSVD(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)
    elif _type=='svdpp':
        svd = SVDpp(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)
    elif _type=='nmf':
        svd = NMF(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)

    svd.fit(train_set)

    # 评估、测试
    data = pd.read_csv(base.joinpath('data/ml-100k/ua.test'), sep='\t')
    data.columns = ['uid', 'jid', 'rating', 'timestamp']

    data['uid'] = uid_le.transform(data['uid'])
    data['jid'] = jid_le.transform(data['jid'])

    data['y_hat'] = data.apply(lambda x:svd.predict(x['uid'], x['jid']), axis=1)
    data = data.dropna()

    print(data.head(100))
    y_true, y_hat = data['rating'].values, data['y_hat'].values
    print(f'Test{data.shape} {_type}'
          'MSE =', metrics.MSE(y_true, y_hat),
          'RMSE =', metrics.RMSE(y_true, y_hat))

if __name__ == '__main__':
    train_set = get_train_set()
    test_svd(_type='funk_svd', train_set=train_set)
    test_svd(_type='bias_svd', train_set=train_set)
    test_svd(_type='svdpp', train_set=train_set)
    test_svd(_type='nmf', train_set=train_set)