# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pathlib
import pandas as pd
import numpy as np
from matrix_factorization.funk_svd import FunkSVD
from matrix_factorization.bias_svd import BiasSVD
from matrix_factorization.svdpp import SVDpp
from common import metrics
from common.encoder import MyLabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

base = pathlib.Path(__file__).parent

def test01():
    # 原始矩阵R
    x = np.array([[5.0, 3.0, 0.0, 1.0],
                  [4.0, 0.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 5.0],
                  [1.0, 0.0, 0.0, 4.0],
                  [0.0, 1.0, 5.0, 4.0]])
    f_svd = FunkSVD()
    f_svd.fit(x)

def test_svd(_type):
    # 读取、预处理数据
    data = pd.read_csv(base.joinpath('data/ml-100k/ua.base'), sep='\t')
    data.columns = ['uid','jid','rating', 'timestamp']

    uid_le, jid_le = MyLabelEncoder(ignore_unknown=True),MyLabelEncoder(ignore_unknown=True)
    uid_le.fit(data['uid'])
    jid_le.fit(data['jid'])

    data['uid'] = uid_le.transform(data['uid'])
    data['jid'] = jid_le.transform(data['jid'])

    x = data[['uid','jid', 'rating']].values

    # 训练
    n_epochs, n_factors,lr = 30, 25, 0.001
    svd = FunkSVD(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)

    if _type=='bias_svd':
        svd = BiasSVD(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)
    elif _type=='svdpp':
        svd = SVDpp(n_epochs=n_epochs, n_factors=n_factors, learning_rate=lr)

    svd.fit(x, is_triple=True)

    # 评估、测试
    data = pd.read_csv(base.joinpath('data/ml-100k/ua.test'), sep='\t')
    data.columns = ['uid', 'jid', 'rating', 'timestamp']

    data['uid'] = uid_le.transform(data['uid'])
    data['jid'] = jid_le.transform(data['jid'])
    data['y_hat'] = data.apply(lambda x:svd.predict(x['uid'], x['jid']), axis=1)
    data = data.dropna()

    print(data.head(100))
    y_true, y_hat = data['rating'].values, data['y_hat'].values
    print('MSE =', metrics.MSE(y_true, y_hat),
          'RMSE =', metrics.RMSE(y_true, y_hat))

if __name__ == '__main__':
    # test01()
    test_svd(_type='funk_svd')
    test_svd(_type='bias_svd')
    test_svd(_type='svdpp')