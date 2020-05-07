# /usr/bin/env python3
# -*- coding:utf-8 -*-
import pathlib
import pandas as pd
import numpy as np
from matrix_factorization.funk_svd import FunkSVD
from matrix_factorization.bias_svd import BiasSVD
from common import metrics

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

def test_svd(biased:bool):
    # 读数据
    data = pd.read_csv(base.joinpath('data/ml-100k/u1.base'), sep='\t')
    data.columns = ['uid','jid','rating', 'timestamp']
    x = data[['uid','jid', 'rating']].values

    # 训练
    if biased:
        svd = BiasSVD(n_epochs=100, n_factors=25, learning_rate=0.005)
    else:
        svd = FunkSVD(n_epochs=100, n_factors=100, learning_rate=0.001)
    svd.fit(x, is_triple=True)

    # 测试
    data = pd.read_csv(base.joinpath('data/ml-100k/u1.test'), sep='\t')
    data.columns = ['uid', 'jid', 'rating', 'timestamp']
    data['uid'] = data['uid'].map(lambda x: svd.u_mapping.get(x))
    data['jid'] = data['jid'].map(lambda x: svd.j_mapping.get(x))

    data = data.dropna()
    data['uid'] = data['uid'].astype(int)
    data['jid'] = data['jid'].astype(int)

    data['y_hat'] = data.apply(lambda x:svd.predict(x['uid'], x['jid']), axis=1)

    print(data.head(100))
    y_true, y_hat = data['rating'], data['y_hat'].values
    print('MSE =', metrics.MSE(y_true, y_hat),
          'RMSE =', metrics.RMSE(y_true, y_hat))

if __name__ == '__main__':
    # test01()
    # test_svd(biased=False)
    test_svd(biased=True)