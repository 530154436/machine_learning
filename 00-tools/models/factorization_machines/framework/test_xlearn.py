#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os
import tempfile
import pywFM
from pathlib import Path
from scipy import sparse
from fastFM import mcmc,sgd
import xlearn as xl
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file

BASE_DIR = str(Path(__file__).resolve().parent)

# os.environ['LIBFM_PATH']='/root/ai_group/libfm/bin/'
os.environ['LIBFM_PATH'] = '/Users/zhengchubin/PycharmProjects/zhihu2019/data/资料/libfm/bin/'

features = np.matrix([
    #     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
    #    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
        [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
        [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
        [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
        [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
        [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
        [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
        [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
    ])
target = [1, 1, 0, 1, 0, 1, 0]
X_tr, y_tr, X_te, y_te, = features[:5], target[:5], features[5:], target[5:]

def demo_libfm():
    # export PYTHONPATH=~/ai_group/zhihu2019
    # export LIBFM_PATH=/root/ai_group/libfm/bin/
    # os.environ['LIBFM_PATH'] = '/Users/zhengchubin/PycharmProjects/zhihu2019/data/资料/libfm/bin/'

    # features = pd.DataFrame(features)
    # from sklearn.datasets import dump_svmlight_file
    # dump_svmlight_file(features, target, '/Users/zhengchubin/Desktop/xx.svm')
    # print(features.head())

    fm = pywFM.FM(task='classification', learning_method='mcmc', num_iter=100, init_stdev=0.7,
                  k0=1, k1=1,k2=16, verbose=10)

    # split features and target for train/test
    # first 5 are train, last 2 are test
    model = fm.run(X_tr, y_tr, X_te, y_te)
    print(model.predictions, type(model.predictions))
    # you can also get the model weights
    print(model.weights)
    print(model.pairwise_interactions)

def demo_xlearn_0():
    # Generate predictions
    fm_model = xl.FMModel(task='binary', init=0.1, epoch=100, k=16, lr=0.1, reg_lambda=0.01, opt='sgd',
                          n_jobs=4, metric='auc', stop_window=10)  # log=str(BASE_DIR)+'/xlearn.log'
    fm_model.fit(X_tr, y_tr, eval_set=[X_te, y_te], is_lock_free=False)
    y_pred = fm_model.predict(X_te)
    print(y_pred, type(y_pred))

def demo_xlearn_1():
    # Generate predictions
    temp_output_file = tempfile.NamedTemporaryFile(delete=True)
    fm_model = xl.create_fm()

    # convert to libsvm format
    # converts train and test data to libSVM format
    dump_svmlight_file(X_tr, y_tr, BASE_DIR+'/tmp/train.libsvm')
    dump_svmlight_file(X_te, y_te, BASE_DIR+'/tmp/test.libsvm')

    # set training and validation data
    fm_model.setTrain(BASE_DIR+'/tmp/train.libsvm')
    fm_model.setValidate(BASE_DIR+'/tmp/test.libsvm')

    # define params and train
    param = {'task': 'binary',
             'lr': 0.1,
             'k': 16,
             'lambda': 0.0002,
             'metric': 'auc',
             'epoch': 100}
    fm_model.fit(param, BASE_DIR+'/tmp/model.out')
    fm_model.setTest(BASE_DIR+'/tmp/test.libsvm')

    # 预测
    fm_model.predict(BASE_DIR+'/tmp/model.out', temp_output_file.name)
    y_pred = pd.read_csv(temp_output_file.name, header=None)[0].values

    print(y_pred, type(y_pred))
    temp_output_file.close()

def demo_fastfm():
    fm = mcmc.FMClassification(n_iter=100, init_stdev=0.1, rank=16, random_state=123, copy_X=True)
    y_pred = fm.fit_predict_proba(sparse.csr_matrix(X_tr), np.array(y_tr), sparse.csr_matrix(X_te))
    print(y_pred, type(y_pred))

if __name__ == '__main__':
    demo_libfm()
    demo_xlearn_0()
    demo_xlearn_1()
    demo_fastfm()