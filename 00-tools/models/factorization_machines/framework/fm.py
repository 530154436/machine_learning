#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import tempfile
import xlearn as xl
import numpy as np
import pandas as pd

class XLearn_FM():
    '''
    xxx.libsvm 数据格式: label index:value index:value
    '''
    @classmethod
    def create_fm_model(cls, train_x, train_y, dev_x=None, dev_y=None, model_output=None, iterations=100,
                        thread_count=4, task='binary', k=16, lr=0.1, metric='auc', stop_window=100):
        '''
        因子分解机模型
        :param train_x:         训练集特征
        :param train_y:         训练集标签
        :param dev_x:           验证集特征
        :param dev_y:           验证集标签
        :param iterations:      迭代次数
        :param model_output:    模型保存路径
        :param thread_count:    CPU核数
        :param task:            任务类型
        :param k:               隐向量维度
        :param lr:              学习率
        :param metric:          评测函数
        :param stop_window:     Early-Stop
        :return:
        '''
        fm_model = xl.FMModel(task=task, init=0.1, k=k, lr=lr, reg_lambda=0.01, epoch=iterations, opt='sgd',
                              n_jobs=thread_count, metric=metric, stop_window=stop_window)
        if dev_x and dev_y:
            fm_model.fit(train_x, train_y, eval_set=[dev_x, dev_y], is_lock_free=False)
        else:
            fm_model.fit(train_x, train_y, is_lock_free=False)

        return fm_model

    @classmethod
    def create_fm_model_by_file(cls, train_path, model_output, valid_path=None, iterations=100,
                                thread_count=4, task='binary', k=16, lr=0.1, metric='auc', stop_window=100):
        # from sklearn.datasets import dump_svmlight_file
        # dump_svmlight_file(X_tr, y_tr, 'train.libsvm')
        # dump_svmlight_file(X_te, y_te, 'test.libsvm')
        fm_model = xl.create_fm()
        fm_model.setTrain(str(train_path))
        if valid_path:
            fm_model.setValidate(str(valid_path))
        param = {'task': task,
                 'lr': lr,
                 'k': k,
                 'lambda': 0.002,
                 'metric': metric,
                 'epoch': iterations,
                 'stop_window': stop_window,
                 'nthread': thread_count}
        fm_model.fit(param, str(model_output))
        return fm_model

    @classmethod
    def predict(cls, model_path, test_path, set_sigmoid=True) -> np.ndarray:
        fm_model = xl.create_fm()
        temp_output_file = tempfile.NamedTemporaryFile(delete=True)
        if set_sigmoid:
            fm_model.setSigmoid() # 将分数通过 setSigmoid() API 转换到（0-1）之间
        fm_model.setTest(str(test_path))

        # 预测
        fm_model.predict(str(model_path), temp_output_file.name)
        y_pred = pd.read_csv(temp_output_file.name, header=None)[0].values
        temp_output_file.close()

        return y_pred

class Libfm():

    @classmethod
    def create_fm_model(cls):
        pass

    @classmethod
    def predict(cls):
        pass
