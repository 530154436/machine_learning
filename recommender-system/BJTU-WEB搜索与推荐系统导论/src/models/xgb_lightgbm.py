#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import traceback
from typing import Tuple, List

import numpy as np
import lightgbm as lgb
import matplotlib.pylab as plt
from lightgbm import LGBMClassifier
from matplotlib.axes import Axes
from scipy import sparse


class XgbLightGBWrapper(object):
    @classmethod
    def create_classifier(cls, train_x, train_y, dev_x=None, dev_y=None, cate_features=None, iterations=100,
                          model_file='model.txt', device='cpu', task='binary', thread_count=16):
        '''
        LGBM 模型，可直接输入类别变量

        categorical_feature：
            list of strings or int, or 'auto', optional (default='auto'), only supports categorical with int type
            需要先做label encoding。用特定算法（On Grouping for Maximum Homogeneity）
            找到optimal split，效果优于ONE。 也可以选择采用one-hot encoding。

        # GPU
        # brew install libomp
        !python -m pip uninstall lightgbm
        !python -m pip install lightgbm --install-option=--gpu

        https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py#L82-L84
        https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
        '''
        if dev_x is None: dev_x = train_x
        if dev_y is None: dev_y = train_y
        if cate_features is None: cate_features = 'auto'

        # num_leaves的值不超过2 ^ (max_depth)
        model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=128, learning_rate=0.05, n_estimators=iterations,
                                   max_bin=255, subsample_for_bin=50000, objective=task, min_split_gain=0,
                                   min_child_weight=1, min_child_samples=100, subsample=0.9, subsample_freq=1,
                                   colsample_bytree=1, reg_alpha=4, reg_lambda=4, max_depth=7,
                                   seed=100, n_jobs=thread_count, silent=True, device=device)
        model_lgb.fit(train_x, train_y,
                      eval_names=['eval'],
                      eval_metric=['logloss', 'auc'],
                      eval_set=[(dev_x, dev_y)],
                      early_stopping_rounds=30,
                      categorical_feature=cate_features)
        if model_file:
            try:
                model_lgb.booster_.save_model(model_file)
                print(f'模型保存成功 {model_file}.')
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print(f'模型保存失败 {model_file}.')
        return model_lgb

    @classmethod
    def create_model(cls, train_x, train_y, dev_x=None, dev_y=None, cate_features=None, iterations=100,
                     model_file='model.txt', device='cpu', thread_count=16, params:dict=None):
        '''
        官方文档: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        调参:    https://www.cnblogs.com/bjwu/p/9307344.html
        '''
        if dev_x is None: dev_x = train_x
        if dev_y is None: dev_y = train_y
        if cate_features is None: cate_features='auto'
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(dev_x, dev_y, reference=lgb_train)
        default_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'num_leaves': 64,
            'num_trees': 100,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'early_stopping_round': None,
            'num_threads': thread_count,
            'device_type':device,
            'random_seed':1000
        }
        if params is not None:
            default_params.update(params)

        print('Start training...')
        gbm = lgb.train(default_params, lgb_train, num_boost_round=iterations, valid_sets=lgb_eval,
                        categorical_feature=cate_features)
        if model_file:
            gbm.save_model(model_file)
        return gbm

    @classmethod
    def predict(cls, model_file:str, x, pred_leaf=False, num_leaves=64):
        '''
        预测
        :param model:          gbdt模型
        :param x:       测试数据集
        :param predict_leaf: 获取叶子的预测结果 => 为模型融合作准备
        :return:
        '''
        model = lgb.Booster(model_file=model_file)

        if pred_leaf:
            # 预测每一条训练数据落在了每棵树的哪个叶子结点上 => shape = (n,num_trees) => [leaf_i] 0<=leaf_i<=num_leaves
            y_pred = model.predict(x, pred_leaf=pred_leaf)
            print('y_pred:', y_pred.shape, y_pred.min(), y_pred.max())
            # transformed_matrix = np.zeros((y_pred.shape[0], y_pred.shape[1] * num_leaf), dtype=np.int32)
            transformed_matrix = sparse.lil_matrix((y_pred.shape[0], y_pred.shape[1] * num_leaves), dtype=np.int32)
            for i in range(0, y_pred.shape[0]):
                if i % 10000 == 0: print('gbdt leaf trans2sparse', i)
                temp = np.arange(y_pred.shape[1]) * num_leaves + np.array(y_pred[i])
                transformed_matrix[i, temp] = 1
            transformed_matrix = transformed_matrix.tocsr()
            print(f'transformed_matrix.shape=({transformed_matrix.shape})')
            return transformed_matrix

        y_pred = model.predict(x)
        print(f'y_pred.shape={y_pred.shape}, {type(y_pred)},{y_pred[:5]}')

        return y_pred

    @classmethod
    def plot_feature_importance(cls, model_file:str, max_num_features=30):
        '''
        预测
        :param model:          gbdt模型
        :return:
        '''
        model = lgb.Booster(model_file=model_file)
        plt.figure(figsize=(60, 30))
        ax: Axes = lgb.plot_importance(model, max_num_features=max_num_features)
        ax.set_title("Feature Importance")
        plt.subplots_adjust(left=0.4)
        plt.savefig(r"LightGBMFeatureImportance.png")
        plt.show()

    @classmethod
    def feature_importance(cls, model_file:str, max_num_features=None, ignore_zero: bool = False) -> List[Tuple[str, float]]:
        '''
        预测
        :param model:          gbdt模型
        :return: (feature, score)
        '''
        booster = lgb.Booster(model_file=model_file)
        importance = booster.feature_importance()
        feature_name = booster.feature_name()

        tuples = sorted(zip(feature_name, importance), key=lambda x: x[1], reverse=True)
        if ignore_zero:
            tuples = [x for x in tuples if x[1] > 0]
        if max_num_features is not None and max_num_features > 0:
            tuples = tuples[:max_num_features]
        return tuples
