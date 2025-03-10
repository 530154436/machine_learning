#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.preprocessing import LabelEncoder

from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.models.xgb_lightgbm import XgbLightGBWrapper
from bjtu_programming.search_rs.src.param import SampleParam
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil

pandarallel.initialize()

pd.set_option('display.width', 800)  # 设置打印宽度
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 注意Key不能重复
KEY_FILE = {
    ('user_id',): 'u_agg',
    ('user_id', 'ctx_net_env'): 'u_cross_ctx_net_env_ccnt',
    ('user_id', 'ctx_time_expose_hour'): 'u_cross_ctx_time_expose_hour_ccnt',
    ('user_id', 'ctx_time_expose_day_of_week'): 'u_cross_ctx_time_expose_day_of_week_ccnt',
    ('user_id', 'ctx_time_expose_is_weekend'): 'u_cross_ctx_time_expose_is_weekend_ccnt',
    ('user_id', 'ctx_time_expose_hour_bucket'): 'u_cross_ctx_time_expose_hour_bucket_ccnt',
    ('user_id', 'ctx_time_expose_is_opening_hours'): 'u_cross_ctx_time_expose_is_opening_hours_ccnt',
    ('user_id', 'i_category1'): 'u_cross_i_category1_ccnt',
    ('user_id', 'i_category2'): 'u_cross_i_category2_ccnt',
    ('item_id',): 'i_agg'
}

CAT_LABEL_ENCODERS = {
    'user_id': LabelEncoder(),
    'u_device': LabelEncoder(),
    'u_os': LabelEncoder(),
    'u_province': LabelEncoder(),
    'u_city': LabelEncoder(),
    'u_age_prob': LabelEncoder(),
    'u_sex_prob': LabelEncoder(),
    'ctx_net_env': LabelEncoder(),

    'ctx_time_expose_hour': None,
    'ctx_time_expose_day_of_week': LabelEncoder(),
    'ctx_time_expose_is_weekend': None,
    'ctx_time_expose_hour_bucket': LabelEncoder(),
    'ctx_time_expose_is_opening_hours': None,

    'item_id': LabelEncoder(),
    'i_category1': LabelEncoder(),
    'i_category2': LabelEncoder(),
}


SELECT_COLUMNS = [
    'user_id',
    'item_id',
    # 'ctx_net_env',
    # 'ctx_time_expose_hour_bucket',
    # 'ctx_time_expose_hour',
    # 'ctx_time_expose_is_opening_hours',
    # 'ctx_time_expose_day_of_week',
    # 'ctx_time_expose_is_weekend',
    # 'ctx_cross_i_time_release_days',
    'i_category1',
    'i_category2',
    'u_device',
    'u_os',
    'u_province',
    'u_city',
    'u_age_prob',
    'u_sex_prob',
    'u_total_cnt',
    'u_click_cnt',
    'u_ctr',
    # 'u_cross_ctx_net_env_ccnt',
    # 'u_cross_ctx_time_expose_hour_ccnt',
    # 'u_cross_ctx_time_expose_day_of_week_ccnt',
    # 'u_cross_ctx_time_expose_is_weekend_ccnt',
    # 'u_cross_ctx_time_expose_hour_bucket_ccnt',
    # 'u_cross_ctx_time_expose_is_opening_hours_ccnt',
    'u_cross_i_category1_ccnt',
    'u_cross_i_category2_ccnt',
    'i_total_cnt',
    'i_click_cnt',
    'i_ctr'
]


def concat_features_by_param(_parm: SampleParam, valid):
    _loader_train = TrainLoader()
    _f_name = _loader_train.get_file_name_by_param(_parm, valid=valid)
    _df_train: pd.DataFrame = PickleUtil.load_from_pkl(f'{_f_name}.pkl')
    if 'cnt_read_duration' in _df_train.columns:
        _df_train = _df_train.drop(columns=['cnt_read_duration'])
    print('加载拼接主文件', _df_train.shape, f'{_f_name}.pkl')
    print(_df_train.head())

    _df_item = ItemLoader().load_csv()
    _df_item = _df_item[["item_id", "i_category1", "i_category2"]]
    _df_train = pd.merge(_df_train, _df_item, on='item_id', how='left')
    print('拼接特征:', _df_item.shape, ItemLoader().csv_file)

    _df_user = PickleUtil.load_from_pkl('user_info.pkl')
    _df_train = pd.merge(_df_train, _df_user, on='user_id', how='left')
    print('拼接特征:', _df_user.shape, 'user_info.pkl')

    del _df_item, _df_user
    gc.collect()

    # 待拼接的文件
    tra_file_name = _loader_train.get_file_name_by_param(_parm, valid=False)
    for keys, subfix in KEY_FILE.items():
        pkl_file = f'{tra_file_name}_{subfix}.pkl'
        _df_sub: pd.DataFrame = PickleUtil.load_from_pkl(pkl_file)
        print('拼接特征:', _df_sub.shape, pkl_file)

        _df_train = pd.merge(_df_train, _df_sub, how='left', on=list(keys))

    print('拼接特征完成', _df_train.shape)
    return _df_train


def train():
    _if_correct_samples = [0, 1]
    _if_skip_aboves = [0, 1]
    for _if_correct_sample in _if_correct_samples:
        for _if_skip_above in _if_skip_aboves:
            parm = SampleParam(if_skip_above=_if_skip_above, if_correct_sample=_if_correct_sample)
            loader_train = TrainLoader()
            f_name = loader_train.get_file_name_by_param(parm, valid=False)

            df_tra = concat_features_by_param(parm, valid=False)
            df_val = concat_features_by_param(parm, valid=True)

            for cat_feat, encoder in CAT_LABEL_ENCODERS.items():
                if not encoder:
                    continue

                print('分类变量标签编码转换:', cat_feat, df_tra[cat_feat].shape)
                encoder.fit(df_tra[cat_feat])
                # encoder.fit(df_tra[cat_feat].append(df_val[cat_feat]))

                # error处理
                df_val[cat_feat] = df_val[cat_feat].parallel_map(lambda s: -1 if s not in encoder.classes_ else s)
                encoder.classes_ = np.append(encoder.classes_, -1)

                df_tra[cat_feat] = encoder.transform(df_tra[cat_feat])
                df_val[cat_feat] = encoder.transform(df_val[cat_feat])

            PickleUtil.save2pkl(CAT_LABEL_ENCODERS, f'{f_name}_CAT_LABEL_ENCODERS.pkl')

            # 训练数据
            df_tra = MemoryOptimizer.reduce_mem_usage(df_tra)
            df_val = MemoryOptimizer.reduce_mem_usage(df_val)
            cat_feats = list(set(CAT_LABEL_ENCODERS.keys()) & set(SELECT_COLUMNS))
            # 筛选特征
            print(df_tra.columns)
            df_tra = df_tra[SELECT_COLUMNS + ['is_click']]
            df_val = df_val[SELECT_COLUMNS + ['is_click']]

            X_tra, Y_tra = df_tra.drop(columns=["is_click"]), df_tra["is_click"].values
            X_val, Y_val = df_val.drop(columns=["is_click"]), df_val["is_click"].values
            # X_tra, Y_tra = df_tra.drop(columns=["user_id", "item_id", "is_click"]), df_tra["is_click"].values
            # X_val, Y_val = df_val.drop(columns=["user_id", "item_id", "is_click"]), df_val["is_click"].values

            del df_tra, df_val
            gc.collect()
            print("训练集", X_tra.shape, Y_tra.shape)
            print("验证集", X_val.shape, Y_val.shape)

            print("模型训练中..")
            model_file = DATA_DIR.joinpath(f'{f_name}.model')
            XgbLightGBWrapper.create_classifier(X_tra, Y_tra, X_val, Y_val,
                                                model_file=model_file, cate_features=cat_feats)
            tuples = XgbLightGBWrapper.feature_importance(model_file, ignore_zero=True)
            for i in tuples:
                print(i)
            print()

            break
        break


if __name__ == '__main__':
    train()
