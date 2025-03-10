#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc

import numpy as np
import pandas as pd

from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.config import CAT_FEATURES, SELECT_COLUMNS
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.dataset.loader_user import UserLoader
from bjtu_programming.search_rs.src.models.xgb_lightgbm import XgbLightGBWrapper
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer


def get_cross_count(x, x_mapper, y_mapper, top_n):
    x_keys = x_mapper.get(x['user_id'], [])[:top_n]
    y_keys = y_mapper.get(x['item_id'], [])[:top_n]
    return len(set(x_keys) & set(y_keys))


def concat_features(df: pd.DataFrame, _valid=False):

    df_sub = UserLoader().load_pkl()
    df = pd.merge(df, df_sub, how='left', on='user_id')

    df_sub = UserLoader().load_agg_pkl()
    df = pd.merge(df, df_sub, how='left', on='user_id')

    df_sub = ItemLoader().load_pkl()
    df = pd.merge(df, df_sub, how='left', on='item_id')

    del df_sub
    # gc.collect()
    ########################### AUC: 0.6977 ###########################
    # df_sub = UserLoader().load_cross_pkl(['user_id', 'item_id'],
    #                                      name=f'user_his_click_feat_{_valid}')
    # df_sub = df_sub[['user_id', 'item_id', 'u_hist_timestamp_diff_min']]
    # df_sub.loc[df_sub['u_hist_timestamp_diff_min'] >= 864000, 'u_hist_timestamp_diff_min'] = 1
    # df_sub.loc[df_sub['u_hist_timestamp_diff_min'] < 864000, 'u_hist_timestamp_diff_min'] = 0
    # df = pd.merge(df, df_sub, how='left', on=['user_id', 'item_id'])

    # df_sub = UserLoader().load_cross_pkl(['user_id', 'item_id'], name='item_cf')
    # df = pd.merge(df, df_sub, how='left', on=['user_id', 'item_id'])
    # df = df.fillna(0)

    # print("加载U-I标题关键词偏好交集")
    # u_click_keys = UserLoader().load_dict_pkl('title')
    # i_click_keys = ItemLoader().load_dict_pkl('title')
    # df['u_i_title_cross_top3'] = df.apply(lambda x: get_cross_count(x, u_click_keys, i_click_keys, top_n=3), axis=1)
    # df['u_i_title_cross_top5'] = df.apply(lambda x: get_cross_count(x, u_click_keys, i_click_keys, top_n=5), axis=1)
    # df['u_i_title_cross_top10'] = df.apply(lambda x: get_cross_count(x, u_click_keys, i_click_keys, top_n=10), axis=1)

    # print("加载U-I内容关键词偏好交集")
    # u_click_keys = UserLoader().load_dict_pkl('keys')
    # i_click_keys = ItemLoader().load_dict_pkl('keys')
    # df['u_i_keys_cross_top3'] = df.apply(lambda x: get_cross_count(x, u_click_keys, i_click_keys, top_n=3), axis=1)
    # df['u_i_keys_cross_top5'] = df.apply(lambda x: get_cross_count(x, u_click_keys, i_click_keys, top_n=5), axis=1)
    # df['u_i_keys_cross_top10'] = df.apply(lambda x: get_cross_count(x, u_click_keys, i_click_keys, top_n=10), axis=1)

    # u_click_keys = UserLoader().get_dict_pkl_name('keys')

    # # AUC 降低
    # df_sub = ItemLoader().load_agg_pkl()
    # df = pd.merge(df, df_sub, how='left', on='item_id')
    # df['i_click_cnt'] = df['i_click_cnt'] * np.exp(-df['ctx_cross_i_time_release_diff_days'])

    # 加载组合特征: AUC 降低 => 笛卡尔积没有？？
    # other_columns = [
    #     "i_category1", "i_category2",
    #     # 'ctx_net_env', 'ctx_time_expose_hour_bucket', 'ctx_time_expose_hour',
    #     # 'ctx_time_expose_is_opening_hours', 'ctx_time_expose_day_of_week', 'ctx_time_expose_is_weekend'
    # ]
    # user_profiles = [
    #     # 'u_os',
    #     'user_id', 'u_age_max', 'u_sex_max', 'u_device',  'u_city', 'u_province'
    # ]
    # for column in other_columns:
    #     for u_col in user_profiles:
    #         keys = [u_col, column]
    #         df_sub = UserLoader().load_cross_pkl(keys=keys)
    #         df = pd.merge(df, df_sub, how='left', on=keys)
    #         del df_sub
    #         gc.collect()

    # for keys in [('user_id', 'i_category2'), ('u_age_max', 'i_category2'),
    #              ('user_id', 'i_category1'), ('u_sex_max', 'i_category2'),
    #              ('u_province', 'i_category2')]:
    #     df_sub = UserLoader().load_cross_pkl(keys=keys)
    #     df = pd.merge(df, df_sub, how='left', on=keys)

    # 筛选特定的特征
    drop_columns = [
        # 'ctx_time_expose_hour_bucket',
        # 'ctx_time_expose_is_opening_hours',
        # 'ctx_time_expose_is_weekend',
        # 'i_title_word_len',
        # 'i_title_char_len',
        # 'u_click_ctx_timestamp_expose_mean',
        # 'u_click_ctx_timestamp_expose_std',
        # 'ctx_time_expose_day_of_week',
        # 'ctx_timestamp_expose',
        # 'u_click_ctx_timestamp_expose_min',
        # 'i_timestamp_release',
        # 'user_id',
        # 'item_id',
        # 'i_category1',
        # 'u_click_ctx_time_expose_hour_mean',
        # 'ctx_net_env',
        # 'u_sex_max',
        # 'u_age_max',
        # 'u_os',
        # 'u_click_i_keys_len_max',
        # 'u_province'
    ]
    drop_columns = list(set(drop_columns) & set(df.columns))
    df = df.drop(columns=drop_columns)

    # 特征分析
    # https: // zhuanlan.zhihu.com / p / 39377036
    df = MemoryOptimizer.reduce_mem_usage(df)
    print(df.columns)
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.head(10))
    return df


def train():
    df_tra = TrainLoader().load_pkl_by_valid(valid=False)
    df_tra = concat_features(df_tra, _valid=False)

    df_val = TrainLoader().load_pkl_by_valid(valid=True)
    df_val = concat_features(df_val, _valid=True)

    X_tra, Y_tra = df_tra.drop(columns=["is_click"]), df_tra["is_click"].values
    X_val, Y_val = df_val.drop(columns=["is_click"]), df_val["is_click"].values
    cat_features = list(set(CAT_FEATURES) & set(X_tra.columns))
    del df_tra, df_val
    gc.collect()

    X_tra = X_tra.astype('float32')
    Y_tra = Y_tra.astype('float32')
    X_val = X_val.astype('float32')
    Y_val = Y_val.astype('float32')
    print("训练集", X_tra.shape, Y_tra.shape)
    print("验证集", X_val.shape, Y_val.shape)

    print("模型训练中..")
    model_file = DATA_DIR.joinpath(f'{TrainLoader().get_csv_file_name()}.model')
    XgbLightGBWrapper.create_classifier(X_tra, Y_tra, X_val, Y_val, model_file=model_file,
                                        cate_features=cat_features, iterations=500, thread_count=4)
    XgbLightGBWrapper.plot_feature_importance(model_file)
    tuples = XgbLightGBWrapper.feature_importance(model_file, ignore_zero=True)
    best_columns = list(map(lambda x: x[0], tuples))
    for column, score in tuples:
        print(column, score)
    print("得分>0交集:", set(X_tra.columns) & set(best_columns))
    print("得分>0差集:", set(X_tra.columns) - set(best_columns))
    print()


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    train()
