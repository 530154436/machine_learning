#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.dataset.loader_user import UserLoader
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pd_util import PandasUtil
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


USER_LABEL_ENCODER = {
    'u_device': LabelEncoder(),
    'u_os': LabelEncoder(),
    'u_province': LabelEncoder(),
    'u_city': LabelEncoder(),
    'u_age_max': LabelEncoder(),
    'u_sex_max': LabelEncoder(),
}


def gen_user_static_feature():

    def _max_prob(x: str):
        if pd.isna(x):
            return None
        max_label, max_prob = None, 0
        for item in str(x).split(','):
            if float(item.split(':')[1]) > max_prob:
                max_label = item.split(':')[0]
        return max_label

    def _str_to_probs(x: str):
        if pd.isna(x):
            return []
        d = OrderedDict()  # 有脏数据
        for item in str(x).split(','):
            d[item.split(':')[0]] = float(item.split(':')[1])
        return list(d.values())

    df_user = UserLoader().load_csv()
    print("处理性别、年龄...")
    # df_user['u_age_probs'] = df_user['u_age_prob'].parallel_apply(_str_to_probs)
    # df_user['u_sex_probs'] = df_user['u_sex_prob'].parallel_apply(_str_to_probs)
    # PandasUtil.split_col(df_user, 'u_age_probs', _type='float32')
    # PandasUtil.split_col(df_user, 'u_sex_probs', _type='float32')
    df_user['u_age_max'] = df_user['u_age_prob'].parallel_apply(_max_prob)
    df_user['u_sex_max'] = df_user['u_sex_prob'].parallel_apply(_max_prob)
    df_user.drop(columns=['u_sex_prob', 'u_age_prob'], inplace=True)
    # df_user.drop(columns=['u_age_prob', 'u_age_probs', 'u_sex_prob', 'u_sex_probs'], inplace=True)

    for cat_feat, encoder in USER_LABEL_ENCODER.items():
        df_user[cat_feat] = encoder.fit_transform(df_user[cat_feat])
    df_user = MemoryOptimizer.reduce_mem_usage(df_user)
    print(df_user.head())
    PickleUtil.save2pkl(df_user, UserLoader().get_pkl_name())
    PickleUtil.save2pkl(USER_LABEL_ENCODER, UserLoader().get_label_encoder_pkl_name())
    del df_user
    gc.collect()


def gen_user_agg_feature():
    df_tra: pd.DataFrame = TrainLoader().load_pkl_by_valid(valid=False)

    print("生成用户-统计特征：用户是否点击新闻的统计值")
    print('用户数:', df_tra['user_id'].nunique())
    print('物品数:', df_tra['item_id'].nunique())
    df_agg = df_tra.groupby(by=['user_id']).agg(
        u_total_cnt=pd.NamedAgg(column="is_click", aggfunc="count"),
        u_click_cnt=pd.NamedAgg(column="is_click", aggfunc="sum"),
    )
    df_agg['u_ctr'] = df_agg['u_click_cnt'] / df_agg['u_total_cnt']
    df_agg = df_agg.reset_index()

    print("生成用户-统计特征: ", df_agg.shape)
    print(df_agg.head())
    df_agg = MemoryOptimizer.reduce_mem_usage(df_agg)
    PickleUtil.save2pkl(df_agg, UserLoader().get_agg_pkl_name())

    print("生成用户点击-交叉特征：用户点击新闻的 曝光时间-新闻创建时间 差的统计值")
    print("生成用户点击-交叉特征：用户点击新闻的 曝光时间小时级 的统计值")
    print("生成用户点击-交叉特征：用户点击新闻的 字数 的统计值")
    df_tra = df_tra[df_tra['is_click'] == 1]
    df_item = ItemLoader().load_pkl()
    df_tra = pd.merge(df_tra, df_item, on='item_id', how='left')
    print(df_tra.head())
    df_tra = df_tra.groupby(by=['user_id']).agg(
        u_click_diff_days_mean=pd.NamedAgg(column='ctx_cross_i_time_release_diff_days', aggfunc="mean"),
        # u_click_diff_days_std=pd.NamedAgg(column='ctx_cross_i_time_release_diff_days', aggfunc="std"),
        u_click_ctx_time_expose_hour_mean=pd.NamedAgg(column='ctx_time_expose_hour', aggfunc="mean"),
        # u_click_ctx_time_expose_hour_std=pd.NamedAgg(column='ctx_time_expose_hour', aggfunc="std"),
        u_click_ctx_timestamp_expose_mean=pd.NamedAgg(column='ctx_cross_i_time_release_diff_timestamp', aggfunc="mean"),
        # u_click_ctx_timestamp_expose_min=pd.NamedAgg(column='ctx_cross_i_time_release_diff_timestamp', aggfunc="max"),
        # u_click_ctx_timestamp_expose_max=pd.NamedAgg(column='ctx_cross_i_time_release_diff_timestamp', aggfunc="min"),
        # u_click_ctx_timestamp_expose_std=pd.NamedAgg(column='ctx_cross_i_time_release_diff_timestamp', aggfunc="std"),
        u_click_i_keys_len_mean=pd.NamedAgg(column='i_keys_len', aggfunc="mean"),
        # u_click_i_keys_len_max=pd.NamedAgg(column='i_keys_len', aggfunc="max"),
        # u_click_i_keys_len_min=pd.NamedAgg(column='i_keys_len', aggfunc="min"),
        # u_click_i_keys_len_std=pd.NamedAgg(column='i_keys_len', aggfunc="std"),
    )
    df_tra = df_tra.reset_index()

    df_tra = pd.merge(df_agg, df_tra, on='user_id', how='outer')
    print(df_tra.head())
    df_tra = MemoryOptimizer.reduce_mem_usage(df_tra)
    PickleUtil.save2pkl(df_tra, UserLoader().get_agg_pkl_name())
    print(df_tra.head())

    del df_tra, df_agg
    gc.collect()


def gen_user_features():
    # 1025
    # gen_user_static_feature()
    gen_user_agg_feature()


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    from pandarallel import pandarallel
    pandarallel.initialize()

    gen_user_features()
