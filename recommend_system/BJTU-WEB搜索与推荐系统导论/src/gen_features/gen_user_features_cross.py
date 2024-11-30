#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc

import pandas as pd
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.dataset.loader_user import UserLoader
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


def gen_user_profile_cross_feature():
    df_tra: pd.DataFrame = TrainLoader().load_pkl_by_valid(valid=False)
    print("生成用户-物品交叉特征...")
    df_tra = df_tra[df_tra['is_click'] == 1]

    df_item = ItemLoader().load_pkl()
    df_user = UserLoader().load_pkl()
    df_tra = pd.merge(df_tra, df_item, on='item_id', how='left')
    df_tra = pd.merge(df_tra, df_user, on='user_id', how='left')

    # 新闻的类别/上下文环境特征 分别与 用户所有属性ID、性别、年龄 交叉
    # TODO: 这里有问题，不应该计数
    other_columns = [
        "i_category1", "i_category2",
        'ctx_net_env', 'ctx_time_expose_hour_bucket', 'ctx_time_expose_hour',
        'ctx_time_expose_is_opening_hours', 'ctx_time_expose_day_of_week', 'ctx_time_expose_is_weekend'
    ]
    user_profiles = ['user_id', 'u_age_max', 'u_sex_max', 'u_device', 'u_os', 'u_city', 'u_province']
    for column in other_columns:
        for u_col in user_profiles:
            df_col = df_tra.groupby(by=[u_col, column]).agg(
                click_cnt=pd.NamedAgg(column='is_click', aggfunc="sum"),
                # total_cnt=pd.NamedAgg(column='is_click', aggfunc="count")
            )
            # df_col['ctr'] = df_col[f'click_cnt'] / df_col['total_cnt']
            df_col.rename(columns={'click_cnt': f'{u_col}_cross_{column}_click_cnt',
                                   'total_cnt': f'{u_col}_cross_{column}_total_cnt',
                                   'ctr': f'{u_col}_cross_{column}_ctr'}, inplace=True)
            df_col = df_col.reset_index()

            df_col = MemoryOptimizer.reduce_mem_usage(df_col)
            PickleUtil.save2pkl(df_col, UserLoader().get_cross_pkl_name([u_col, column]))
            print(df_col.head())

            del df_col
            gc.collect()


def gen_user_click_seqs():
    # 生成用户点击序列
    df_tra: pd.DataFrame = TrainLoader().load_pkl_by_valid(valid=False)
    df_tra = df_tra[df_tra['is_click'] == 1]
    df_tra = df_tra[['user_id', 'item_id', 'ctx_timestamp_expose']]

    user_click_seq = dict()
    print('生成用户点击序列..')
    for i, (user_id, chunk) in enumerate(df_tra.groupby(by=['user_id'])):

        chunk = chunk.sort_values(by='ctx_timestamp_expose', ascending=False)
        ids = chunk['item_id'].tolist()
        user_click_seq[user_id] = ids

    PickleUtil.save2pkl(user_click_seq, UserLoader().get_seq_pkl_name('click_seq'))


def gen_user_features_cross():
    # 1025
    # gen_user_click_seqs()

    # 都没啥用？？？
    gen_user_profile_cross_feature()


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    from pandarallel import pandarallel
    pandarallel.initialize()

    gen_user_features_cross()
