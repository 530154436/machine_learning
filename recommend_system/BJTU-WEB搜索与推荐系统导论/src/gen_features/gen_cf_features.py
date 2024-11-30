#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
import math

import pandas as pd

from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.dataset.loader_user import UserLoader
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


def calc_item_cf(x, u_click_seq, i_click_seq, top_n=None):
    user_id = x['user_id']
    item_id = x['item_id']

    u_clicks = u_click_seq.get(user_id, [])
    i_vector = set(i_click_seq.get(item_id, []))
    # print('u_clicks', user_id, u_clicks)
    # print('i_vector', item_id, i_vector)

    total_item_cf_sim = 0
    if not top_n:
        top_n = len(u_clicks)
    # total = sum(range(top_n)) + top_n
    for pos, click_i_id in enumerate(u_clicks):

        if top_n and pos >= top_n:
            break

        click_i_vector = set(i_click_seq.get(click_i_id, []))
        if len(i_vector) != 0 and click_i_vector != 0:
            item_cf_sim = len(i_vector & click_i_vector) / math.sqrt(len(i_vector) * len(click_i_vector))
        else:
            item_cf_sim = 0
        # print('click_i_id', click_i_id, click_i_vector)
        # print(item_cf_sim)

        # 位置降权
        weight = (top_n - pos) / top_n
        total_item_cf_sim += weight * item_cf_sim
    # print()
    return total_item_cf_sim


def gen_item_cf_features():
    df_tra: pd.DataFrame = TrainLoader().load_pkl_by_valid(valid=False)
    df_tra = df_tra[['user_id', 'item_id']]

    print("生成ItemCF特征...")
    u_click_seq = UserLoader().load_dict_pkl('click_seq')
    i_click_seq = ItemLoader().load_dict_pkl('click_seq')
    df_tra['item_cf_sim_top1'] = df_tra.apply(lambda x: calc_item_cf(x, u_click_seq, i_click_seq, top_n=5), axis=1)
    # df_tra['item_cf_sim_top3'] = df_tra.parallel_apply(lambda x: calc_item_cf(x, u_click_seq, i_click_seq, top_n=3),
    #                                                    axis=1)
    # df_tra['item_cf_sim_top5'] = df_tra.parallel_apply(lambda x: calc_item_cf(x, u_click_seq, i_click_seq, top_n=5),
    #                                                    axis=1)
    # df_tra['item_cf_sim'] = df_tra.parallel_apply(lambda x: calc_item_cf(x, u_click_seq, i_click_seq), axis=1)

    print("生成ItemCF特征: ", df_tra.shape)
    print(df_tra.head())
    df_agg = MemoryOptimizer.reduce_mem_usage(df_tra)
    PickleUtil.save2pkl(df_agg, UserLoader().get_cross_pkl_name(['user_id', 'item_id'], name='item_cf'))

    del df_tra, df_agg
    gc.collect()


def gen_cf_features():
    # 1025
    gen_item_cf_features()


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    from pandarallel import pandarallel
    pandarallel.initialize()

    gen_cf_features()
