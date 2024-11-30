#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 用户历史行为相关特征
import functools
import gc
import math
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.dataset.loader_user import UserLoader
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


def gen_user_his_feature_timestamp(row, u_click_seq: dict, item_property: dict, top_n: int = 5):
    """
    计算候选item与与用户最后N次点击item的时间差特征、字数差特征、相似度的和(最大， 最小，均值)
    :return:
    """
    user_id = row.get('user_id')
    item_id = row.get('item_id')
    cur_profile = item_property.get(item_id)

    # 遍历用户的最后N次点击文章
    _sum, _max, _min, _mean = 0, 0, 10000000, 0
    for hit_item_id in u_click_seq.get(user_id, []):
        hit_profile = item_property.get(hit_item_id)
        _diff = cur_profile - hit_profile
        # print(f"候选item_id: {item_id}, {cur_profile}")
        # print(f"历史item_id: {hit_item_id}, {hit_profile}, {_diff}")

        _sum += _diff
        if _diff > _max:
            _max = _diff
        if _diff < _min:
            _min = _diff
    _mean = _sum / top_n
    return _max, _min, _sum, _mean


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    from pandarallel import pandarallel
    pandarallel.initialize()

    print('生成用户历史行为时间差特征...')
    user_seq_dict = UserLoader().load_dict_pkl('click_seq')
    df_item = ItemLoader().load_pkl()
    # column = 'i_timestamp_release'
    column = 'i_keys_len'
    df_item = df_item.set_index('item_id')[column].to_dict()
    methods = ['max', 'min', 'sum', 'mean']
    columns = [f'u_hit_{column}_{m}' for m in methods]
    keys = ['user_id', 'item_id']
    for _valid in [False, True]:
        df_tra: pd.DataFrame = TrainLoader().load_pkl_by_valid(valid=_valid)
        df_tra[columns] = df_tra.parallel_apply(lambda x: gen_user_his_feature_timestamp(x, user_seq_dict, df_item),
                                                axis=1, result_type="expand")
        print(df_tra[keys+columns].head())

        PickleUtil.save2pkl(df_tra[keys+columns],
                            UserLoader().get_cross_pkl_name(keys=['user_id', 'item_id'],
                                                            name=f'user_his_click_feat_{_valid}'))
