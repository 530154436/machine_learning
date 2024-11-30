#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
# import sys
# sys.path.append("/root/zhengchubin/programing")
import pandas as pd

from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.dataset.loader_user import UserLoader
from bjtu_programming.search_rs.src.feature_engineering.context.time import DateConverter
from bjtu_programming.search_rs.src.feature_engineering.context.time_feature import TimeFeature
from bjtu_programming.search_rs.src.param import SampleParam
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil
from pandarallel import pandarallel

# https://zhuanlan.zhihu.com/p/61746020
pandarallel.initialize()


def gen_ctx_feature(df: pd.DataFrame, _prefix_file):
    print("生成时间特征...")
    # 1. 曝光时间对应的时间段、是否工作时间、星期几、是否周末
    TimeFeature.pipline_parallel(df)

    # 2. 曝光时间较新闻发布时间的距离
    df_item = ItemLoader().load_csv()
    df_item = df_item[['item_id', 'i_time_release']]
    df_item['i_time_release'] = df_item['i_time_release'].parallel_apply(DateConverter.timestamp_to_datetime)
    df = pd.merge(df, df_item, how='left', on='item_id')
    df["ctx_cross_i_time_release_days"] = pd.to_datetime(df["ctx_time_expose"]) - pd.to_datetime(df["i_time_release"])
    df["ctx_cross_i_time_release_days"] = df["ctx_cross_i_time_release_days"].parallel_map(lambda x: x.days)

    # cnt_read_duration暴露标签信息
    df = df.drop(columns=[TimeFeature.column, 'i_time_release'])
    df = MemoryOptimizer.reduce_mem_usage(df)
    PickleUtil.save2pkl(df, _prefix_file + '.pkl')
    print(df.head())
    del df_item
    gc.collect()


def gen_user_static_feature():

    def _max_prob(x: str):
        if pd.isna(x):
            return None
        max_label, max_prob = None, 0
        for item in str(x).split(','):
            if float(item.split(':')[1]) > max_prob:
                max_label = item.split(':')[0]
        return max_label

    df_user = UserLoader().load_csv()
    df_user['u_age_prob'] = df_user['u_age_prob'].parallel_apply(_max_prob)
    df_user['u_sex_prob'] = df_user['u_sex_prob'].parallel_apply(_max_prob)

    df_user = MemoryOptimizer.reduce_mem_usage(df_user)
    print(df_user.head())
    PickleUtil.save2pkl(df_user, 'user_info.pkl')
    del df_user
    gc.collect()


def gen_user_agg_feature(df: pd.DataFrame, _prefix_file):
    print("生成用户聚合特征...")
    df_agg = df.groupby(by=['user_id']).agg(
        u_total_cnt=pd.NamedAgg(column="is_click", aggfunc="count"),
        u_click_cnt=pd.NamedAgg(column="is_click", aggfunc="sum"),
        # cnt_read_duration 暴露标签，特征穿越了
    )
    df_agg['u_ctr'] = df_agg['u_click_cnt'] / df_agg['u_total_cnt']
    df_agg = df_agg.reset_index()
    print(df_agg.head())
    df_agg = MemoryOptimizer.reduce_mem_usage(df_agg)
    PickleUtil.save2pkl(df_agg, _prefix_file + '_' + 'u_agg' + '.pkl')

    del df_agg
    gc.collect()


def gen_user_cross_ctx_feature(df: pd.DataFrame, _prefix_file):
    print("生成用户-上下文交叉特征...")
    other_cols = ['ctx_net_env', 'ctx_time_expose_hour_bucket', 'ctx_time_expose_hour',
                  'ctx_time_expose_is_opening_hours', 'ctx_time_expose_day_of_week', 'ctx_time_expose_is_weekend']
    for column in other_cols:
        new_column_name = f'u_cross_{column}_ccnt'
        df_col = df.groupby(by=['user_id', column]).agg(
            click_cnt=pd.NamedAgg(column='is_click', aggfunc="sum")
        )
        df_col.rename(columns={'click_cnt': new_column_name}, inplace=True)
        df_col = df_col.reset_index()

        df_col = MemoryOptimizer.reduce_mem_usage(df_col)
        PickleUtil.save2pkl(df_col, _prefix_file + '_' + new_column_name + '.pkl')
        print(df_col.head())

        del df_col
        gc.collect()


def gen_user_cross_item_feature(df: pd.DataFrame, _prefix_file):
    print("生成用户-物品交叉特征...")
    df = df[['user_id', 'item_id', 'is_click']]
    df_item = ItemLoader().load_csv()
    df_item = df_item[["item_id", "i_category1", "i_category2", "i_keys_prob"]]
    df = pd.merge(df, df_item, on='item_id', how='left')

    # 1. 用户和新闻类别属性特征交集的点击次数：一级分类、二级分类
    for column in ["i_category1", "i_category2"]:
        new_column_name = f'u_cross_{column}_ccnt'
        df_col = df.groupby(by=['user_id', column]).agg(
            click_cnt=pd.NamedAgg(column='is_click', aggfunc="sum")
        )
        df_col = df_col.rename(columns={'click_cnt': new_column_name})
        df_col = df_col.reset_index()

        df_col = MemoryOptimizer.reduce_mem_usage(df_col)
        PickleUtil.save2pkl(df_col, _prefix_file + '_' + new_column_name + '.pkl')
        print(df_col.head())

        del df_col
        gc.collect()
    # TODO: 2. 用户和新闻的TopN关键词: Top3、Top5


def gen_item_static_feature(df: pd.DataFrame, _prefix_file):
    pass


def gen_item_agg_feature(df: pd.DataFrame, _prefix_file):
    print("生成Item聚合特征...")
    df_item = df.groupby(by=['item_id']).agg(
        i_total_cnt=pd.NamedAgg(column="is_click", aggfunc="count"),
        i_click_cnt=pd.NamedAgg(column="is_click", aggfunc="sum")
    )
    df_item['i_ctr'] = df_item['i_click_cnt'] / df_item['i_total_cnt']
    df_item = df_item.reset_index()
    print(df_item.head())
    df_item = MemoryOptimizer.reduce_mem_usage(df_item)
    PickleUtil.save2pkl(df_item, _prefix_file + '_' + 'i_agg' + '.pkl')

    del df_item
    gc.collect()


def gen_features(param: SampleParam, valid: bool):
    """
    生成训练集数据
    :param param:
    :param valid:
    :return:
    """
    # 加载数据集
    loader_train = TrainLoader()
    _prefix_file = loader_train.get_file_name_by_param(param, valid)
    df: pd.DataFrame = loader_train.load_csv_by_param(param, valid)
    df = df.drop(columns=['u_cnt_flush', 'ctx_show_pos', 'cnt_read_duration'])

    gen_ctx_feature(df, _prefix_file)

    if valid:
        print("生成特征完成.")
        return

    gen_user_static_feature()
    gen_user_agg_feature(df, _prefix_file)
    gen_user_cross_ctx_feature(df, _prefix_file)
    gen_user_cross_item_feature(df, _prefix_file)

    gen_item_agg_feature(df, _prefix_file)
    print("生成特征完成.")


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    _if_correct_samples = [0, 1]
    _if_skip_aboves = [0, 1]
    _valids = [False, True]
    # _if_correct_samples = [0]
    # _if_skip_aboves = [0]

    for _valid in _valids:
        for _if_correct_sample in _if_correct_samples:
            for _if_skip_above in _if_skip_aboves:
                _parm = SampleParam(if_skip_above=_if_skip_above, if_correct_sample=_if_correct_sample)
                gen_features(_parm, _valid)
