#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
import sys
# sys.path.append("/root/zhengchubin/programing")
import pandas as pd
from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_test import TestLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.feature_engineering.context.time import DateConverter, DateUtil
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


def gen_ctx_feature(df: pd.DataFrame, column: str = 'ctx_time_expose', is_test: bool = False):
    print("生成时间特征...")
    # 一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
    df[f'{column}_hour_bucket'] = df[column].parallel_apply(DateUtil.get_hour_bucket)

    # 是否工作时间
    df[f'{column}_hour'] = df[column].parallel_apply(DateUtil.get_hour)
    df[f'{column}_is_opening_hours'] = 0
    mask = ((df[f'{column}_hour'] >= 8) & (df[f'{column}_hour'] < 22))
    df.loc[mask, f'{column}_is_opening_hours'] = 1

    # 星期几
    df[f'{column}_day_of_week'] = df[column].parallel_apply(DateUtil.get_day_of_week)

    # 是否周末
    df[f'{column}_is_weekend'] = df[f'{column}_day_of_week'].parallel_apply(lambda x: 1 if x in [5, 6] else 0)
    df = MemoryOptimizer.reduce_mem_usage(df)

    print("生成曝光时间较新闻发布时间的距离...")
    df_item = ItemLoader().load_csv()
    df_item = df_item[['item_id', 'i_timestamp_release']]
    df_item['i_time_release'] = df_item['i_timestamp_release'].parallel_apply(
        lambda x: DateConverter.timestamp_to_datetime(x, unit='ms'))

    df = pd.merge(df, df_item, how='left', on='item_id')
    df["ctx_cross_i_time_release_diff_days"] = (df[column] - df["i_time_release"]).parallel_map(lambda x: max(x.days, 0))
    df["ctx_cross_i_time_release_diff_timestamp"] = df["ctx_timestamp_expose"] - df["i_timestamp_release"]

    # 过滤：曝光时间 > 物品创建时间
    if not is_test:
        mask = df["ctx_cross_i_time_release_diff_timestamp"] > 0
        df = df.loc[mask]
    df = MemoryOptimizer.reduce_mem_usage(df)

    del df_item
    gc.collect()
    return df


def run_test():
    # 加载数据集
    df: pd.DataFrame = TestLoader().load_csv()

    # 上下文特征
    df['ctx_time_expose'] = df['ctx_timestamp_expose'].parallel_apply(
        lambda x: DateConverter.timestamp_to_datetime(x, unit='ms'))
    df = gen_ctx_feature(df, is_test=True)

    df.drop(columns=['i_time_release', 'i_timestamp_release', 'ctx_time_expose'], inplace=True)
    print(df.head())

    _file = TestLoader().get_csv_file_name()
    PickleUtil.save2pkl(df, DATA_DIR.joinpath(f"{_file}.pkl"))
    print('测试集:', df.shape)

    del df
    gc.collect()


def run_train():
    """
    预处理
    """
    # 加载数据集
    df: pd.DataFrame = TrainLoader().load_csv()

    # 样本去重去噪：客户端多次上报的同一卡片曝光、消费
    print("样本去重去噪前:", df.shape)
    df['ctx_time_expose'] = df['ctx_timestamp_expose'].parallel_apply(
        lambda x: DateConverter.timestamp_to_datetime(x, unit='ms'))

    df = df.sort_values(by=['ctx_time_expose', "u_cnt_flush", "ctx_show_pos", 'is_click'],
                        ascending=[True, True, True, True])
    df: pd.DataFrame = df.drop_duplicates(subset=["user_id", "item_id", "ctx_time_expose",
                                                  "u_cnt_flush", "ctx_show_pos"],
                                          keep='last')
    df = df.reset_index(drop=True)

    # 上下文特征
    df = gen_ctx_feature(df)
    print("样本去重去噪后:", df.shape)
    print(df.head())

    # 按时间升序拆分训练集、验证集
    df = df.drop(columns=['i_time_release', 'i_timestamp_release', 'ctx_time_expose',
                          'ctx_show_pos', 'cnt_read_duration'])
    # cnt_read_duration暴露标签信息
    # print(df[df['user_id'] == 1000014754])

    train_size = int(df.shape[0] * 0.8)
    df_tra = df[:train_size]
    df_val = df[train_size:]
    print('训练集:', df_tra.shape)
    print('验证集:', df_val.shape)

    _file = TrainLoader().get_csv_file_name()
    PickleUtil.save2pkl(df_tra, DATA_DIR.joinpath(f"{_file}_tra.pkl"))
    PickleUtil.save2pkl(df_val, DATA_DIR.joinpath(f"{_file}_val.pkl"))
    print("数据集切分完成.\n")

    del df, df_tra, df_val
    gc.collect()


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    from pandarallel import pandarallel
    pandarallel.initialize()
    run_train()
    run_test()
