#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import sys
# sys.path.append("/root/zhengchubin/programing")
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait
from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.feature_engineering.sampling import Sampler
from bjtu_programming.search_rs.src.param import SampleParam


def split_to_train_valid(param: SampleParam):
    """
    划分数据集
    :param param:
    :return:
    """
    # 加载数据集
    loader_train = TrainLoader()
    df: pd.DataFrame = loader_train.load_csv()

    # 样本去重去噪：客户端多次上报的同一卡片曝光、消费
    print(param)
    print("样本去重去噪前:", df.shape)
    df: pd.DataFrame = df.drop_duplicates(subset=["user_id", "item_id", "ctx_time_expose",
                                                  "u_cnt_flush", "ctx_show_pos"])
    df = df.reset_index(drop=True)
    print("样本去重去噪后:", df.shape)

    # 正样本校验：1/点击并且阅读时长大于阈值 0/其他
    if param.if_correct_sample and param.min_cnt_read_duration:
        column = list(df.columns).index('is_click')
        print(f"正样本校验：1/点击并且阅读时长大于{param.min_cnt_read_duration}s, 0/其他")
        print(df['is_click'].value_counts())
        mask = (df['is_click'] == 1) & (df['cnt_read_duration'] < param.min_cnt_read_duration)
        indices = df.loc[mask].index
        df.iloc[indices, column] = 0
        print(df["is_click"].value_counts())

    # skip_above 采样
    if param.if_skip_above:
        executor = ProcessPoolExecutor(max_workers=4)
        futures = []

        print("skip_above 采样中...:")
        selected_indices = []

        for user_id, chunk in df.groupby(by='user_id'):
            # print(user_id)
            # 物品ID、曝光时间、刷新次数、展示位置、是否点击
            chunk = chunk[['item_id', 'ctx_time_expose', 'u_cnt_flush', 'ctx_show_pos', 'is_click']]

            # 单进程
            # candidates = Sampler.skip_above(chunk, if_neg_sampling=True)
            # selected_indices.extend(candidates)

            # 多进程
            future = executor.submit(Sampler.skip_above, chunk, if_neg_sampling=True)
            futures.append(future)

        wait(futures)
        for future in futures:
            selected_indices.extend(future.result())

        df = df.iloc[selected_indices]
        df: pd.DataFrame = df.drop_duplicates(subset=["user_id", "item_id", "ctx_time_expose",
                                                      "u_cnt_flush", "ctx_show_pos"])
        df = df.reset_index(drop=True)
        del selected_indices
        # del futures
        print("skip_above 采样后:", df.shape)
        print(df['is_click'].value_counts())

    # 按时间升序拆分训练集、验证集
    df = df.sort_values(by=['ctx_time_expose'], ascending=[True])
    train_size = int(df.shape[0] * 0.7)
    df_tra = df[:train_size]
    df_val = df[train_size:]

    _file = loader_train.get_csv_file_name()
    df_tra.to_csv(DATA_DIR.joinpath(f"{loader_train.get_file_name_by_param(param, valid=False)}.csv"), index=False)
    df_val.to_csv(DATA_DIR.joinpath(f"{loader_train.get_file_name_by_param(param, valid=True)}.csv"), index=False)
    print("数据集切分完成.\n")


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    for _if_correct_sample in [0, 1]:
        for _if_skip_above in [0, 1]:
            _parm = SampleParam(if_correct_sample=_if_correct_sample,
                                if_skip_above=_if_skip_above, min_cnt_read_duration=10)
            split_to_train_valid(_parm)
