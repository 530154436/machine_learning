#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2021/9/15 14:01
# @function:    调整数据类型，从而减少内存使用
import pandas as pd
import numpy as np
from typing import Union


class MemoryOptimizer(object):

    @classmethod
    def mem_usage(cls, _obj: Union[pd.DataFrame, pd.Series],
                  n_digits: int = 6, verbose: bool = False) -> float:
        """
        统计 Pandas 对象内存占用情况
        Args:
            _obj:       Pandas 对象
            n_digits:   内存使用量保留的小数位数
            verbose:    是否打印
        Returns:
        """
        if isinstance(_obj, pd.DataFrame):
            usage_b = _obj.memory_usage(deep=True).sum()
        else:
            # pd.Series
            usage_b = _obj.memory_usage(deep=True)

        # 字节转MB
        usage_mb = usage_b / 1024 ** 2

        if verbose:
            print(f"Sum: {usage_mb:03.6f} MB")

        return round(usage_mb, ndigits=n_digits)

    @classmethod
    def reduce_mem_usage(cls, df: pd.DataFrame, verbose: bool = True,
                         categorical_columns: Union[list, set, tuple] = None,
                         date_columns: Union[list, set, tuple] = None) -> pd.DataFrame:
        """
        调整数据类型，从而减少内存使用。优化点:
            1. 整型/浮点根据数值范围降低字节位数，如 int64 => int8
            2. 整型若为正整数，则调整为无符号整型，如 int8 => uint8
            3. 字符串类型若为分类变量，如 object => category
            4. 日期类型 str => datetime

        Args:
            df:                     数据帧
            verbose:                是否打印调试信息
            categorical_columns:    分类变量列名
            date_columns:           日期类型列名
        Returns:
        """

        # 整型
        int_types = ['int', 'int8', 'int16', 'int32', 'int64']

        # 浮点
        float_types = ['float', 'float16', 'float32', 'float64']

        start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

        # 哪些列包含空值，空值用-999填充。why：因为np.nan当做float处理
        # NAlist = []

        for col in df.columns:
            d_type = df[col].dtype

            if d_type == 'object':
                try:
                    num_unique_values = len(df[col].unique())
                    num_total_values = len(df[col])

                    # category 类型用于不同值的数量少于值的总数量的 50% 的 object 列
                    if categorical_columns is not None \
                            and col in categorical_columns \
                            and num_unique_values / num_total_values < 0.5:
                        df[col] = df[col].astype('category')

                    # 字符串的日期类型
                    if date_columns and col in date_columns:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
                except TypeError as err:
                    print(f'column={col}, d_type={d_type}: {err}')

            # 整型
            if d_type in int_types:
                _min = df[col].min()
                _max = df[col].max()

                # 无符号整型
                if _min >= 0:
                    if _max <= np.iinfo(np.uint8).max:  # 255
                        df[col] = df[col].astype(np.uint8)
                    elif _max <= np.iinfo(np.uint16).max:  # 65535
                        df[col] = df[col].astype(np.uint16)
                    elif _max <= np.iinfo(np.uint32).max:  # 4294967295
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)

                # 有符号整型
                else:
                    if _min >= np.iinfo(np.int8).min and _max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif _min > np.iinfo(np.int16).min and _max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif _min > np.iinfo(np.int32).min and _max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif _min > np.iinfo(np.int64).min and _max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

            # 浮点类型
            if d_type in float_types:
                _min = df[col].min()
                _max = df[col].max()

                if _min >= np.finfo(np.float16).min and _max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif _min >= np.finfo(np.float32).min and _max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

            if verbose:
                print("{0:20}\t{1:20}==>\t{2:20}".format(col, str(d_type), str(df[col].dtype)))

        end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        if verbose:
            print('Mem usage from {:5.2f} Mb decreased to {:5.2f} Mb ({:.1f}% reduction)'
                  .format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
