#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from typing import Union


class PandasUtil(object):

    @classmethod
    def read_csv(cls, file: Union[str, Path], header=None,
                 chunk_size: int = 1000, sep: str = ",") -> pd.DataFrame:
        """
        分块读取 CSV 文件
        :param header:
        :param sep:         分隔符
        :param file:        待读取的文件
        :param chunk_size:  分块大小
        :return:
        """
        _list = []
        iterator = pd.read_csv(file, iterator=True, chunksize=chunk_size, sep=sep, header=header)
        for row, chunk in enumerate(iterator, start=1):
            _list.append(chunk)
        return pd.concat(_list)

    @classmethod
    def read_csv_by_dd(cls, file: Union[str, Path], separator: str = ",") -> pd.DataFrame:
        """
        分块读取 CSV 文件
        """
        df = dd.read_csv(file, sep=separator)
        # df = dd.read_csv(file, sep=separator).compute()
        return df

    @classmethod
    def split_col(cls, data: pd.DataFrame, column, _type=None):
        """拆分成列

        :param _type:
        :param data: 原始数据
        :param column: 拆分的列名
        :type data: pandas.core.frame.DataFrame
        :type column: str
        """
        max_len = max(list(map(len, data[column].values)))  # 最大长度
        new_col = data[column].apply(lambda x: x + [None] * (max_len - len(x)))  # 补空值，None可换成np.nan
        new_col = np.array(new_col.tolist()).T  # 转置
        for i, j in enumerate(new_col):
            data[column + str(i)] = j
            if _type:
                 data[column + str(i)] = data[column + str(i)].astype(_type)
        return data


if __name__ == '__main__':
    from pyinstrument import Profiler
    from bjtu_programming.search_rs import DATA_DIR

    profiler = Profiler()
    profiler.start()
    # _file = DATA_DIR.joinpath("train_info.txt")
    _file = DATA_DIR.joinpath("train_info_5000000.txt")
    # _df = read_csv(DATA_DIR.joinpath("train_info.txt"), separator="\t", chunk_size=10000)
    # _df = PandasUtil.read_csv(_file, separator="\t", chunk_size=50000)
    _df = PandasUtil.read_csv_by_dd(_file, separator="\t")
    print(_df.shape)

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
