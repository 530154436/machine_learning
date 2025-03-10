#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd
from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_base import BaseLoader
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil
from bjtu_programming.search_rs.src.v1.param import SampleParam
from bjtu_programming.search_rs.src.utils.pd_util import PandasUtil


class TrainLoader(BaseLoader):

    def __init__(self):
        # self.__csv_file = "train_info.txt"
        self.__csv_file = "train_info_2000000.txt"

        self.__columns = [
            "user_id", "item_id", "ctx_timestamp_expose", "ctx_net_env",
            "u_cnt_flush", "ctx_show_pos", "is_click", "cnt_read_duration"
        ]
        super().__init__(self.__csv_file, self.__columns)

    def get_file_name_by_param(self, param: SampleParam, valid) -> str:
        _file = self.get_csv_file_name()
        if valid:
            _file = f"{_file}_val_{param.if_correct_sample}_{param.if_skip_above}"
        else:
            _file = f"{_file}_tra_{param.if_correct_sample}_{param.if_skip_above}"
        return _file

    def load_csv_by_param(self, param: SampleParam, valid, head: int = 0):
        _file = self.get_file_name_by_param(param, valid)
        _file = DATA_DIR.joinpath(_file + '.csv')
        df = PandasUtil.read_csv(_file, sep=",", chunk_size=50000, header=None)
        df.columns = self.columns
        print(f"{df.shape}: {_file.name}")
        if head:
            print(df.head(head))
        return df

    def load_pkl_by_valid(self, valid: bool = False) -> pd.DataFrame:
        _file = self.get_csv_file_name()
        if valid:
            _file = DATA_DIR.joinpath(f"{_file}_val.pkl")
        else:
            _file = DATA_DIR.joinpath(f"{_file}_tra.pkl")

        df = PickleUtil.load_from_pkl(_file)
        print(f"{df.shape}: {_file.name}")
        return df


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    TrainLoader().load_csv()
    # TrainLoader().load_csv_by_param(SampleParam(), valid=False)
    # TrainLoader().load_csv_by_param(SampleParam(), valid=True)
    TrainLoader().load_pkl_by_valid()
    _df = TrainLoader().load_pkl_by_valid(valid=True)
    print(_df.head())
