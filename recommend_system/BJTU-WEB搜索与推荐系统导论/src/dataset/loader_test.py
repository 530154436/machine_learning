#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd
from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_base import BaseLoader
from bjtu_programming.search_rs.src.v1.param import SampleParam
from bjtu_programming.search_rs.src.utils.pd_util import PandasUtil


class TestLoader(BaseLoader):

    def __init__(self):
        self.__csv_file = "test_info.txt"

        self.__columns = [
            "order", "user_id", "item_id", 'ctx_timestamp_expose', "ctx_net_env", "u_cnt_flush"
        ]
        super().__init__(self.__csv_file, self.__columns)


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    TestLoader().load_csv()
