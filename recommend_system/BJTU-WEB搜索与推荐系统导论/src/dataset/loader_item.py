#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from typing import Dict, List

import pandas as pd

from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_base import BaseLoader
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


class ItemLoader(BaseLoader):

    def __init__(self):
        self.__csv_file = "news_info.txt"
        self.__columns = [
            "item_id", "i_title", "i_timestamp_release",
            "i_cnt_img", "i_category1", "i_category2", "i_keys_prob"
        ]
        super().__init__(self.__csv_file, self.__columns)

    def get_dict_pkl_name(self, name='keys') -> str:
        return f"{self.get_csv_file_name()}_{name}.pkl"

    def load_dict_pkl(self, name='keys') -> Dict[str, List[str]]:
        _file = DATA_DIR.joinpath(self.get_dict_pkl_name(name))
        df = PickleUtil.load_from_pkl(_file)
        print("加载文章关键词", len(df))
        return df

    def get_seq_pkl_name(self, name='click_seq') -> str:
        return f"{self.get_csv_file_name()}_{name}.pkl"

    def load_seq_pkl(self, name='click_seq') -> List[str]:
        _file = DATA_DIR.joinpath(self.get_dict_pkl_name(name))
        df = PickleUtil.load_from_pkl(_file)
        print(len(df), _file)
        return df


if __name__ == '__main__':
    pd.set_option('display.width', 800)  # 设置打印宽度
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    ItemLoader().load_csv(head=10)
    d = ItemLoader().load_dict_pkl('title')
    for i in d:
        print(i)
        break

