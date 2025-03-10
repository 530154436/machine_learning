#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd
from typing import Optional, Union, Dict

from sklearn.preprocessing import LabelEncoder

from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.utils.pd_util import PandasUtil
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


class BaseLoader(object):

    def __init__(self, path: str, columns: Optional[list]):
        self.__csv_file = path
        self.__columns = columns

    @property
    def csv_file(self) -> str:
        return self.__csv_file

    @property
    def columns(self) -> Optional[list]:
        return self.__columns

    def get_csv_file_name(self):
        return self.__csv_file.split(".")[0]

    def load_csv(self, head: int = 0, header=None) -> pd.DataFrame:
        _csv = DATA_DIR.joinpath(self.__csv_file)
        df = PandasUtil.read_csv(_csv, sep="\t", chunk_size=50000, header=header)
        df.columns = self.__columns
        print(f"{df.shape}: {_csv.name}")
        if head:
            print(df.head(head))
        return df

    def get_label_encoder_pkl_name(self) -> str:
        return f"{self.get_csv_file_name()}_label_encoder_.pkl"

    def load_label_encoder(self) -> Dict[str, LabelEncoder]:
        _file = DATA_DIR.joinpath(self.get_label_encoder_pkl_name())
        encoder: Dict[str, LabelEncoder] = PickleUtil.load_from_pkl(_file)
        print(f"Labels: {encoder.keys()}: {_file.name}")
        return encoder

    def get_pkl_name(self) -> str:
        return f"{self.get_csv_file_name()}.pkl"

    def load_pkl(self) -> pd.DataFrame:
        _file = DATA_DIR.joinpath(self.get_pkl_name())
        df = PickleUtil.load_from_pkl(_file)
        print(f"{df.shape}: {_file.name}")
        return df

    def get_agg_pkl_name(self):
        return self.get_csv_file_name() + '_agg.pkl'

    def load_agg_pkl(self):
        _file = DATA_DIR.joinpath(self.get_agg_pkl_name())
        df = PickleUtil.load_from_pkl(_file)
        print(f"{df.shape}: {_file.name}")
        return df

    def get_cross_pkl_name(self, keys: Union[list, tuple], name=''):
        return self.get_csv_file_name() + '_' + '_'.join(keys) + '_' + name + '.pkl'

    def load_cross_pkl(self, keys: Union[list, tuple], name='') -> pd.DataFrame:
        _file = DATA_DIR.joinpath(self.get_cross_pkl_name(keys, name))
        df = PickleUtil.load_from_pkl(_file)
        print(f"{df.shape}: {_file.name}")
        return df

