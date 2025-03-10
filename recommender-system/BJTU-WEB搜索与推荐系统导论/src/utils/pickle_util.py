#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pickle
from typing import Any

from bjtu_programming.search_rs import DATA_DIR


class PickleUtil(object):

    base_dir = DATA_DIR

    @classmethod
    def save2pkl(cls, obj: Any, file_name: str):
        file = cls.base_dir.joinpath(file_name)
        with open(file, 'wb') as file:
            pickle.dump(obj, file)
        print(f'Save to {file} finished.')

    @classmethod
    def load_from_pkl(cls, file_name: str) -> Any:
        file = cls.base_dir.joinpath(file_name)
        with open(file, 'rb') as file:
            obj = pickle.load(file)
        # print(f'Load from {file} finished.')
        return obj
