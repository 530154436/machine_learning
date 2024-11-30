#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os

import pandas as pd

from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.utils.pd_opt_util import MemoryOptimizer
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil


for file in DATA_DIR.iterdir():
    if file.suffix == '.pkl':
        obj: pd.DataFrame = PickleUtil.load_from_pkl(file)
        obj = MemoryOptimizer.reduce_mem_usage(obj)

        os.remove(file)
        PickleUtil.save2pkl(obj, file)
        print(obj.head())
