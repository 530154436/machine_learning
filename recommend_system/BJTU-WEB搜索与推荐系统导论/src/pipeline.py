#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from bjtu_programming.search_rs.src.evaluation import evaluate
from bjtu_programming.search_rs.src.gen_features.gen_item_features import gen_item_features
from bjtu_programming.search_rs.src.gen_features.gen_user_features import gen_user_features
from bjtu_programming.search_rs.src.gen_features.gen_user_features_cross import gen_user_features_cross
from bjtu_programming.search_rs.src.split_to_train_valid import run_train, run_test
from bjtu_programming.search_rs.src.train import train

# https://zhuanlan.zhihu.com/p/61746020
from pandarallel import pandarallel
pandarallel.initialize()


# run_train()
# run_test()
# gen_item_features()
# gen_user_features()
gen_user_features_cross()
# train()
# evaluate()
