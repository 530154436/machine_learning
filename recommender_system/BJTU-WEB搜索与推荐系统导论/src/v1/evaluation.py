#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import numpy as np
import pandas as pd

from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_item import ItemLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.feature_engineering.context.time import DateConverter
from bjtu_programming.search_rs.src.feature_engineering.context.time_feature import TimeFeature
from bjtu_programming.search_rs.src.models.xgb_lightgbm import XgbLightGBWrapper
from bjtu_programming.search_rs.src.param import SampleParam
from bjtu_programming.search_rs.src.train import KEY_FILE, SELECT_COLUMNS
from bjtu_programming.search_rs.src.utils.pickle_util import PickleUtil

test_file = DATA_DIR.joinpath("test_info.txt")
df_test = pd.read_csv(test_file,
                      sep='\t',
                      header=None)
df_test.columns = ["order", "user_id", "item_id", 'ctx_time_expose', "ctx_net_env", "u_cnt_flush"]

print("生成时间特征...")
# 1. 曝光时间对应的时间段、是否工作时间、星期几、是否周末
TimeFeature.pipline(df_test)

# 2. 曝光时间较新闻发布时间的距离
df_item = ItemLoader().load_csv()
df_item = df_item[['item_id', 'i_time_release', "i_category1", "i_category2"]]
df_item['i_time_release'] = df_item['i_time_release'].apply(DateConverter.timestamp_to_datetime)
df_test = pd.merge(df_test, df_item, how='left', on='item_id')

_df_user = PickleUtil.load_from_pkl('user_info.pkl')
df_test = pd.merge(df_test, _df_user, on='user_id', how='left')

df_test["ctx_cross_i_time_release_days"] = pd.to_datetime(df_test["ctx_time_expose"]) - pd.to_datetime(df_test["i_time_release"])
df_test["ctx_cross_i_time_release_days"] = df_test["ctx_cross_i_time_release_days"].map(lambda x: x.days)
df_test = df_test.drop(columns=[TimeFeature.column, 'i_time_release', 'u_cnt_flush'])

# 拼接特征
_if_correct_samples = [0, 1]
_if_skip_aboves = [0, 1]
for _if_correct_sample in _if_correct_samples:
    for _if_skip_above in _if_skip_aboves:
        _loader_train = TrainLoader()
        _parm = SampleParam(if_skip_above=_if_skip_above, if_correct_sample=_if_correct_sample)
        tra_file_name = _loader_train.get_file_name_by_param(_parm, valid=False)
        for keys, subfix in KEY_FILE.items():
            pkl_file = f'{tra_file_name}_{subfix}.pkl'
            print(pkl_file)
            _df_sub: pd.DataFrame = PickleUtil.load_from_pkl(pkl_file)
            print('拼接特征:', _df_sub.shape, pkl_file)

            df_test = pd.merge(df_test, _df_sub, how='left', on=list(keys))
        print('拼接特征完成', df_test.shape)
        # print(df_test.describe())
        print(df_test.info())

        CAT_LABEL_ENCODERS = PickleUtil.load_from_pkl(f'{tra_file_name}_CAT_LABEL_ENCODERS.pkl')
        for cat_feat, encoder in CAT_LABEL_ENCODERS.items():
            if not encoder:
                continue

            # error处理
            df_test[cat_feat] = df_test[cat_feat].map(lambda s: -1 if s not in encoder.classes_ else s)
            encoder.classes_ = np.append(encoder.classes_, -1)

            print('分类变量标签编码转换:', cat_feat)
            df_test[cat_feat] = encoder.transform(df_test[cat_feat])

        # 筛选特征
        cat_feats = list(set(CAT_LABEL_ENCODERS.keys()) & set(SELECT_COLUMNS))
        df_test = df_test[SELECT_COLUMNS].fillna(0)
        print(df_test.head())

        # 预测
        model_file = DATA_DIR.joinpath(f"{tra_file_name}.model")
        predicts = XgbLightGBWrapper.predict(model_file, df_test)
        print(predicts)

        df = pd.DataFrame(
            {"pred": predicts}
        )
        df.to_csv(DATA_DIR.joinpath(f"{tra_file_name}_predicts.csv"), header=False)

        break
    break
