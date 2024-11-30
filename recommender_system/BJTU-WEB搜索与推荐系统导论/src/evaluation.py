#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd
from bjtu_programming.search_rs import DATA_DIR
from bjtu_programming.search_rs.src.dataset.loader_test import TestLoader
from bjtu_programming.search_rs.src.dataset.loader_train import TrainLoader
from bjtu_programming.search_rs.src.models.xgb_lightgbm import XgbLightGBWrapper
from bjtu_programming.search_rs.src.train import concat_features


def evaluate():
    # 预测
    df_test = TestLoader().load_pkl()
    orders = df_test['order']

    df_test = df_test.drop(columns=['order'])
    df_test = concat_features(df_test)
    df_test = df_test.astype('float32')

    tra_file_name = TrainLoader().get_csv_file_name()
    model_file = DATA_DIR.joinpath(f"{tra_file_name}.model")
    predicts = XgbLightGBWrapper.predict(model_file, df_test)
    print(predicts)

    df = pd.DataFrame({
        "order": orders,
        "pred": predicts
    })
    df.to_csv(DATA_DIR.joinpath("predicts.csv"),
              header=False, index=False)


if __name__ == '__main__':
    evaluate()
