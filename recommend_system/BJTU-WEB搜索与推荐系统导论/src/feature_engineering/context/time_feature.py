#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd

from bjtu_programming.search_rs.src.feature_engineering.context.time import DateConverter, DateUtil


class TimeFeature(object):
    column = 'ctx_time_expose'
    fea_columns = [
        f'{column}_hour',
        f'{column}_hour_bucket',
        f'{column}_is_opening_hours',
        f'{column}_day_of_week',
        f'{column}_is_weekend',
    ]

    @classmethod
    def pipline(cls, df: pd.DataFrame):

        # 时间特征
        df[cls.column] = df[cls.column].apply(DateConverter.timestamp_to_datetime)

        # 一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
        df[f'{cls.column}_hour_bucket'] = df[cls.column].apply(DateUtil.get_hour_bucket)

        # 是否工作时间
        df[f'{cls.column}_hour'] = df[cls.column].apply(DateUtil.get_hour)
        df[f'{cls.column}_is_opening_hours'] = 0
        mask = ((df[f'{cls.column}_hour'] >= 8) & (df[f'{cls.column}_hour'] < 22))
        df.loc[mask, f'{cls.column}_is_opening_hours'] = 1

        # 星期几
        df[f'{cls.column}_day_of_week'] = df[cls.column].apply(DateUtil.get_day_of_week)

        # 是否周末
        df[f'{cls.column}_is_weekend'] = df[f'{cls.column}_day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

    @classmethod
    def pipline_parallel(cls, df: pd.DataFrame):

        # 时间特征
        df[cls.column] = df[cls.column].parallel_apply(DateConverter.timestamp_to_datetime)

        # 一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
        df[f'{cls.column}_hour_bucket'] = df[cls.column].parallel_apply(DateUtil.get_hour_bucket)

        # 是否工作时间
        df[f'{cls.column}_hour'] = df[cls.column].parallel_apply(DateUtil.get_hour)
        df[f'{cls.column}_is_opening_hours'] = 0
        mask = ((df[f'{cls.column}_hour'] >= 8) & (df[f'{cls.column}_hour'] < 22))
        df.loc[mask, f'{cls.column}_is_opening_hours'] = 1

        # 星期几
        df[f'{cls.column}_day_of_week'] = df[cls.column].parallel_apply(DateUtil.get_day_of_week)

        # 是否周末
        df[f'{cls.column}_is_weekend'] = df[f'{cls.column}_day_of_week'].parallel_apply(lambda x: 1 if x in [5, 6] else 0)
