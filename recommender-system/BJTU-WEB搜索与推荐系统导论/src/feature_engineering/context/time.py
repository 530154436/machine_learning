#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 特征工程系列：时间特征构造以及时间序列特征构造
# https://cloud.tencent.com/developer/article/1536537
import pytz
import pandas as pd
from typing import Union, Optional
from datetime import datetime

TIME_ZONE = pytz.timezone('Asia/Shanghai')

# 一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
PERIOD_DICT = {
    23: '深夜', 0: '深夜', 1: '深夜',
    2: '凌晨', 3: '凌晨', 4: '凌晨',
    5: '早晨', 6: '早晨', 7: '早晨',
    8: '上午', 9: '上午', 10: '上午', 11: '上午',
    12: '中午', 13: '中午',
    14: '下午', 15: '下午', 16: '下午', 17: '下午',
    18: '傍晚',
    19: '晚上', 20: '晚上', 21: '晚上', 22: '晚上',
}
PERIODS = ['凌晨', '上午', '傍晚', '深夜', '下午', '晚上', '中午', '早晨']

# 一年中的哪个季度
SEASON_DICT = {
    1: '春季', 2: '春季', 3: '春季',
    4: '夏季', 5: '夏季', 6: '夏季',
    7: '秋季', 8: '秋季', 9: '秋季',
    10: '冬季', 11: '冬季', 12: '冬季',
}


class DateConverter(object):

    @classmethod
    def timestamp_to_datetime(cls, timestamp: Union[int, str], unit: str = "ms") -> Optional[datetime]:
        """
        获取当前日期
        """
        if pd.isnull(timestamp):
            return None
        elif isinstance(timestamp, (str, float)):
            timestamp = int(timestamp)
        if unit == "ms":
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=TIME_ZONE)


class DateUtil(object):

    @classmethod
    def get_day_of_week(cls, date: datetime) -> int:
        """
        获取星期几：[0, 6]
        """
        if not isinstance(date, datetime):
            raise TypeError("date must be datetime Type, not %s" % type(date))
        return date.weekday()

    @classmethod
    def get_hour(cls, date: datetime) -> int:
        """
        获取一天中的某个小时
        """
        if not isinstance(date, datetime):
            raise TypeError("date must be datetime Type, not %s" % type(date))
        return date.hour

    @classmethod
    def get_hour_bucket(cls, date: datetime) -> int:
        """
        获取一天中的时间段
        """
        return PERIODS.index(PERIOD_DICT.get(cls.get_hour(date)))


if __name__ == '__main__':
    dt = DateConverter.timestamp_to_datetime(1624892884920)
    print(dt)
    print(DateUtil.get_day_of_week(dt))
    print(DateUtil.get_hour_bucket(dt))

