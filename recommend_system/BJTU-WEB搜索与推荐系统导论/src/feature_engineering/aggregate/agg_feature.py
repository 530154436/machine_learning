#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import pandas as pd

from bjtu_programming.search_rs.src.feature_engineering.context.time import DateConverter, DateUtil


class AggregateFeature(object):

    @classmethod
    def group_by(cls, df: pd.DataFrame) -> pd.DataFrame:
        pass
