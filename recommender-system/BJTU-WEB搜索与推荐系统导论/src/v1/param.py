#!/usr/bin/env python3
# -*- coding:utf-8 -*--
class SampleParam(object):
    """
    采样参数配置
    """
    def __init__(self, **kwargs):
        self.if_correct_sample = kwargs.get('if_correct_sample', 0)
        self.if_skip_above = kwargs.get('if_skip_above', 0)
        self.min_cnt_read_duration = kwargs.get('min_cnt_read_duration', 3)

    def __str__(self):
        return self.__class__.__name__ + \
               f": if_correct_sample{self.if_correct_sample}, " \
               f"if_skip_above={self.if_skip_above}," \
               f"min_cnt_read_duration={self.min_cnt_read_duration}"

    __repr__ = __str__


# class TrainParam(SampleParam):
#     """
#     训练参数配置
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def __str__(self):
#         return self.__class__.__name__ + \
#                super().__str__()
#
#     __repr__ = __str__
