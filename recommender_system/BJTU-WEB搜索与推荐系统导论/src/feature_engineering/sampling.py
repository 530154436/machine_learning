#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@author:zhengchubin
@time: 2021/04/20
@function:
"""
import pandas as pd
from typing import List


class Sampler(object):

    @classmethod
    def skip_above(cls, chunk: pd.DataFrame, if_neg_sampling: bool = True) -> List[int]:
        """
        正负样本构造：
        采用 skip-above 策略，即用户点过的Item之上，没有点过的Item作为负例。
        根据用户最后一次点击行为的位置，过滤掉最后一次点击之后的展示，可以人认为用户没有看到。
        (TODO)Skip Above同样存在着Ground Truth问题，具体的负样本可以有如下策略:
          1.随机抽取一定比例(baseline)。
          2.在用户未点击的部分，选择流行度高的作为负样本。
          3.在用户未点击的部分，删除用户近期已发生观看行为的物品。
          4.在用户未点击的部分，统计相应的曝光数据，取Top作为负样本。
        :param chunk:            1个用户的所有点击、曝光记录
        :param if_neg_sampling:
        """
        for column in ['item_id', 'ctx_time_expose', 'u_cnt_flush', 'ctx_show_pos', 'is_click']:
            assert column in chunk.columns

        # 曝光时间、刷新次数、展示位置、是否点击
        chunk = chunk.sort_values(by=['ctx_time_expose', 'u_cnt_flush', 'ctx_show_pos'],
                                  ascending=[True, True, True])

        # shift(1): 向下偏移1行, 正数则为一个批次的开始位置
        chunk['ctx_show_pos_diff'] = (chunk["ctx_show_pos"].shift(1) - chunk["ctx_show_pos"]).fillna(1)
        # print(chunk)

        positive_cnt = 0
        selected_indices = []
        candidates = []
        for index, pos_diff, is_click in zip(chunk.index, chunk['ctx_show_pos_diff'], chunk['is_click']):
            # 新批次开始，清空缓存
            if pos_diff > 0:
                candidates.clear()
            # 最后一次点击位置之前的样本保留
            candidates.append(index)
            if is_click == 1:
                positive_cnt += 1
                selected_indices.extend(candidates)
                # print(candidates)
                candidates.clear()

        # 负样本采样中： 在用户未点击的部分，统计相应的曝光数据，取Top作为负样本。
        if if_neg_sampling:
            df_expose = chunk.loc[chunk['is_click'] == 0]
            if not df_expose.empty:
                df_expose = df_expose.groupby(by=["item_id"]).agg(
                    un_click=pd.NamedAgg(column="is_click", aggfunc="count")
                )
                df_expose = df_expose.reset_index()
                df_expose = df_expose.sort_values(by=['un_click'], ascending=[False])
                df_expose = df_expose[df_expose['un_click'] > 1]

            # 正负样本比例: 1:10
            if not df_expose.empty:
                need_neg = positive_cnt * 11 - len(candidates)
                negs = df_expose["item_id"].tolist()[:need_neg]
                neg = chunk.loc[chunk['item_id'].isin(negs)].index.tolist()
                selected_indices.extend(neg)
                selected_indices = list(set(selected_indices))

        return selected_indices
