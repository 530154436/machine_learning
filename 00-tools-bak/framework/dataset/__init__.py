#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2023/9/8 17:02
# @function:
from collections import namedtuple

SparseFeat = namedtuple('SparseFeat', ['name', 'nunique', 'embed_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dim'])
BertFeat = namedtuple('BertFeat', ['input_ids', 'input_mask', 'segment_ids', 'label_ids'])
