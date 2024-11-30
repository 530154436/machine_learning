#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from typing import List

from src.framework import detect_device


class NerConfig(object):

    def __init__(self):

        device = detect_device()

        # 预训练、数据集配置
        self.pre_train_path: str = "/data/modelfiles/bert-base-chinese"
        self.batch_size: int = 64
        self.max_seq_length: int = 100
        self.label_list: List[str] = ["O", "B-NT", "I-NT"]

        self.lstm_num_layers = 1
        self.lstm_hidden_size = 128
        self.dropout = 0.1
        self.num_labels = len(self.label_list)
        self.model_file = "ner.pth"
