#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from typing import List

from src.framework import detect_device
from src.config_gpu import BASE_DIR


class NerConfig(object):

    def __init__(self):

        self.device = detect_device()

        # 预训练、数据集配置
        # self.pre_train_path: str = "/data/modelfiles/chinese-bert-wwm-ext"
        self.pre_train_path: str = BASE_DIR.joinpath("data", "modelfiles", "chinese-bert-wwm-ext")
        self.batch_size: int = 32
        self.max_seq_length: int = 100
        self.label_list: List[str] = ["O", "B-NT", "I-NT"]

        self.lstm_num_layers = 1
        self.lstm_hidden_size = 128
        self.dropout = 0.3
        self.num_labels = len(self.label_list)
        self.model_file = BASE_DIR.joinpath("data", "modelfiles", "wzalgo_recommender_nlp", "content_ner.pth")
