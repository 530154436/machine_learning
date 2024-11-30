#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2023/9/8 17:45
# @function:
from typing import List, Tuple

import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.framework.cfgs.ner_config import NerConfig
from src.framework.dataset.ner.ner_processing import NERDataProcessor, convert_example2features


class NERDataLoader(object):

    def __init__(self, param: NerConfig, corpus_path: Tuple[str, str, str]):
        """
        :param param: 配置信息
        :param corpus_path: (训练集文件路径, 验证集文件路径, 测试集文件路径)
        :return: [(sentence, label), ...]
        """
        self.corpus_path = corpus_path

        # 数据集配置
        self.pre_train_path: str = param.pre_train_path
        self.batch_size: int = param.batch_size
        self.max_seq_length: int = param.max_seq_length
        self.label_list: List[str] = param.label_list

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(param.pre_train_path)
        self.train_iter, self.val_iter, self.test_iter = None, None, None

    def _create_dataset(self, file_path: str) -> TensorDataset:
        """
        构建数据集
        :param file_path: 文件路径
        :return:
        """
        samples = NERDataProcessor.read_sample_from_file(file_path)
        features = convert_example2features(samples,
                                            tokenizer=self.tokenizer,
                                            max_seq_length=self.max_seq_length,
                                            label_list=self.label_list)
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids = [], [], [], []
        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)
        dataset = TensorDataset(torch.LongTensor(all_input_ids),
                                torch.LongTensor(all_input_mask),
                                torch.LongTensor(all_segment_ids),
                                torch.LongTensor(all_label_ids))
        return dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = self._create_dataset(self.corpus_path[0])
        self.train_iter = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return self.train_iter

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = self._create_dataset(self.corpus_path[1])
        self.val_iter = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return self.val_iter

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = self._create_dataset(self.corpus_path[2])
        self.test_iter = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return self.test_iter


if __name__ == "__main__":
    from src.config_gpu import BASE_DIR
    _corpus_path = (BASE_DIR.joinpath("data", "content_ner", "ner_annotations.train"),
                    BASE_DIR.joinpath("data", "content_ner", "ner_annotations.dev"),
                    BASE_DIR.joinpath("data", "content_ner", "ner_annotations.test"))
    dataloader = NERDataLoader(NerConfig(), _corpus_path)
    td = dataloader.train_dataloader()
