#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
from typing import List
import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizer
from seqeval.metrics.sequence_labeling import get_entities

from src.config_gpu import timer, logger, print_gpu_memory_usage
from src.framework.cfgs.ner_config import NerConfig
from src.framework.dataset.ner.ner_processing import convert_text_to_features_v2
from src.framework.model.ner.bert_bi_lstm_crf import BertBiLstmCrf


class Predictor4NerCrf:
    def __init__(self, param: NerConfig):
        self.param = param
        self.model: nn.Module = None
        self.label_idx2name: dict = {i: name for i, name in enumerate(param.label_list)}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(param.pre_train_path)
        self.setup()

    @print_gpu_memory_usage
    def setup(self):
        logger.info(f"加载模型中...文件路径: {self.param.model_file}")
        self.model = BertBiLstmCrf(self.param)
        self.model.load_state_dict(torch.load(self.param.model_file, map_location=self.param.device))
        self.model.to(self.param.device)
        self.model.eval()
        logger.info(f"模型加载成功！使用 {self.param.device}, 文件路径: {self.param.model_file}")

    @timer
    @torch.no_grad()
    def predict(self, sentences: List[str]) -> List:
        """
        序列标注预测
        """
        if not sentences:
            return []
        input_ids, input_mask, segment_ids = convert_text_to_features_v2(sentences,
                                                                         tokenizer=self.tokenizer,
                                                                         max_seq_length=self.param.max_seq_length)
        input_ids = input_ids.to(self.param.device)
        input_mask = input_mask.to(self.param.device)
        segment_ids = segment_ids.to(self.param.device)
        _, y_predicts = self.model(input_ids, input_mask, segment_ids)

        del input_ids, input_mask, segment_ids

        # 获取真实的标签名称
        result = []
        for sentence, y_pred in zip(sentences, y_predicts):
            # tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            # 构造输入为 [CLS] xxx [SEP]，所以需要去掉前后的特殊字符
            labels = [self.label_idx2name.get(y) for y in y_pred[1: len(y_pred)-1]]
            entities = get_entities(labels)
            result_sub = []
            for (label, start, end) in entities:
                result_sub.append({
                    "text": sentence[start: end+1],
                    "label": label,
                    "start": start,
                    "end": end,
                })
            result.append(result_sub)
        return result

    @torch.no_grad()
    def evaluate(self, data_loader, device="cpu"):
        """
        序列标注离线评估
        """
        y_trues, y_preds = list(), list()
        for batch, (xy) in enumerate(data_loader, start=1):
            xy_tuple = tuple(x.to(device) for x in xy)
            loss, y_pred = self.model(*xy_tuple)

            # 真实的标签
            input_ids, input_mask, segment_ids, y_true = xy
            y_true = torch.masked_select(y_true, input_mask.bool())
            y_trues.extend(y_true.tolist())

            # 预测的标签
            for sample_pred in y_pred:
                y_preds.extend(sample_pred)

            assert len(y_trues) == len(y_preds)

        from sklearn.metrics import classification_report
        print(classification_report(y_trues, y_preds, target_names=self.param.label_list, zero_division=0))
