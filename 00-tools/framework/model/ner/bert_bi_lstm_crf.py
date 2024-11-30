#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2023/9/6 20:37
# @function:
from typing import List

import torch
from torch import nn
from transformers import BertModel
from torchcrf import CRF
from src.framework.cfgs.ner_config import NerConfig


class BertBiLstmCrf(nn.Module):
    """
    """
    def __init__(self, param: NerConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(param.pre_train_path)
        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                              bidirectional=True,
                              num_layers=param.lstm_num_layers,
                              hidden_size=param.lstm_hidden_size,
                              batch_first=True)
        self.fc = nn.Linear(in_features=param.lstm_hidden_size * 2, out_features=param.num_labels)
        self.crf = CRF(num_tags=param.num_labels, batch_first=True)
        self.dropout = nn.Dropout(param.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                segment_ids: torch.Tensor, labels: torch.Tensor = None) -> tuple:
        # [batch_size, max_seq_length, lstm_hidden_size * 2] => [64, 100, 256]
        embeds = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]

        # [batch_size, max_seq_length, lstm_hidden_size * 2] => [64, 100, 256]
        lstm_out, _ = self.bilstm(embeds)
        lstm_out = self.dropout(lstm_out)

        # [batch_size, max_seq_length, num_labels] => [64, 100, 3]
        lstm_out = self.fc(lstm_out)

        # 计算损失
        loss = None
        if labels is not None:
            loss = -self.crf(lstm_out, labels, attention_mask.bool(), reduction="mean")

        # 预测解码
        predicts: List[list] = self.crf.decode(lstm_out, mask=attention_mask.bool())

        return loss, predicts
