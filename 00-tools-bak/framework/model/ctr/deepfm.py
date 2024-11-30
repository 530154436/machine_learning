#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import torch
from torch import nn
from typing import List

from src.framework.dataset import DenseFeat, SparseFeat
from src.framework.basic.layers import DNN, FM

"""
参考 https://github.com/shenweichen/DeepCTR-Torch
"""


class DeepFM(nn.Module):
    """Instantiates the DeepFM Network architecture.
    :param sparse_feat_nuniqs: An iterable containing sparse features used by linear part of the model.
    :param dense_columns: An iterable containing dense features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.
    """
    def __init__(self, dense_feats: List[DenseFeat], sparse_feats: List[SparseFeat], l2_reg_embedding=0.00001,
                 dnn_hidden_units=(256, 128, 64), l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=True, device='cpu'):
        super(DeepFM, self).__init__()
        self.dense_feats = dense_feats
        self.sparse_feats = sparse_feats
        self.input_dim = len(dense_feats) + sum(feat.embed_dim for feat in sparse_feats)

        # 一阶: 数值+离散
        self.linear = nn.Linear(self.input_dim, 1)
        # FM二阶交叉
        self.fm = FM()
        self.sparse_emb_layers = nn.ModuleList()  # 类别特征的二阶表示
        for feat in sparse_feats:
            self.sparse_emb_layers.append(nn.Embedding(num_embeddings=feat.nunique, embedding_dim=feat.embed_dim))
        # DNN部分
        self.dnn = DNN(self.input_dim, dnn_hidden_units, activation=dnn_activation,
                       l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)

        # self.add_regularization_weight(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

        self.to(device)

    def forward(self, dense_inputs, sparse_inputs=None):
        """
        inputs
        dense_inputs:  数值型特征输入（可能没有）  [bs, dense_inputs]
        sparse_inputs: 类别型特征输入  [bs, sparse_inputs]
        """
        # dense_inputs, sparse_inputs = inputs
        sparse_embed = []
        for i, (feat, embed_layer) in enumerate(zip(self.sparse_feats, self.sparse_emb_layers)):
            # print(i, embed_layer, feat, sparse_inputs[:, i].max())
            # [sparse_feat_size, embedding]
            sparse_embed.append(embed_layer(sparse_inputs[:, i].unsqueeze(1)))

        # [bs, sparse_feat_size, embedding]
        sparse_embed = torch.cat(sparse_embed, dim=1)

        # [bs, embedding]
        x = torch.cat((dense_inputs, torch.flatten(sparse_embed, 1)), dim=-1)

        # 一阶部分
        logit = self.linear(x)
        # FM二阶部分
        logit += self.fm(sparse_embed)
        # DNN部分
        logit += self.dnn(x)
        return torch.sigmoid(logit)
