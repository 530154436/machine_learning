#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2021/5/26 8:58
# @function:
# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras


class SVD(keras.Model):
    """
        隐语义模型的矩阵分解方法(LFM/Funk-SVD)+正则项+偏置项
        simon-funk博客       https://sifter.org/~simon/journal/20061211.html
        surprise            https://surprise.readthedocs.io/en/stable/matrix_factorization.html
        基于矩阵分解的协同过滤  https://mathpretty.com/11495.html
        Numpy实现库          https://github.com/lxmly/recsyspy/tree/master/
        TF2实现库            https://github.com/Praful932/Tf-Rec
    """
    def __init__(self, n_users, n_items, global_mean, embedding_dim=16, biased=True, **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = np.float32(global_mean)
        self.embedding_dim = embedding_dim
        self.biased = biased

        # 用户隐向量 P(mxk) trainable
        self.user_embedding = keras.layers.Embedding(input_dim=self.n_users,
                                                     output_dim=self.embedding_dim,
                                                     embeddings_initializer=keras.initializers.RandomNormal(),
                                                     embeddings_regularizer=keras.regularizers.l2())
        # 物品隐向量 Q(nxk)
        self.item_embedding = keras.layers.Embedding(input_dim=self.n_items,
                                                     output_dim=self.embedding_dim,
                                                     embeddings_initializer=keras.initializers.RandomNormal(),
                                                     embeddings_regularizer=keras.regularizers.l2())
        if biased:
            # 用户偏置，表示某一特定用户的行为习惯。 Pu (mx1)
            self.user_bias = keras.layers.Embedding(input_dim=self.n_users,
                                                    output_dim=1,
                                                    embeddings_initializer=keras.initializers.Zeros(),
                                                    embeddings_regularizer=keras.regularizers.l2())

            # 物品偏置，表示某一特定物品得到的点击情况。 Qi (nx1)
            self.item_bias = keras.layers.Embedding(input_dim=self.n_items,
                                                    output_dim=1,
                                                    embeddings_initializer=keras.initializers.Zeros(),
                                                    embeddings_regularizer=keras.regularizers.l2())

    def call(self, inputs, training=None, mask=None):
        """
        正向传播: 计算预测值
        Args:
            inputs:     输入数据二元组 (用户索引, Item索引)
            training:   是否为训练模式
            mask:
        """
        user, item = inputs[:, 0], inputs[:, 1]
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        # \hat{r} = Pu*Qj
        rating = tf.reduce_sum(
            tf.multiply(user_embed, item_embed), axis=1, keepdims=True)  # 点乘

        # \hat{r} = Pu*Qj + Pu + Qj + \mu
        if self.biased:
            u_bias, i_bias = self.user_bias(user), self.item_bias(item)
            _sum = tf.add(self.global_mean, tf.add(u_bias, i_bias))
            rating = tf.add(rating, _sum)

        return tf.sigmoid(rating)


def train(train_x_y, model: tf.keras.Model, epochs=10, input_dim=None):
    """
    TF高阶API
    :param train_x_y:   训练集
    :param model:       模型实例
    :param epochs:      迭代次数
    :return:
    """
    if input_dim:
        model.build(input_shape=(None, input_dim))
        print(model.summary())

    # 为训练选择优化器和损失函数
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=tf.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['binary_accuracy'])

    # 训练
    # model.fit(train_x_y, epochs=epochs, validation_data=val_xy, validation_freq=5)
    model.fit(train_x_y, epochs=epochs)
    return model
