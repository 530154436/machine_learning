# /usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy

class DeepFM(tf.keras.Model):

    def __init__(self, input_dim, num_factors):
        super(DeepFM, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass