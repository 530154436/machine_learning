# /usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy

class FFM(tf.keras.Model):

    def __init__(self, input_dim, num_factors):
        super(FFM, self).__init__()

    def l2_norm(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass