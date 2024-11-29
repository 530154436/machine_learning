# /usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy

'''
    简单粗暴 TensorFlow 2 https://tf.wiki/zh_hans/basic/models.html
    https://greeksharifa.github.io/machine_learning/2019/12/21/FM/
    https://zhuanlan.zhihu.com/p/58508137
'''

class FM(tf.keras.Model):

    def __init__(self, input_dim, num_factors):
        super(FM, self).__init__()

        # 常数项、线性项、交叉项 => 系数
        self.w0 = tf.Variable([0.0], trainable=True)
        self.w1 = tf.Variable(tf.zeros([input_dim]), trainable=True)
        self.v = tf.Variable(tf.random.normal(shape=(input_dim, num_factors)), trainable=True)

        # 正则项系数
        self.lambda_w0 = tf.constant(0.01, name='lambda_w0')
        self.lambda_w1 = tf.constant(0.01, name='lambda_w1')
        self.lambda_v = tf.constant(0.01, name='lambda_v')

    def l2_norm(self):
        '''
        计算正则项 (交叉项不知道怎么算?!)
        '''
        l2 = tf.reduce_sum(
            tf.add(
                # tf.multiply(self.lambda_w0, tf.square(self.w0)),
                tf.reduce_sum(tf.multiply(self.lambda_w1,  tf.square(self.w1))),
                tf.reduce_sum(tf.multiply(self.lambda_v, tf.square(self.v)))
            )
        )
        return l2

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs: x=[b, input_dim]: b表示批次大小
        '''
        # [b,input_dim] => [b,]
        linear = tf.reduce_sum(tf.multiply(inputs, self.w1), axis=1)            # linear = \sum_{i=1}^n w_i \cdot x_i

        # [b,input_dim] => [ b,num_factors]
        dot = tf.matmul(inputs, self.v)                                         # dot = \sum_{i=1}^n v_{if} \cdot x_i
        square_dot = tf.matmul(tf.square(inputs), tf.square(self.v))            # square_dot = \sum_{i=1}^n v_{if}^2 \cdot x_i^2

        # [b,num_factors] => [b,]
        cross = 0.5 * tf.reduce_sum(tf.square(dot) - square_dot, axis=1)  # 对列求和: \sum_{f=1}^k(dot^2 - square_dot)

        y_hat = self.w0+linear+cross

        return tf.sigmoid(y_hat)

def train_on_batch(x, y_true, model:tf.keras.Model, optimizer):
    '''
    按批次训练(正向传播)
    :param x:           特征集
    :param y_true:      标签集
    :param model:       模型对象
    :param optimizer:   优化器
    :param norm         正则化项
    :return:
    '''
    # 计算梯度 => 相当于计算偏导数 \partial{loss}/\partial{\theta}
    with tf.GradientTape() as tape:
        y_hat = model.call(x)

        # loss = tf.losses.mean_squared_error(y, y_hat)
        loss = tf.losses.binary_crossentropy(y_true, y_hat, from_logits=False)
        loss += model.l2_norm()

    # 计算梯度
    grads = tape.gradient(loss, sources=model.trainable_variables)

    # 自动更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

    return y_hat, loss

def train_step_by_step(train_x_y, model:tf.keras.Model, epochs=10):
    '''
    TF低阶API
    :param train_x_y:   训练集
    :param model:       模型实例
    :param epochs:      迭代次数
    :return:
    '''
    accuracy = BinaryAccuracy(threshold=0.5)
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # 对整个数据集迭代
    for epoch in range(epochs):
        loss_history = []

        # 对每个批次迭代
        for x,y in train_x_y:
            y_hat, loss = train_on_batch(x, y, model, optimizer)

            # 评估
            accuracy.update_state(y, y_hat)
            loss_history.append(loss)

        print(f'epoch:{epoch}, loss:    {float(tf.reduce_mean(loss_history))}')
        print(f'epoch:{epoch}, accuracy:{float(accuracy.result())}\n')

    return model

def train(train_x_y, model:tf.keras.Model, val_xy=None, epochs=10, input_dim=None):
    '''
    TF高阶API
    :param train_x_y:   训练集
    :param model:       模型实例
    :param epochs:      迭代次数
    :return:
    '''
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