# /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tools import dataset
from factorization_machines.fm import FM,train_step_by_step,train

os.environ["TF_CPP_MIN_LOG_LEVEL"]='10' # 只显示 warning 和 Error

def main(model, x_tr, x_te, y_tr, y_te, epochs=100):
    ''''
    训练模型模板
    '''
    x_tr = tf.convert_to_tensor(x_tr, dtype=tf.float32)
    y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32)
    x_te = tf.convert_to_tensor(x_te, dtype=tf.float32)
    y_te = tf.convert_to_tensor(y_te, dtype=tf.float32)

    train_x_y = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(500).batch(20)  # A `Dataset`
    test_x_y = tf.data.Dataset.from_tensor_slices((x_te, y_te)).shuffle(500).batch(20)

    # 训练
    train(train_x_y, model, epochs=epochs, input_dim=x_tr.shape[1])
    train_step_by_step(train_x_y, model, epochs=epochs)

    # 评估
    # model.evaluate(test_x_y)
    accuracy = BinaryAccuracy(threshold=0.5)
    for x, y in test_x_y:
        y_hat = model.call(x)
        accuracy.update_state(y, y_hat)
    print(f'Test accuracy:{float(accuracy.result())}\n')

if __name__ == '__main__':
    # 加载数据(需归一化)
    x_tr, x_te, y_tr, y_te = dataset.load_breast_cancer(test_size=0.2, normalization=True)
    model = FM(input_dim=x_tr.shape[1], num_factors=10)

    main(model,x_tr, x_te, y_tr, y_te)
