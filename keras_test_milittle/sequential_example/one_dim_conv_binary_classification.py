#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/20 23:09
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : one_dim_conv_binary_classification.py
# @Software: PyCharm

seq_length = 10

def model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
    import numpy as np


    x_train = np.random.random((1000, seq_length, 100))
    x_train = np.reshape(x_train, newshape=([1000, seq_length, 100]))
    y_train = np.random.randint(2, size = (1000, 1))

    x_test = np.random.random((200, seq_length, 100))
    y_test = np.random.randint(2, size = (200, 1))


    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(x_train, y_train, batch_size=16, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=16)
    metrics_names = model.metrics_names


if __name__ == '__main__':
    score, metrics_names = model()
    print(score, metrics_names)
