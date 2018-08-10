#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/20 23:08
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : lstm_binary_classification.py
# @Software: PyCharm

max_features = 1024

def model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import LSTM
    import numpy as np

    x_train = np.random.random((1000, 1024))
    y_train = np.random.randint(2, size = (1000, 1))

    x_test = np.random.random((200, 1024))
    y_test = np.random.randint(2, size = (200, 1))

    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=16, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=16)

    metrics_names = model.metrics_names

    return score, metrics_names

if __name__ == '__main__':
    score, metrics_names = model()
    print(score, metrics_names)