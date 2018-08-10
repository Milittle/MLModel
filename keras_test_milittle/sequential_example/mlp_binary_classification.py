#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/20 23:06
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : mlp_binary_classification.py
# @Software: PyCharm
#

def model():
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    # Generate dummy data
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
    x_test = np.random.random((100, 20))
    y_test = np.random.randint(2, size=(100, 1))

    # x_test = x_train[:200, :]
    # y_test = y_train[:200, :]

    model = Sequential()
    model.add(Dense(1024, input_dim=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2056, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              epochs=1000,
              batch_size=1000)
    score = model.evaluate(x_test, y_test, batch_size=200)
    metrics_names = model.metrics_names

    return score, metrics_names


if __name__ == '__main__':
    print(model())
