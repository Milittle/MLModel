#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/20 22:14
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : base.py
# @Software: PyCharm

from keras.models import Sequential
from keras.layers import Dense, Activation
import keras


### 一、构建序贯模型方式

# 第一种方式，直接用Sequential list的形式构建model
def one_list():
    model = Sequential(
        [
            Dense(32, input_shape=(784,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax'),
        ]
    )
    return model


# 第二种方式，使用Sequential 的add方法来添加每一层，构建model
def step_by_step():
    model = Sequential()
    model.add(Dense(32, input_shape=(784, )))
    model.add(Activation('relu'))

    return model


# 大家可以注意到，上面的输入层（也就是模型的第一层）都会定义shape，也就是输入大小
# 这样后面的各层才能推导出中间数据的shape，因此中间层不需要指出数据的shape（可以自动推导出中间的shape）
# 有下面几种方式指定输入层的shape
# 1. 传递一个input_shape参数给第一层，input_shape是一个tuple类型的数据，可以填入None。
# 如果填入None则表示，此位置可能是任意正整数，数据的batch大小不应该包含在这个参数中，也就是只包括单个batch数据的大小
# 2. 对于2D层数据来说，利用input_dim参数就可以指定shape， 对于3D时域层来讲需要使用input_dim和input_length来指定shape
# 3. 如果需要指定固定batch_size的输入，那么需要指定batch_size的值和input_shape()


### 二、编译模型，这个过程是通过model的compile()函数来实现的。

# compile函数有三个参数，
'''
    :param
    优化器     optimizer 
               该参数可指定为预定义的优化器名（字符串形式）
               或者是一个Optimizer的类对象，详情请参照optimizers
    损失函数   loss
                模型最小化的函数，可指定为预定义的损失函数（字符串形式）
                或者为一个损失函数，详情可见losses
    指标列表   metrics
                对于分类问题来讲，我们一般设置metrics=['accuracy']
                这个指标可以是预定义好的名字（字符串形式）
                也可以是一个用户定制的函数，指标函数应该返回单个张量， 或者一个完成metric_name -> metric_value映射的字典
                详情请参照性能评估          
'''

# 大家也看到了，其实keras构建模型很简单，但是也是死板，固定层堆叠，
# 但是后续会有很多不同的层出现，也是很不错的

# 多分类任务loss编译
def categorical_compile(model = Sequential()):
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 二分类任务loss编译
def binary_compile(model = Sequential()):
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# 回归任务loss编译
def mse_compile(model = Sequential()):
    model.compile(optimizer='rmsprop',
                  loss='mse')

    return model

# 自定义metrics 二分类编译
def custom_metrics_compoile(model = Sequential()):
    import keras.backend as K
    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer='resprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', mean_pred])
    return model

# 三、也是最重要的训练部分
'''
    keras是以numpy数组作为输入数据和标签的数据类型，训练模型一般使用fit函数
    该函数的详细信息，请见https://keras-cn.readthedocs.io/en/latest/models/sequential/
    例子如下：
'''


# 很明显看出来fit方法适合一次性加载数据，但是一般对于视觉任务来讲，
# 一次性加载数据是不可能的事情（因为数据量太大了，根本没法一次性加载到内存中），所以后面会讲到逐步
# 加载数据的方式，以及训练的过程。
def train(model = Sequential()):
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    model.fit(data, labels, epochs=10, batch_size=32)

# 下面是一个完整的例子，是一个有一千个数据，分十类的小例子
# 一层输入层，一层隐藏层、一层输出层，都是全连接
# 输入层shape为100，隐藏层为32，输出层也就是预测层10类

def ttt():
    # 构建网络模型
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 生成数据
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))

    # 转化普通的数字为one_hot编码格式
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    # 训练模型，一次取32批数据进行一次迭代训练
    model.fit(data, one_hot_labels, epochs=1000, batch_size=32)

if __name__ == '__main__':
    ttt()



