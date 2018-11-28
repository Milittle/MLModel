#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/2 10:46
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : test.py
# @Software: PyCharm

import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4]))
dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

sess = tf.Session()

# print(dataset2.output_shapes)
# print(dataset2.output_types)

# one-shot iterator

dataset3 = tf.data.Dataset.range(10).map(lambda x: x + tf.random_uniform([],minval=-10, maxval=10, dtype=tf.int64)).repeat(2)

iterator1 = dataset3.make_one_shot_iterator()

value = iterator1.get_next()

while True:
    try:
        print(sess.run(value))
    except tf.errors.OutOfRangeError:
        print("error")
        break

# initializable

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
# 必须显示run iterator.initializer
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
    value = sess.run(next_element)
    assert i == value
