#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23 9:50
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : test.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np

sess = tf.Session()

a = np.array([[1,2,3,4],[4,5,6,7]])

a = tf.to_float(a)

nor = tf.nn.l2_normalize(a, axis=0)

print(sess.run(nor))
