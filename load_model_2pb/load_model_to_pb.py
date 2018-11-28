#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 8:18
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : load_model_to_pb.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.python.tools import freeze_graph


model_path  = '../datasets_test/cifar/model/model.ckpt-832'


def main():

    saver = tf.train.import_meta_graph('../datasets_test/cifar/model/model.ckpt-832.meta')

    with tf.Session() as sess:
        last_point = tf.train.latest_checkpoint('../datasets_test/cifar/model/')
        saver.restore(sess, last_point)
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path, 'add','save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")

if __name__ == '__main__':
    main()