#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 10:07
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : drop_tf_dropoutL.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.core.framework import graph_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_frozen_graph_path', './pb_model/resnetv2_imagenet_frozen_graph.pb', '')
tf.app.flags.DEFINE_string('output_frozen_graph_path', './output_model/pb_model/resnetv2_imagenet_frozen_graph.pb', '')


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]

def open_graph_pb():
    graph = tf.GraphDef()
    with tf.gfile.Open(FLAGS.input_frozen_graph_path, 'rb') as f:
        data = f.read()
        graph.ParseFromString(data)
    return graph

def display_graph_node(graph):
    display_nodes(graph.node)


def remove_dropout_node_and_save(graph):
    graph.node[36].input[0] = 'fc1/relu1'

    nodes = graph.node[:2] + graph.node[6:28] + graph.node[36:]

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes)
    with tf.gfile.GFile(FLAGS.output_frozen_graph_path, 'wb') as f:
        f.write(output_graph.SerializeToString())
        print('****************Save the without pb file done!!!**************')

def remove_resnet_dropout(graph):
    nodes = graph.node[:701] + graph.node[-1:]
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes)

    with tf.gfile.GFile(FLAGS.output_frozen_graph_path, 'wb') as f:
        f.write(output_graph.SerializeToString())
        print('****************Save the without pb file done!!!**************')


def main(_):
    graph = open_graph_pb()
    display_graph_node(graph)
    # remove_dropout_node_and_save(graph)
    remove_resnet_dropout(graph)

if __name__ == '__main__':
    tf.app.run()





