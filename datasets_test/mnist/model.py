#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 20:08
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : model.py
# @Software: PyCharm
#coding=utf-8


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter_step', 1000, 'define iteration times')
tf.app.flags.DEFINE_integer('batch_size', 128, 'define batch size')
tf.app.flags.DEFINE_integer('classes', 10, 'define classes')
tf.app.flags.DEFINE_float('keep_drop', 0.5, 'define keep dropout')
tf.app.flags.DEFINE_float('lr', 0.001, 'define learning rate')
tf.app.flags.DEFINE_string('model_path', 'model\\','define model path')
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', 'define model name')
tf.app.flags.DEFINE_string('meta_graph_name', 'model.meta', 'define model name')
tf.app.flags.DEFINE_bool('use_model', False, 'define use_model sign')
tf.app.flags.DEFINE_bool('is_train', False, 'define train sign')
tf.app.flags.DEFINE_bool('is_test', False, 'define train sign')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define W & b
def weight_variable(para, name):
    # 采用截断的正态分布，标准差stddev＝0.1
    initial = tf.truncated_normal(para,stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(para, name):
    initial = tf.constant(0.1, shape=para)
    return tf.Variable(initial, name)

# define conv & pooling
def conv2d(x,W):
    return tf.nn.conv2d( x,W,strides=[1,1,1,1],padding='SAME' )

def max_pool_2(x, name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name=name)

def network():

    # define the placeholder by using feed the data
    with tf.name_scope('input_placeholder'):
        x = tf.placeholder(tf.float32, [None, 784], 'x')  # 28*28=784 dim
        x_input = tf.reshape(x, [-1, 28, 28, 1], 'x_reshape')  # reshape for conv, -1表示不固定数量，1为通道数
        y_label = tf.placeholder(tf.float32, [None, FLAGS.classes], 'y_label')  # label - 10 dim

    # define convolution layer1
    with tf.name_scope('conv_layer1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='w_conv_1')  # Weight in:1  out:32
        b_conv1 = bias_variable([32], name='b_conv_1')  # bias
        h_relu1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1, name='relu_1')  # relu
        h_pool1 = max_pool_2(h_relu1, name='pool_1')  # pool after relu1

    # define convolution layer2
    with tf.name_scope('conv_layer2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='w_conv_2')  # Weight in:32  out:64
        b_conv2 = bias_variable([64], name='b_conv_2')  # bias for 64 kernel
        h_relu2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='relu_2')  # relu
        h_pool2 = max_pool_2(h_relu2, name='pool_2')  # pool after relu2

    # define the first FC layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='w_fc1')  # Weight in:7*7res*64  out:1024
        b_fc1 = bias_variable([1024], name='b_fc1')  # bias for 1024
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='pool1')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='relu1')

    # adding the dropout, in order to restrain overfitting
    with tf.name_scope('drop_out'):
        keep_prob = tf.placeholder(tf.float32, name='drop_out_placeholder')
        drop_fc1 = tf.nn.dropout(h_fc1, keep_prob, name='drop_out_fc')

    # define the second FC layer, by using softmax
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, FLAGS.classes], name='w_fc2')  # Weight in:1024  out:10
        b_fc2 = bias_variable([FLAGS.classes], name='b_fc2')  # bias for 10, 10类划分
        y = tf.nn.softmax(tf.matmul(drop_fc1, W_fc2) + b_fc2, name='y_out')  # 计算结果

    global_step = tf.Variable(0, trainable=False)

    # define the loss
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]), name='cross_entropy')
    with tf.name_scope('train_op'):
        train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(cross_entropy,
                                                               global_step=global_step,
                                                               name='train_operation')  # Adam 替代SGD

    # define the accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1), name='condition')
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    return x, y, keep_prob, y_label, train_step, accuracy, global_step

def train():

    # the sign which save the meta graph, just once.
    a = False
    x, y, keep_prob, y_label, train_step, accuracy, global_step = network()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)

    if FLAGS.use_model:
        model_t = tf.train.latest_checkpoint(FLAGS.model_path)
        saver.restore(sess, model_t)

    for i in range(FLAGS.max_iter_step):
        batch = mnist.train.next_batch(FLAGS.batch_size)  # 每50个一个batch
        if i % 100 == 0:
            # eval执行过程－训练精度
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1.0})
            print("step {step}, training accuracy {acc}".format(step=i, acc=train_accuracy))
            if (train_accuracy > 0.5):
                if a == 0:
                    saver.export_meta_graph(FLAGS.model_path + FLAGS.meta_graph_name)
                    a = True
                saver.save(sess, FLAGS.model_path + FLAGS.model_name, global_step=global_step, write_meta_graph=False)
        sess.run(train_step, feed_dict={x: batch[0], y_label: batch[1], keep_prob: FLAGS.keep_drop})

def test():

    if FLAGS.use_model:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(FLAGS.model_path + FLAGS.meta_graph_name)
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))

            graph = tf.get_default_graph()


            # one operation possibly have many outputs, so you need specify the which output, such as "name:0"
            x = graph.get_tensor_by_name("input_placeholder/x:0")
            y_label = graph.get_tensor_by_name("input_placeholder/y_label:0")
            keep_prob = graph.get_tensor_by_name("drop_out/drop_out_placeholder:0")
            accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")

            feed_dict = {x: mnist.test.images,
                         y_label: mnist.test.labels,
                         keep_prob: 1.0}

            acc = sess.run(accuracy, feed_dict=feed_dict)
            print("test accuracy {acc:.4f}".format(acc=acc))
    else:
        return

def save_pb_file():

    if FLAGS.use_model:
        saver = tf.train.import_meta_graph(FLAGS.model_path + FLAGS.meta_graph_name)


        model_t = tf.train.latest_checkpoint(FLAGS.model_path)
        saver.restore(sess, model_t)

        graphdef = tf.get_default_graph().as_graph_def()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, ['fc2/y_out'])

        return tf.graph_util.remove_training_nodes(frozen_graph)
    else:
        return False

def main():
    if FLAGS.is_train:
        train()
    elif FLAGS.is_test:
        test()
    else:
        graph_def = save_pb_file()

        if graph_def is False:
            raise ValueError("The meta graph do not exist!!!")

        output_file = './graph.pb'
        with tf.gfile.GFile(name = output_file, mode = 'w') as f:
            s = graph_def.SerializeToString()
            f.write(s)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, IndexError) as ve:
        print(ve)