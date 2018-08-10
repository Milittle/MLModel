#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/2 10:25
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : simple.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops

ops.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step_size', 530, 'define iteration time')
tf.app.flags.DEFINE_integer('batch_size', 256, 'define batch size')
tf.app.flags.DEFINE_boolean('trainable', True, 'define trainable sign')
tf.app.flags.DEFINE_float('lr', 1e-3, 'define learning rate')
tf.app.flags.DEFINE_integer('input_size', 28, 'define input size ')
tf.app.flags.DEFINE_integer('time_step_size', 28, 'define time step size')
tf.app.flags.DEFINE_integer('hidden_layer_size', 256, 'define hidden layer size')
tf.app.flags.DEFINE_integer('layer_num', 2, 'define layer depth')
tf.app.flags.DEFINE_integer('class_num', 10, 'define class number')
tf.app.flags.DEFINE_boolean('save_model', True, 'define save model')
tf.app.flags.DEFINE_string('model_path', 'model/', 'define model path')
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', 'define model name')
tf.app.flags.DEFINE_bool('use_model', False, 'define whether use model or not')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)

_x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, FLAGS.class_num])
keep_prob = tf.placeholder(tf.float32)

x = tf.reshape(_x, [-1, 28, 28])

def lstm_cell(cell_type, num_nodes, keep_prob):
    if cell_type == "lstm":
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(num_nodes)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


mlstm_cell = rnn.MultiRNNCell([lstm_cell("lstm", FLAGS.hidden_layer_size, keep_prob) for _ in range(FLAGS.layer_num)],
                              state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size=FLAGS.batch_size, dtype=tf.float32)

outputs = list()

state = init_state

with tf.variable_scope('RNN'):
    for timestep in range(FLAGS.time_step_size):
        (cell_output, state) = mlstm_cell(x[:, timestep, :], state)
        outputs.append(cell_output)

h_state = outputs[-1]

W = tf.Variable(tf.truncated_normal([FLAGS.hidden_layer_size, FLAGS.class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[FLAGS.class_num]), dtype=tf.float32)

y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))

global_step = tf.Variable(tf.constant(0), trainable=False)

train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(cross_entropy, global_step=global_step)

correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pre, 'float'))

saver = tf.train.Saver(max_to_keep=5)


def train():
    epoch = 1
    sess.run(tf.global_variables_initializer())
    if FLAGS.use_model:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))
    for i in range(FLAGS.max_step_size):
        batch_data = mnist.train.next_batch(FLAGS.batch_size)
        if (i + 1) % 215 == 0:
            epoch += 1
            if FLAGS.save_model:
                saver.save(sess, FLAGS.model_path + FLAGS.model_name, global_step=global_step)
            train_accuracy = sess.run(accuracy, feed_dict={_x: batch_data[0], y: batch_data[-1], keep_prob: 1})
            print('Iteration {iter}, step {step}, training accuracy {acc}'.format(iter=mnist.train.epochs_completed,
                                                                                  step=i, acc=train_accuracy))
        sess.run(train_op, feed_dict={_x: batch_data[0], y: batch_data[-1], keep_prob: 0.5})
        loss = sess.run(cross_entropy, feed_dict={_x: batch_data[0], y: batch_data[-1], keep_prob: 0.5})
        print('epoch:{epoch} / {epoch_sum}'.format(epoch = epoch, epoch_sum= FLAGS.max_step_size // 215))
        print('step:{step} loss: {loss}'.format(step = i, loss = loss))


def test():
    sess.run(tf.global_variables_initializer())
    if FLAGS.use_model:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))
    for i in range(39):
        test_batch = mnist.train.next_batch(256)
        test_accuracy = sess.run(accuracy, feed_dict={_x: test_batch[0], y: test_batch[-1], keep_prob: 1})
        print('step:{step} test acc {acc}'.format(step=i, acc=test_accuracy))
    sess.close()


def main(_):
    if FLAGS.trainable:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
