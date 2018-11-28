#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 20:42
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : model.py
# @Software: PyCharm
# coding=utf-8
from cifar import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()

# define max_iter_step  batch_size
# batch size already define in cifar10.py file, so i do not define in this file.
flags = tf.app.flags
flags.DEFINE_integer('max_iter_step', 10000, 'define iteration times')
flags.DEFINE_string('model_path', 'model\\', 'define model path')
flags.DEFINE_bool('use_model', True, 'define use model sign')

FLAGS = flags.FLAGS


# define variable_with_weight_loss
# 和之前定义的weight有所不同，
# 这里定义附带loss的weight，通过权重惩罚避免部分权重系数过大，导致overfitting
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# 下载数据集 － 调用cifar10函数下载并解压
cifar10.maybe_download_and_extract()
# 注意路径
cifar_dir = '.\\cifar10_data\\cifar-10-batches-bin'

# 采用 data augmentation进行数据处理
# 生成训练数据，训练数据通过cifar10_input的distort变化
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=cifar_dir, batch_size=FLAGS.batch_size)
# 测试数据（eval_data 测试数据）
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=cifar_dir, batch_size=FLAGS.batch_size)

# 创建输入数据，采用 placeholder
x_input = tf.placeholder(tf.float32, [FLAGS.batch_size, 24, 24, 3])
y_input = tf.placeholder(tf.int32, [FLAGS.batch_size])

# 创建第一个卷积层 input:3(channel) kernel:64 size:5*5
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.conv2d(x_input, weight1, [1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 创建第二个卷积层 input:64 kernel:64 size:5*5
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
bias2 = tf.Variable(tf.constant(0, 1, shape=[64]))
conv2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建第三个层－全连接层  output:384
reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 创建第四个层－全连接层  output:192
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层  output:10
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
results = tf.add(tf.matmul(local4, weight5), bias5)


# 定义loss
def loss(results, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=results, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 计算loss
loss = loss(results, y_input)
global_step = tf.Variable(0, trainable=False)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)  # Adam
top_k_op = tf.nn.in_top_k(results, y_input, 1)  # top1 准确率

sess = tf.Session()  # 创建session
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=3)
with sess.as_default():
    tf.train.start_queue_runners()  # 启动多线程加速

if FLAGS.use_model and os.path.exists(FLAGS.model_path + "checkpoint"):
    print("model")
    model_t = tf.train.latest_checkpoint(FLAGS.model_path)
    saver.restore(sess, model_t)


# 开始训练
for step in range(FLAGS.max_iter_step):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={x_input: image_batch, y_input: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / duration
        sec_per_batch = float(duration)

        saver.save(sess, FLAGS.model_path + "model.ckpt", global_step=global_step)

        format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

# 评测模型在测试集上的准确度
num_examples = 10000
import math

num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
true_count = 0
total_sample_count = num_iter * FLAGS.batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={x_input: image_batch, y_input: label_batch})
    true_count += np.sum(predictions)
    step += 1

# 打印结果
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)