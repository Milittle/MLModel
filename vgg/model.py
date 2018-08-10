#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 17:18
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : model.py
# @Software: PyCharm

# coding=utf-8
from datetime import datetime
import math
import time
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter_step', 10000, 'define iteration times')
tf.app.flags.DEFINE_integer('batch_size', 32, 'define batch size')
tf.app.flags.DEFINE_integer('num_batches', 100, 'define num batches')
tf.app.flags.DEFINE_integer('classes', 10, 'define classes')
tf.app.flags.DEFINE_float('keep_drop', 0.5, 'define keep dropout')
tf.app.flags.DEFINE_float('lr', 0.001, 'define learning rate')
tf.app.flags.DEFINE_string('model_path', 'model\\','define model path')
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', 'define model name')
tf.app.flags.DEFINE_bool('use_model', False, 'define use_model sign')


# define conv  
def conv(input, name, kh, kw, n_out, dh, dw, p):
    n_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input, kernel, [1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(wx_add_b, name=scope)
        p += [kernel, biases]
        return relu

    # define fc


def fc(input, name, n_out, p):
    n_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="biases")
        relu = tf.nn.relu_layer(input, kernel, biases, name=scope)
        p += [kernel, biases]
        return relu

    # define pool


def max_pool(input, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


# define inference  
def inference(input, keep_prob):
    p = []
    conv1_1 = conv(input, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = max_pool(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = max_pool(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = max_pool(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = max_pool(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)

    conv5_1 = conv(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = max_pool(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    fc6 = fc(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)

    predit = tf.argmax(softmax, 1)
    return predit, softmax, fc8, p;


# define time run  
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration

    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([FLAGS.batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # time  
        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


def main(_):
    run_benchmark()

if __name__ == '__main__':
    tf.app.run()
