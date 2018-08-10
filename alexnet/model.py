#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 14:10
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


# define print shape  
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())



def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        wx = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv1 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv1)

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        wx = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv2 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        wx = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv3 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        wx = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv4 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        wx = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        wx_add_b = tf.nn.bias_add(wx, biases)
        conv5 = tf.nn.relu(wx_add_b, name=scope)
        parameters += [kernel, biases]
    print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)
    return pool5, parameters


# define time run  
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
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
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # time
        time_tensorflow_run(sess, pool5, "Forward")
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")

def main(_):
    run_benchmark()

if __name__ == '__main__':
    tf.app.run()
