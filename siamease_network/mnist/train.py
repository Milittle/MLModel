#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/25 22:16
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : train.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np

from siamease_network.mnist import util
from siamease_network.mnist.DataShuffler import *
from siamease_network.mnist.lenet import Lenet

SEED = 10


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    with tf.name_scope('euclidean_distance') as scope:
        #d = tf.square(tf.sub(x, y))
        #d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
        return d


def compute_contrastive_loss(left_feature, right_feature, label, margin, is_target_set_train=True):

    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2

    OR MAYBE THAT

    L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation

    """

    with tf.name_scope("contrastive_loss"):
        label = tf.to_float(label)
        one = tf.constant(1.0)

        d = compute_euclidean_distance(left_feature, right_feature)

        d_sqrt = tf.sqrt(d)

        loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d

        loss = 0.5 * tf.reduce_mean(loss)
        # #first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
        # #first_part = tf.mul(label, tf.square(d))  # (Y-1)*(d^2)
        # between_class = tf.exp(tf.multiply(one-label, tf.square(d)))  # (1-Y)*(d^2)
        # max_part = tf.square(tf.maximum(margin-d, 0))
        #
        # within_class = tf.multiply(label, max_part)  # (Y) * max((margin - d)^2, 0)
        # #second_part = tf.mul(one-label, max_part)  # (Y) * max((margin - d)^2, 0)
        #
        # loss = 0.5 * tf.reduce_mean(within_class + between_class)

        # return loss, tf.reduce_mean(within_class), tf.reduce_mean(between_class)
        return loss

def compute_contrastive_loss_andre(left_feature, right_feature, label, margin, is_target_set_train=True):

    """
    Compute the contrastive loss as in

    https://gitlab.idiap.ch/biometric/xfacereclib.cnn/blob/master/xfacereclib/cnn/scripts/experiment.py#L156


    With Y = [-1 +1] --> [POSITIVE_PAIR NEGATIVE_PAIR]

    L = log( m + exp( Y * d^2)) / N



    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation

    """

    with tf.name_scope("contrastive_loss_andre"):
        label = tf.to_float(label)
        d = compute_euclidean_distance(left_feature, right_feature)

        loss = tf.log(tf.exp(tf.multiply(label, d)))
        loss = tf.reduce_mean(loss)

        # Within class part
        genuine_factor = tf.multiply(label-1, 0.5)
        within_class = tf.reduce_mean(tf.log(tf.exp(tf.multiply(genuine_factor, d))))

        # Between class part
        impostor_factor = tf.multiply(label + 1, 0.5)
        between_class = tf.reduce_mean(tf.log(tf.exp(tf.multiply(impostor_factor, d))))

        # first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
        return loss, between_class, within_class


def main():
    BATCH_SIZE = 128
    BATCH_SIZE_TEST = 128
    #BATCH_SIZE_TEST = 300
    ITERATIONS = 10000
    VALIDATION_TEST = 100
    perc_train = 0.9
    CONTRASTIVE_MARGIN = 1
    USE_GPU = True
    ANDRE_LOSS = False
    LOG_NAME = 'logs'

    train_data, train_labels, test_data, test_labels = util.load_mnist()

    data = np.concatenate((train_data, test_data), axis=0)
    label = np.concatenate((train_labels, test_labels), axis=0)

    data_shuffler = DataShuffler(train_data, train_labels, scale=True)

    # Creating the variables
    lenet_architecture = Lenet(seed=SEED, use_gpu=USE_GPU)
    # Siamease place holders - Training
    train_left_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE*2, 28, 28, 1), name="left")
    train_right_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE*2, 28, 28, 1), name="right")
    labels_data = tf.placeholder(tf.int32, shape=BATCH_SIZE*2)
    # Creating the graphs for training
    lenet_train_left = lenet_architecture.create_lenet(train_left_data)
    lenet_train_right = lenet_architecture.create_lenet(train_right_data)

    # Siamease place holders - Validation
    validation_data = tf.placeholder(tf.float32, shape=(data_shuffler.validation_data.shape[0], 28, 28, 1), name="validation")
    labels_data_validation = tf.placeholder(tf.int32, shape=BATCH_SIZE_TEST)
    # Creating the graphs for validation
    lenet_validation = lenet_architecture.create_lenet(validation_data, train=False)

    if ANDRE_LOSS:
        loss, between_class, within_class = compute_contrastive_loss_andre(lenet_train_left, lenet_train_right, labels_data, CONTRASTIVE_MARGIN)
    else:
        # Regular contrastive loss
        loss = compute_contrastive_loss(lenet_train_left, lenet_train_right, labels_data, CONTRASTIVE_MARGIN)


    distances = compute_euclidean_distance(lenet_train_left, lenet_train_right)


    #regularizer = (tf.nn.l2_loss(lenet_architecture.W_fc1) + tf.nn.l2_loss(lenet_architecture.b_fc1) +
    #                tf.nn.l2_loss(lenet_architecture.W_fc2) + tf.nn.l2_loss(lenet_architecture.b_fc2))
    #loss += 5e-4 * regularizer

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01, # Learning rate
        batch * BATCH_SIZE,
        data_shuffler.train_data.shape[0],
        0.95 # Decay step
    )

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    optimizer = tf.train.AdamOptimizer(0.01, name='Adam').minimize(loss, global_step=batch)

    # pp = PdfPages("groups.pdf")
    # Training
    with tf.Session() as session:

        #Trying to write things on tensor board
        train_writer = tf.summary.FileWriter(LOG_NAME + '/train',
                                              session.graph)

        test_writer = tf.summary.FileWriter(LOG_NAME + '/test',
                                              session.graph)

        tf.summary.scalar('loss', loss)
        # tf.summary.scalar('between_class', between_class)
        # tf.summary.scalar('within_class', within_class)
        tf.summary.scalar('lr', learning_rate)
        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        session.run(init)

        for step in range(ITERATIONS+1):

            batch_left, batch_right, labels = data_shuffler.get_pair(BATCH_SIZE, zero_one_labels=not ANDRE_LOSS)

            feed_dict = {train_left_data: batch_left,
                         train_right_data: batch_right,
                         labels_data: labels}

            _, l, lr, summary = session.run([optimizer, loss, learning_rate, merged], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

            print("Step {0}. Loss Validation = {1}".
                  format(step, l))

            if step % VALIDATION_TEST == 0:

                # Siamese validation
                batch_left, batch_right, labels = data_shuffler.get_pair(n_pair=BATCH_SIZE_TEST,
                                                                         is_target_set_train=False,
                                                                         zero_one_labels=not ANDRE_LOSS)
                feed_dict = {train_left_data: batch_left,
                             train_right_data: batch_right,
                             labels_data: labels}

                d, lv, summary = session.run([distances, loss, merged], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)


                ###################################
                # Siamese as a feature extractor
                ###################################
                batch_train_data, batch_train_labels = data_shuffler.get_batch(
                    data_shuffler.validation_data.shape[0],
                    train_dataset=True)

                features_train = session.run(lenet_validation,
                                             feed_dict={validation_data: batch_train_data[:]})

                batch_validation_data, batch_validation_labels = data_shuffler.get_batch(
                    data_shuffler.validation_data.shape[0],
                    train_dataset=False)

                features_validation = session.run(lenet_validation,
                                                  feed_dict={validation_data: batch_validation_data[:]})

                acc = util.compute_accuracy(features_train, batch_train_labels, features_validation, batch_validation_labels, 10)

                print("Step {0}. Loss Validation = {1}, acc = {2}".
                      format(step, lv, acc))

                # fig = util.plot_embedding_lda(features_validation, batch_validation_labels)
                # pp.savefig(fig)

                #if ANDRE_LOSS:
                #    positive_scores = d[numpy.where(labels == -1)[0]].astype("float")
                #    negative_scores = d[numpy.where(labels == 1)[0]].astype("float")
                #else:
                #    positive_scores = d[numpy.where(labels == 1)[0]].astype("float")
                #    negative_scores = d[numpy.where(labels == 0)[0]].astype("float")

                #threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
                #far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
                #eer = ((far + frr) / 2.) * 100.

        print("End !!")
        train_writer.close()
        test_writer.close()
        # pp.close()


if __name__ == '__main__':
    main()