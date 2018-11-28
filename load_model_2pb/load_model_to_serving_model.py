#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 14:25
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : load_model_to_serving_model.py
# @Software: PyCharm

import tensorflow as tf


def main(_):
    saver = tf.train.import_meta_graph('../datasets_test/cifar/model/model.ckpt-9991.meta')
    sess = tf.Session()
    last_ckpt = tf.train.latest_checkpoint('../datasets_test/cifar/model/')
    saver.restore(sess, last_ckpt)

    input_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
    output_tensor = sess.graph.get_tensor_by_name('Add:0')

    model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
    model_output = tf.saved_model.utils.build_tensor_info(output_tensor)

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'Placeholder': model_input},
        outputs = {'add': model_output},
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    stored_path_ = './models/cifar10/1'


    try:
        builder = tf.saved_model.builder.SavedModelBuilder(stored_path_)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.TPU],
            signature_def_map = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature_definition
            })
        builder.save(as_text = True)
        print('Model is already store in {}'.format(stored_path_))
    except AssertionError as e:
        print(e)

if __name__ == '__main__':
    tf.app.run()