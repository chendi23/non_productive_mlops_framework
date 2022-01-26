# -*- coding: utf-8 -*-
"""
@Time : 2021/12/6 4:15 下午
@Auth : zcd_zhendeshuai
@File : data_loader.py
@IDE  : PyCharm

"""
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_eager_execution()

def parse_example(example, params):
    expected_features = {}
    expected_features['Xi'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['Xv'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['labels'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    parsed_feature_dict = tf.parse_single_example(example, features=expected_features)
    label = parsed_feature_dict['labels']

    label = tf.io.decode_raw(label, out_type=tf.float32)
    label = tf.reshape(label, [])
    Xi = tf.io.decode_raw(parsed_feature_dict['Xi'], out_type=tf.float32)
    Xi = tf.reshape(Xi, [params.field_size])
    Xv = tf.io.decode_raw(parsed_feature_dict['Xv'], out_type=tf.float32)
    Xv = tf.reshape(Xv, [params.field_size])
    parsed_feature_dict['Xi'] = Xi
    parsed_feature_dict['Xv'] = Xv
    parsed_feature_dict.pop('labels')

    return parsed_feature_dict, label

def input_fn(file_dir_list, params):
    files = tf.data.Dataset.list_files(file_dir_list, shuffle=False)
    """
    data_set = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10, block_length=1, sloppy=False)) \
        .map(lambda x: parse_example(x, params), num_parallel_calls=4) \
        .batch(params.batch_size) \
        .prefetch(4000)
    """
    if params.mode == 'train':
        data_set = tf.data.TFRecordDataset(files, buffer_size=params.batch_size * params.batch_size).map(lambda x: parse_example(x, params), num_parallel_calls=4).shuffle(buffer_size=params.batch_size * 10).batch(params.batch_size, drop_remainder=True).prefetch(1)
        iterator = data_set.make_one_shot_iterator()
        features_dict, labels = iterator.get_next()

        #return features_dict, labels
        return data_set

    elif params.mode == 'predict':
        data_set = tf.data.TFRecordDataset(files, buffer_size=params.batch_size * params.batch_size).map(lambda x: parse_example(x, params), num_parallel_calls=4).batch(params.test_count, drop_remainder=True).prefetch(1)
        iterator = data_set.make_one_shot_iterator()
        features_dict, labels = iterator.get_next()
        return features_dict, labels


def get_file_list(input_path):
    file_list = tf.gfile.ListDirectory(input_path)
    file_dir_list = []
    for i in file_list:
        file_dir_list.append(input_path+'/'+i)
    print('number of train_files:', len(file_dir_list))
    return file_dir_list