# -*- coding: utf-8 -*-
"""
@Time : 2021/12/6 10:23 上午
@Auth : zcd_zhendeshuai
@File : data_preprocessing.py
@IDE  : PyCharm

"""
import datetime
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import dvc.api
import config.global_variables as gl
from utils.logger_config import get_logger
from utils.feature_set import FeatureDictionary, DataParser
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger(gl.LOG_DATA_PROCESSING_DIR)


def get_raw_data(repo=None):
    user_url = dvc.api.get_url(
        path=gl.DATA_DIR + 'raw_data/' + 'users.csv',
        repo=repo,
        rev=None)
    items_url = dvc.api.get_url(
        path=gl.DATA_DIR + 'raw_data/' + 'items.csv',
        repo=repo,
        rev=None)
    ctr_url = dvc.api.get_url(
        path=gl.DATA_DIR + 'raw_data/' + 'ctr.csv',
        repo=repo,
        rev=None)
    ratings_url = dvc.api.get_url(
        path=gl.DATA_DIR + 'raw_data/' + 'ratings.csv',
        repo=repo,
        rev=None)

    return user_url, items_url, ctr_url, ratings_url


def read_csv_from_remote(user_url, items_url, ctr_url, ratings_url):
    users, items, ratings, ctrs = pd.read_csv(user_url, sep=';'), pd.read_csv(items_url, sep=';'), \
                                  pd.read_csv(ratings_url, sep=';'), pd.read_csv(ctr_url, sep=';')

    return users, items, ctrs, ratings


def clean_data(users, items, ctrs, mode='non_seq', ratings=None):
    combine_item_clicking = pd.merge(ctrs, items[['item_id']], on='item_id', how='inner')
    user_list = list(combine_item_clicking['user_id'].unique())  # 评分列表中的用户取唯一值，即哪些用户给了评分
    item_list = list(combine_item_clicking['item_id'].unique())  # 评分列表中的商品取唯一值，即哪些商品被评分了
    base_info = {'users': users, 'items': items, 'ratings': ratings, 'ctrs': ctrs,
                 "combine_item_clicking": combine_item_clicking, 'user_list': user_list,
                 'item_list': item_list}
    filter_user = pd.merge(left=base_info['combine_item_clicking'], right=base_info['users'],
                           on=['user_id', 'user_type'],
                           how="inner")  # 用户最少行数相对最少，先用用户表拼接信息
    logger.debug(
        '------------------------filter_user:%s------------------------------------------' % len(filter_user))
    filter_item = pd.merge(left=filter_user, right=base_info['items'], on=['item_id'],
                           how='inner')  # 物品数相对适中，拼接物品信息，过滤掉没有交互的物品
    logger.debug(
        '------------------------filter_item:%s------------------------------------------' % len(filter_item))
    filter_click = pd.merge(left=filter_item, right=base_info['ctrs'],
                            on=['user_id', 'user_type', 'item_id', 'click'],
                            how='inner')  # 交互信息最多，拼接交互信息放在最后
    logger.debug(
        '------------------------filter_click:%s------------------------------------------' % len(filter_click))
    filter_click['id'] = filter_click['user_id'] + "|" + (filter_click['user_type']).map(str) + "|" + filter_click[
        'item_id']
    filter_click['date_time'] = pd.to_datetime(filter_click['date_time'])
    filter_click['date_y'] = filter_click['date_time'].dt.year / 100
    filter_click['date_m'] = filter_click['date_time'].dt.month
    base_time = datetime.datetime.strptime('1949-10-01', '%Y-%m-%d')
    filter_click['date_diff'] = filter_click['date_time'].apply(lambda x: x - base_time).dt.days
    filter_click['date_diff']=filter_click['date_diff'].apply(lambda x: (x - np.mean(x)) / (x - np.std(x)))

    filter_click.drop(columns=gl.NON_SEQ_DELETE_COLS, inplace=True)
    filter_click.dropna(inplace=True)
        # filter_click.fillna('-1',inplace=True)
        # filter_click['missing_feat'] = np.sum((filter_click[gl.NONE_SEQ_MODEL_COLS] == -1).values, axis=1)
    # elif mode == 'seq':
    #     filter_click.drop(columns=gl.SEQ_DELETE_COLS, inplace=True)
    #     filter_click.fillna('-1', inplace=True)
    #     filter_click = filter_click.groupby(['user_id']).apply(lambda x: x.sort_values(by='item_id'))
    #     filter_click['missing_feat'] = np.sum((filter_click[gl.NONE_SEQ_MODEL_COLS] == -1).values, axis=1)
    filter_click.rename(columns={"click": "target"}, inplace=True)  # 将click作为排序target

    logger.debug(
        '------------------------特征列列名%s------------------------------------------' % filter_click.columns.tolist())
    logger.debug(
        '------------------------特征列样例%s------------------------------------------' % filter_click.head(6))
    logger.debug(filter_click.shape[0])
    return filter_click


def generate_input_list(filter_click):
    fd = FeatureDictionary(filter_click)
    feature_dict, feature_dim, field_size = fd.generate_mapping_dict()
    dp = DataParser(df=filter_click, feature_dict=feature_dict, feature_dim=feature_dim, field_size=field_size)
    Xi, Xv, labels = dp.parse()

    return Xi, Xv, labels


def build_tfrecords(lists_dict, rows_count, output_dir):
    logger.debug('---begin building tfrecords---')
    assert not (lists_dict is None)

    def get_Float_ListFeature(value):
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
            value = value.astype(np.float32).tostring()
            value = [value]
            float_list = tf.train.BytesList(value=value)
            return tf.train.Feature(bytes_list=float_list)
        else:
            value = value.astype(np.float32).tostring()
            value = [value]
            float_list = tf.train.BytesList(value=value)
            return tf.train.Feature(float_list)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.tfrecords'
    with tf.io.TFRecordWriter(path=os.path.join(output_dir, filename)) as wr:
        for i in range(int(rows_count)):
            single_row_dict = {}
            for k, v in lists_dict.items():
                single_row_dict[k] = get_Float_ListFeature(v[i])
                # print(single_row_dict)
            features = tf.train.Features(feature=single_row_dict)
            example = tf.train.Example(features=features)
            # print(exanple)
            wr.write(record=example.SerializeToString())

        wr.close()
    logger.debug('---finish building tfrecords---')

    return

def argument_parser(params_dict):
    parser = ArgumentParser()
    for k, v in params_dict.items():
        parser.add_argument('--%s'%k, default=v)
    return parser.parse_args()



def main():
    user_url, items_url, ctr_url, ratings_url = get_raw_data()
    users, items, ctrs, ratings = read_csv_from_remote(user_url, items_url, ctr_url, ratings_url)
    filter_click = clean_data(users, items, ctrs, ratings)
    _get = lambda x, y: [x[i] for i in y]

    Xi, Xv, labels = generate_input_list(filter_click)
    CV = StratifiedKFold(n_splits=gl.K_FOLDS, shuffle=True).split(Xi, labels)
    for i, (train_idx, valid_idx) in enumerate(CV):
        Xi_train, Xv_train, labels_train = _get(Xi, train_idx), _get(Xv, train_idx), _get(labels, train_idx)
        Xi_valid, Xv_valid, labels_valid = _get(Xi, valid_idx), _get(Xv, valid_idx), _get(labels, valid_idx)

        build_tfrecords(lists_dict={'Xi': Xi_train, 'Xv': Xv_train, 'labels': labels_train},
                        output_dir=gl.TFRECORDS_BASE_DIR + 'train/%d'%i, rows_count=len(labels_train))
        build_tfrecords(lists_dict={'Xi': Xi_valid, 'Xv': Xv_valid, 'labels': labels_valid},
                        output_dir=gl.TFRECORDS_BASE_DIR + 'valid/%d'%i, rows_count=len(labels_valid))


if __name__ == '__main__':
    main()
