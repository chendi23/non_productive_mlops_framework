# -*- coding: utf-8 -*-
"""
@Time : 2021/12/13 13:21
@Auth : zcd_zhendeshuai
@File : syn_distribute_bert_bilstm.py
@IDE  : PyCharm

"""

from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import sys
import os
import time
from sklearn.metrics import recall_score, precision_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import config.global_var as gl
from data_utils.preprocessing import load_sentences, tag_mapping, prepare_dataset
from bert.modeling import BertConfig, BertModel, get_assignment_map_from_checkpoint
from utils.logger_config import get_logger

logger = get_logger(gl.LOG_DIR + '/distributed_train.log')

FLAGS = None

log_dir = './logdir'
REPLICAS_TO_AGGREGATE = 2


def get_splitted_data():
    sentences = load_sentences(gl.NER_TRANSFORMED_DATA_DIR + '/transformed.txt', gl.LOWER, gl.ZEROS)
    id_to_tag, tag_to_id = tag_mapping(sentences)
    ids_list, mask_list, segment_ids_list, label_list = prepare_dataset(sentences=sentences,
                                                                        max_seq_length=gl.MAX_SEQ_LENGTH,
                                                                        tag_to_id=tag_to_id, lower=gl.LOWER,
                                                                        train=True)
    total_num = int(len(mask_list))
    train_num = int(total_num * (gl.TRAIN_VALID_RATIO / (1 + gl.TRAIN_VALID_RATIO)))
    valid_num = total_num - train_num

    train_batches = int(train_num / gl.batch_size)
    valid_batches = int(valid_num / gl.batch_size)

    train_index = np.random.choice(total_num, train_num, replace=False)
    valid_index = np.random.choice(total_num, valid_num, replace=False)

    _get = lambda x, y: [x[i] for i in y]

    t_ids, t_labels, t_masks, t_segment_ids = _get(ids_list, train_index), _get(label_list, train_index), \
                                              _get(mask_list, train_index), _get(segment_ids_list, train_index)
    v_ids, v_labels, v_masks, v_segment_ids = _get(ids_list, valid_index), _get(label_list, valid_index), \
                                              _get(mask_list, valid_index), _get(segment_ids_list, valid_index)

    train_dict = {'ids': t_ids, 'labels': t_labels, 'masks': t_masks, 'segment_ids': t_segment_ids}
    valid_dict = {'ids': v_ids, 'labels': v_labels, 'masks': v_masks, 'segment_ids': v_segment_ids}
    return train_dict, valid_dict, train_batches, valid_batches


def get_batch_data(bert_input_dict, step=0):
    ids = np.zeros(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH])
    labels = np.zeros(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH])
    masks = np.zeros(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH])
    segment_ids = np.zeros(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH])
    for i in range(gl.batch_size):
        ids[i, :] = bert_input_dict['ids'][step * gl.batch_size + i]
        labels[i, :] = bert_input_dict['labels'][step * gl.batch_size + i]
        masks[i, :] = bert_input_dict['masks'][step * gl.batch_size + i]
        segment_ids[i, :] = bert_input_dict['segment_ids'][step * gl.batch_size + i]

    return ids, labels, masks, segment_ids


def train(train_dict, valid_dict, train_max_step, valid_max_step):
    # Configure
    config = tf.ConfigProto(log_device_placement=False)

    # Server Setup
    cluster = tf.train.ClusterSpec({
        'ps': ['192.168.3.154:55555'],
        'worker': ['192.168.3.135:44444', '192.168.3.203:22222']
    })  # allows this node know about all other nodes
    if FLAGS.job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=FLAGS.task_index,
                                 config=config)
        server.join()
    else:  # it must be a worker server
        is_chief = (FLAGS.task_index == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=FLAGS.task_index,
                                 config=config)

        # Graph
        worker_device = "/job:%s/task:%d/cpu:0" % (FLAGS.job_name, FLAGS.task_index)
        with tf.device(tf.train.replica_device_setter(ps_tasks=1,
                                                      worker_device=worker_device)):

            ids = tf.placeholder(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH], dtype=tf.int32, name='ids')
            labels = tf.placeholder(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH], dtype=tf.int32, name='labels')
            masks = tf.placeholder(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH], dtype=tf.int32, name='masks')
            segment_ids = tf.placeholder(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH], dtype=tf.int32, name='segment_ids')
            # is_training = tf.placeholder(shape=[], dtype=tf.bool, name='is_training')
            used = tf.sign(tf.abs(ids))
            length = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(length, tf.int32)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            bert_config = BertConfig.from_json_file(gl.ROOT_PATH + '/' + "chinese_L-12_H-768_A-12/bert_config.json")
            bert_base = BertModel(config=bert_config, input_ids=ids, input_mask=masks, token_type_ids=segment_ids,
                                  is_training=True)
            bert_out = bert_base.get_sequence_output()

            with tf.variable_scope('biLSTM'):
                fw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=100, forget_bias=1.0, state_is_tuple=True)
                bw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=100, forget_bias=1.0, state_is_tuple=True)
                (fw_output, bw_output), status = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, bert_out,
                                                                                 dtype=tf.float32,
                                                                                 )
                bilstm_out = tf.concat([fw_output, bw_output], axis=2)
                bilstm_dropout = tf.nn.dropout(bilstm_out, 0.5, name='bilstm_dropout')

            logger.debug('biLSTM_fp_finished')

            with tf.variable_scope('biLSTM_projection'):
                projection_out = tf.layers.dense(inputs=bilstm_dropout, units=gl.ann_num_tags,
                                                 kernel_initializer=tf.keras.initializers.glorot_normal,
                                                 bias_initializer=tf.keras.initializers.glorot_normal,
                                                 )

            logger.debug('biLSTM_projection_fp_finished')

            with tf.variable_scope('crf_layer'):
                trans_matrix = tf.get_variable(name='transition_matrix', shape=[gl.ann_num_tags, gl.ann_num_tags],
                                               initializer=tf.initializers.glorot_normal)

                log_likelihood, trans_matrix = tf.contrib.crf.crf_log_likelihood(inputs=projection_out,
                                                                                 tag_indices=labels,
                                                                                 sequence_lengths=lengths,
                                                                                 transition_params=trans_matrix)
                predictions, _ = tf.contrib.crf.crf_decode(projection_out, trans_matrix, lengths)

            loss = tf.reduce_sum(-log_likelihood)

            # optimizer = tf.train.AdamOptimizer()
            # optimizer = tf.train.SyncReplicasOptimizer(opt=adam_optimizer, total_num_replicas=num_worker,replicas_to_aggregate=num_worker)
            # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            # vars_to_load = [var[0] for var in tf.train.list_variables(gl.INITIAL_CKPT)]
            # assign_map = {var.op.name: var for var in tf.global_variables() if var.op.name in vars_to_load}
            tvars = tf.trainable_variables()
            (assignment_map, initialized_vars) = get_assignment_map_from_checkpoint(tvars, gl.INITIAL_CKPT)

            train_vars = []
            for var in tvars:
                if var.name in initialized_vars:
                    continue
                else:
                    train_vars.append(var)
            grads = tf.gradients(loss, train_vars)
            # train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)
            # create an optimizer then wrap it with SynceReplicasOptimizer
            optimizer = tf.train.AdamOptimizer(0.001)
            optimizer1 = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=REPLICAS_TO_AGGREGATE,
                                                        total_num_replicas=2)

            opt = optimizer1.apply_gradients(zip(grads, train_vars), global_step=global_step)  # averages gradients
            # opt = optimizer1.minimize(REPLICAS_TO_AGGREGATE*loss,
            #                           global_step=global_step) # hackily sums gradients

        # Session
        sync_replicas_hook = optimizer1.make_session_run_hook(is_chief)
        stop_hook = tf.train.StopAtStepHook(last_step=100000)
        hooks = [sync_replicas_hook, stop_hook]

        # Monitored Training Session
        sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                 is_chief=is_chief,
                                                 config=config,
                                                 hooks=hooks,
                                                 stop_grace_period_secs=10)

        print('Starting training on worker %d' % FLAGS.task_index)
        # while not sess.should_stop():
        #     _, r, gs = sess.run([opt, c, global_step])
        #     print(r, 'step: ', gs, 'worker: ', FLAGS.task_index)
        #     if is_chief: time.sleep(1)
        #     time.sleep(1)
        # print('Done', FLAGS.task_index)
        #
        # time.sleep(10)  # grace period to wait before closing session
        for ep in range(gl.EPOCH):
            for step in range(train_max_step):
                t_ids, t_labels, t_masks, t_segment_ids = get_batch_data(train_dict, step)
                # t_ids, t_labels, t_masks, t_segment_ids = np.int32(t_ids), np.int32(t_labels), np.int32(t_masks), np.int32(t_segment_ids)
                t_ids, t_labels, t_masks, t_segment_ids = t_ids.tolist(), t_labels.tolist(), t_masks.tolist(), t_segment_ids.tolist()
                train_feed_dict = {'ids:0': t_ids, 'labels:0': t_labels, 'masks:0': t_masks,
                                   'segment_ids:0': t_segment_ids}
                _, train_loss, train_preds, lens, gl_step = sess.run([opt, loss, predictions, lengths, global_step],
                                                                     feed_dict=train_feed_dict)
                print(train_preds)
                t_preds_1d, t_labels_1d = [], []
                for i in range(gl.batch_size):
                    t_preds_1d.extend(train_preds[i][:lens[i]])
                    t_labels_1d.extend(t_labels[i][:lens[i]])
                logger.debug('batch preds non zero {}'.format(len(np.nonzero(t_preds_1d)[0])))
                # print('batch labels: ', t_labels_1d)
                acc = sum(np.equal(t_labels_1d, t_preds_1d)) / np.sum(lens)
                t_batch_recall = recall_score(y_pred=t_preds_1d, y_true=t_labels_1d, average='macro')
                t_batch_f1 = f1_score(y_pred=t_preds_1d, y_true=t_labels_1d, average='macro')
                logger.debug(
                    'ep={}\t Worker={}\t step={}\t global_step={}\t batch train loss={:.5f}\t batch_acc={:.5f}\t recall={:.5f}\t f1={:.5f}'.format(
                        ep, FLAGS.task_index, step, gl_step, train_loss, acc, t_batch_recall, t_batch_f1))

        sess.close()
        print('Session from worker %d closed cleanly' % FLAGS.task_index)


def main():
    train_dict, valid_dict, train_batches, valid_batches = get_splitted_data()
    train(train_dict, valid_dict, train_batches, valid_batches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.task_index)
    main()
