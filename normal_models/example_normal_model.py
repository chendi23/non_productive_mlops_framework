# -*- coding: utf-8 -*-
"""
@Time : 2021/12/13 5:19 下午
@Auth : zcd_zhendeshuai
@File : example_normal_model.py
@IDE  : PyCharm

"""

import re
import collections
import tensorflow as tf
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class ExampleModel():
    # design the params in the form of dictionary, for hyperparams optimization by e.g Optuna
    def __init__(self, params_dict):
        ### normal settings, e.g batch_size, lr, optimizer, nn tunable structures, e.g dropout rate, dense layers nun..
        self.hidden0 = params_dict['hidden0']
        self.optimizer_name = params_dict['optimizer_name']  # string , e.g 'adam'
        self.epochs = params_dict['epochs']
        self.batch_size = params_dict['batch_size']
        self.lr = params_dict['lr']
        self.loss_fn = params_dict['loss_fn']
        ### for import/export model
        self.pretrained_ckpt_dir = params_dict['pretrained_ckpt_dir']  # where to load the pretrained ckpt-model, default value is ""
        self.pb_model_export_dir = params_dict['pb_model_export_dir']  # where to export the pb-model
        self.tags = params_dict['tags'] #e.g ['foo']
        ### for distributed_training
        self.ps_hosts = params_dict['ps_hosts']  # comma-separated string ,e.g 'host0:1234,host1:2345'
        self.worker_hosts = params_dict['worker_hosts']  # comma-separated string ,e.g 'host0:1234,host1:2345'
        self.job_name = params_dict['job_name']  # string, 'ps' or 'worker'
        self.task_index = params_dict['task_index']  # int

    def fp(self, X):
        w0 = tf.get_variable(dtype=tf.float32, shape=[self.hidden0, 1], initializer=tf.initializers.glorot_normal,
                             name='w0')

        b0 = tf.get_variable(dtype=tf.float32, shape=[1], initializer=tf.initializers.glorot_normal,
                             name='b0')

        output = tf.nn.xw_plus_b(X, w0, b0, 'output')
        logits = tf.nn.sigmoid(output, name='logits')
        return logits

    def get_optimizer(self, isDistributed):
        if self.optimizer_name == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = tf.optimizers.SGD(learning_rate=self.lr)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if not isDistributed:
            return optimizer
        else:
            num_workers = len(self.worker_hosts.split(','))
            return tf.train.SyncReplicasOptimizer(opt=optimizer, total_num_replicas=num_workers,
                                                  replicas_to_aggregate=num_workers)
    def get_loss(self, predictions, labels):
        if self.loss_fn == 'mse':
            loss = tf.losses.mean_squared_error(predictions=predictions, labels=labels)

        elif self.loss_fn == 'log':
            loss = tf.losses.log_loss(predictions=predictions, labels=labels)

        elif self.loss_fn == 'crf':
            pass

        return loss

    def model(self, X, y, isDistributed=False):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        preds = self.fp(X)
        loss = self.get_loss(predictions=preds, labels=y)

        optimizer = self.get_optimizer(isDistributed=isDistributed)

        train_op = self.get_train_op(loss=loss, optimizer=optimizer, global_step=global_step,
                                     ckpt_dir=self.pretrained_ckpt_dir)
        if not isDistributed:
            return (train_op, loss, preds, y)
        else:
            return (optimizer, train_op, loss, preds, y)

    # adapted from bert's modelling, to deal with pretrained model/transform learning/fine tuning task
    @staticmethod
    def get_assignment_map_from_checkpoints(tvars, init_checkpoint):

        assignment_map = {}
        initialized_variable_names = {}
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var
        init_variables = tf.train.list_variables(init_checkpoint)

        assignment_map = collections.OrderedDict()
        for x in init_variables:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1
        return assignment_map, initialized_variable_names

    def get_train_op(self, loss, optimizer, global_step, ckpt_dir=""):

        if not ckpt_dir:
            train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = ExampleModel.get_assignment_map_from_checkpoints(tvars, ckpt_dir)
            tf.train.init_from_checkpoint(ckpt_dir, assignment_map)
            train_vars = []
            for var in tvars:
                if var.name in initialized_variable_names:
                    continue
                else:
                    train_vars.append(var)
            grads = tf.gradients(loss, train_vars)
            train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)

        return train_op

    def fit(self, train_x, train_y, valid_x, valid_y, isDistributed=False):
        if not isDistributed:
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir=self.pb_model_export_dir)
            x_data = tf.placeholder(tf.float32, [None, train_x.shape[1]], 'x_data')
            y_data = tf.placeholder(tf.int32, [None, 1], 'y_data')
            train_set = tf.data.Dataset.from_tensor_slices((x_data, y_data))
            val_set = tf.data.Dataset.from_tensor_slices((x_data, y_data))
            iterator = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
            x_batch, y_batch = iterator.get_next()
            (train_op, loss, preds, y) = self.model(x_batch, y_batch)
            train_initializer = tf.data.Iterator.make_initializer(train_set, name='init_op')
            val_initializer = tf.data.Iterator.make_initializer(val_set)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for ep in range(self.epochs):
                    t_preds_list = []
                    t_labels_list = []
                    v_preds_list = []
                    v_labels_list = []
                    # train the model
                    sess.run(train_initializer, feed_dict={x_data: train_x, y_data: train_y})
                    num_train_examples = len(train_y)
                    total_train_loss = 0
                    try:
                        while True:
                            _, t_bt_loss, t_bt_preds, t_bt_labels = sess.run([train_op, loss, preds, y])
                            total_train_loss += t_bt_loss * self.batch_size  # considering reduce_mean
                            t_preds_list.extend(t_bt_preds)
                            t_labels_list.extend(t_bt_labels)
                    except tf.errors.OutOfRangeError:
                        pass

                    train_metric = roc_auc_score(y_true=t_labels_list, y_score=t_preds_list)

                    # validate the model
                    sess.run(val_initializer, feed_dict={x_data: valid_x, y_data: valid_y})
                    total_val_loss = 0
                    num_val_examples = len(valid_y)
                    try:
                        while True:
                            v_bt_loss, v_vt_preds, v_bt_labels = sess.run(loss, preds, y)
                            total_val_loss += v_bt_loss * self.batch_size
                            v_preds_list.extend(v_vt_preds)
                            v_labels_list.extend(v_bt_labels)
                    except tf.errors.OutOfRangeError:
                        pass

                    valid_metric = roc_auc_score(y_true=v_labels_list, y_score=v_preds_list)

                    print('Epoch {}'.format(str(ep)))
                    print("---------------------------")
                    print('Training mean_loss is {} metric is {:.5f}'.format(total_train_loss / num_train_examples, train_metric))
                    print('Validation mean_loss is {} metric is {:.5f}'.format(total_val_loss / num_val_examples, valid_metric))
                x_info, y_info, y = tf.saved_model.utils.build_tensor_info(x_data),\
                                    tf.saved_model.utils.build_tensor_info(preds),\
                                    tf.saved_model.utils.build_tensor_info(y_data)
                signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={'x': x_info, 'y': y},
                                                                                   outputs={'preds': y_info},
                                                                                   method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                builder.add_meta_graph_and_variables(sess=sess, tags=self.tags,
                                                     signature_def_map={'predict': signature})
                builder.save()

            return valid_metric

        else:
            config = tf.ConfigProto(log_device_placement=False)
            ps_list = self.ps_hosts.split(',')
            worker_list = self.worker_hosts.split(',')
            cluster = tf.train.ClusterSpec(
                {'ps': ps_list,
                 'worker': worker_list,
                 }
            )
            if self.job_name == 'ps':
                server = tf.train.Server(cluster, job_name='ps', task_index=self.task_index, config=config)
                server.join()
            else:
                is_chief = (self.task_index == 0)
                server = tf.train.Server(cluster, job_name='worker', task_index=self.task_index, config=config)
                worker_device = "/job:%s/task:%d/gpu:0" % (self.job_name, self.task_index)
                with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device=worker_device)):
                    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=self.pb_model_export_dir)
                    x_data = tf.placeholder(tf.float32, [None, train_x.shape[1]], 'x_data')
                    y_data = tf.placeholder(tf.int32, [None, 1], 'y_data')
                    train_set = tf.data.Dataset.from_tensor_slices((x_data, y_data))
                    val_set = tf.data.Dataset.from_tensor_slices((x_data, y_data))
                    iterator = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
                    x_batch, y_batch = iterator.get_next()
                    (opt, train_op, loss, preds, y) = self.model(x_batch, y_batch, isDistributed)
                    train_initializer = tf.data.Iterator.make_initializer(train_set, name='init_op')
                    val_initializer = tf.data.Iterator.make_initializer(val_set)

                    sync_replicas_hook = opt.make_session_run_hook(is_chief)
                    # stop_hook = tf.train.StopAtStepHook(last_step=10000)
                    # hooks = [sync_replicas_hook, stop_hook]
                    hooks = [sync_replicas_hook]
                    sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                             hooks=hooks,
                                                             is_chief=is_chief,
                                                             config=config,
                                                             stop_grace_period_secs=10)
                    print('Starting training on worker %d'%self.task_index)

                    for ep in range(self.epochs):
                        t_preds_list = []
                        t_labels_list = []
                        v_preds_list = []
                        v_labels_list = []
                        # train the model
                        sess.run(train_initializer, feed_dict={x_data: train_x, y_data: train_y})
                        num_train_examples = len(train_y)
                        total_train_loss = 0
                        try:
                            while True:
                                _, t_bt_loss, t_bt_preds, t_bt_labels = sess.run([train_op, loss, preds, y])
                                total_train_loss += t_bt_loss * self.batch_size  # considering reduce_mean
                                t_preds_list.extend(t_bt_preds)
                                t_labels_list.extend(t_bt_labels)
                        except tf.errors.OutOfRangeError:
                            pass

                        train_metric = roc_auc_score(y_true=t_labels_list, y_score=t_preds_list)

                        # validate the model
                        sess.run(val_initializer, feed_dict={x_data: valid_x, y_data: valid_y})
                        total_val_loss = 0
                        num_val_examples = len(valid_y)
                        try:
                            while True:
                                v_bt_loss, v_vt_preds, v_bt_labels = sess.run(loss, preds, y)
                                total_val_loss += v_bt_loss * self.batch_size
                                v_preds_list.extend(v_vt_preds)
                                v_labels_list.extend(v_bt_labels)
                        except tf.errors.OutOfRangeError:
                            pass

                        valid_metric = roc_auc_score(y_true=v_labels_list, y_score=v_preds_list)

                        print('Epoch {}'.format(str(ep)))
                        print("---------------------------")
                        print('Training mean_loss is {} metric is {:.5f}'.format(total_train_loss / num_train_examples,
                                                                                 train_metric))
                        print('Validation mean_loss is {} metric is {:.5f}'.format(total_val_loss / num_val_examples,
                                                                                   valid_metric))
                    x_info, y_info, y = tf.saved_model.utils.build_tensor_info(x_data), \
                                        tf.saved_model.utils.build_tensor_info(preds), \
                                        tf.saved_model.utils.build_tensor_info(y_data)
                    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={'x': x_info, 'y': y},
                                                                                       outputs={'preds': y_info},
                                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                    builder.add_meta_graph_and_variables(sess=sess, tags=self.tags,
                                                         signature_def_map={'predict': signature})
                    builder.save()

                return valid_metric
