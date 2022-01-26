# -*- coding: utf-8 -*-
"""
@Time : 2021/12/6 4:10 下午
@Auth : zcd_zhendeshuai
@File : automatic_ml.py
@IDE  : PyCharm

"""
import numpy as np
import uuid
import os
import json
import shutil

import tensorflow as tf
import optuna
import dvc.api
import mlflow.tensorflow
import config.global_variables as gl
from estimator_models import afm_estimator,deepfm_estimator,mmoe_estimator,esmm_estimator
from utils.data_preprocessing import argument_parser
from utils import model_ops, data_loader

if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_eager_execution()

def build_params_dict(trial):
    params_dict = dict()
    params_dict['batch_size'] = gl.BATCH_SIZE
    params_dict['epoch'] = gl.EPOCH
    params_dict['field_size'] = 13
    params_dict['feature_dim'] = 2895
    params_dict['optimizer'] = trial.suggest_categorical(name='optimizer', choices=['adam', 'sgd'])
    if params_dict['optimizer'] == 'adam':
        params_dict['epsilon'] = trial.suggest_float('epsilon', 1e-5, 1e-1)
    else:
        params_dict['momentum'] = trial.suggest_float('momentun', 1e-5, 1e-1)
    params_dict['emb_dim'] = trial.suggest_int('emb_dim', gl.EMB_DIM_MIN, gl.EMB_DIM_MAX)
    # params_dict['use_fm'] = True
    # params_dict['use_deep'] = True
    params_dict['lr'] = trial.suggest_float('learning_rate', gl.LEARNING_RATE_MIN, gl.LEARNING_RATE_MAX)
    # params_dict['momentum'] = trial.suggest_float('momentum', 1e-5, 1e-1)
    # params_dict['epsilon'] = trial.suggest_float('epsilon', 1e-5, 1e-1)
    # params_dict['deep_layers'] = [trial.suggest_int('deep_layers_0', gl.DEEP_LAYERS_MIN, gl.DEEP_LAYERS_MAX),
    #                               trial.suggest_int('deep_layers_1', gl.DEEP_LAYERS_MIN, gl.DEEP_LAYERS_MAX)]
    # params_dict['dropout_fm'] = [trial.suggest_float("dropout_fm_0", gl.DROPOUT_FM_MIN, gl.DROPOUT_FM_MAX),
    #                              trial.suggest_float("dropout_fm_1", gl.DROPOUT_FM_MIN, gl.DROPOUT_FM_MAX)]
    # params_dict['dropout_deep'] = [0.5, 0.5, 0.5]
    params_dict['is_GPU'] = 0
    params_dict['log_step_count_steps'] = 100000
    params_dict['save_checkpoints_steps'] = 100000
    params_dict['keep_checkpoint_steps'] = 100000
    params_dict['keep_checkpoint_max'] = 0
    params_dict['save_summary_steps'] = 100000
    return params_dict


def objective(trial, model_name='afm', is_training=True):
    params_dict = build_params_dict(trial)

    with mlflow.start_run(nested=True):
        mlflow.log_param("data_url", dvc.api.get_url(gl.PROCESSED_DATA_DIR + 'cleaned_data/'))
        mlflow.log_param("data_version", 'v0')
        mlflow.log_artifact(gl.PROCESSED_DATA_DIR + 'features_target/' + 'features.csv')
        mlflow.log_artifact(gl.PROCESSED_DATA_DIR + 'features_target/' + 'target.csv')

        mlflow.log_params(params=params_dict)
        params_dict['model'] = model_name
        if is_training:
            params_dict['mode'] = 'train'
        res = np.zeros([gl.K_FOLDS])
        for i in range(int(gl.K_FOLDS)):
            params_dict['model_dir'] = gl.CKPT_BASE_DIR + '%d' % i
            params_dict['train_path'] = gl.TFRECORDS_BASE_DIR + 'train/%d' % i
            params_dict['predict_path'] = gl.TFRECORDS_BASE_DIR + 'valid/%d' % i
            params_dict['model_pb'] = gl.MODEL_PB_DIR
            params = argument_parser(params_dict)
            model = afm_estimator.model_estimator(params)
            train_file = data_loader.get_file_list(params.train_path)
            valid_file = data_loader.get_file_list(params.predict_path)
            _, results = model_ops.model_fit(model=model, params=params, train_file=train_file, predict_file=valid_file)
            res[i] = results['ctr_auc_metric']
            shutil.rmtree(params.model_dir)

        res_fold_mean = float(np.mean(res))
        mlflow.log_metric('auc', res_fold_mean)

    return res_fold_mean


def set_tfconfig_environ(choose_ps_as_evaluate=False):
    parse_argument()

    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        gl.worker_num = worker_num
        if task_type == "ps":
            # 把第一个ps转为evaluator
            if len(cluster["ps"]) >= 2 and choose_ps_as_evaluate:
                if task_index == 0:
                    tf_config["task"] = {"index": 0, "type": "evaluator"}
                    gl.job_name = "evaluator"
                    gl.task_index = 0
                else:
                    tf_config["task"] = {"index": task_index - 1, "type": task_type}
                    gl.job_name = "ps"
                    gl.task_index = task_index - 1
            else:
                tf_config["task"] = {"index": task_index, "type": task_type}
                gl.job_name = "ps"
                gl.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            gl.job_name = "worker"
            gl.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        if len(cluster["ps"]) >= 2 and choose_ps_as_evaluate:
            del cluster["ps"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

    # if "INPUT_FILE_LIST" in os.environ:
    #     INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    #     if INPUT_PATH:
    #         print("input path:", INPUT_PATH)
    #         gl.train_data = INPUT_PATH.get(gl.train_data)
    #         gl.eval_data = INPUT_PATH.get(gl.eval_data)
    #     else:  # for ps
    #         print("load input path failed.")
    #         gl.train_data = None
    #         gl.eval_data = None


def parse_argument():
    if gl.job_name is None or gl.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")

    if gl.task_index is None or gl.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % gl.job_name)
    print("task index = %d" % gl.task_index)

    os.environ["TF_ROLE"] = gl.job_name
    os.environ["TF_INDEX"] = str(gl.task_index)

    # Construct the cluster and start the server
    ps_spec = gl.ps_hosts.split(",")
    worker_spec = gl.worker_hosts.split(",")

    cluster = {"worker": worker_spec, "ps": ps_spec}

    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)






def main():
    mlflow.set_tracking_uri(f'mysql+pymysql://{gl.MLFLOW_CONNECTED_DATABASE_USER}:{gl.MLFLOW_CONNECTED_DATABASE_PASSWORD}@localhost:{gl.MLFLOW_CONNECTED_DATABASE_PORT}/{gl.MLFLOW_CONNECTED_DATABASE_NAME}')
    mlflow.set_experiment('deepfm')
    #os.environ.pop('TF_CONFIG')
    #parse_argument()
    #set_tfconfig_environ()
    tf.logging.set_verbosity('FATAL')
    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, model_name='afm', is_training=True)
    study.optimize(func=func, n_trials=10)



if __name__ == '__main__':
    main()
