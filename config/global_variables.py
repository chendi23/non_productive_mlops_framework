# -*- coding: utf-8 -*-
"""
@Time : 2021/12/6 10:38 上午
@Auth : zcd_zhendeshuai
@File : global_variables.py
@IDE  : PyCharm

"""
import os

# dvc
DVC_LOCAL_REPO = '/Users/chendi/PycharmProjects/dvc_remote'
DVC_REMOTE_REPO = 'gdrive://1fh3eHJvqpiX5WViwSrGaomyv6lLshZwV'

#mlflow
MLFLOW_CONNECTED_DATABASE_NAME ='db_mlflow'
MLFLOW_CONNECTED_DATABASE_USER = 'mlflow'
MLFLOW_CONNECTED_DATABASE_PASSWORD = 'zcd19961023'
MLFLOW_CONNECTED_DATABASE_PORT = 3306
MLFLOW_GUI_PORT = 5000


#basic dirs
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = PROJECT_ROOT_DIR + '/data/'

#log dir
LOG_DIR = DATA_DIR + '/log/'
LOG_DATA_PROCESSING_DIR = LOG_DIR + 'data_processing.log'
LOG_TRAIN_DIR = LOG_DIR + 'train.log'

#processed data dir
PROCESSED_DATA_DIR  = DATA_DIR + 'processed_data/'
TFRECORDS_BASE_DIR  = DATA_DIR + 'tfrecords_dataset/'


#checkpoint&pb dir
CKPT_BASE_DIR = DATA_DIR + '/ckpt/'
MODEL_PB_DIR = DATA_DIR + '/pb_models/'


#fearure cols
SEQ_DELETE_COLS = ['user_name', 'org_id', 'position_id', 'grade_id', 'grade_name',
               'source', 'heat', 'type', 'i_keywords_label','u_entities_label',
               'position_id', 'user_type', 'content', 'category_id', 'title', 'id', 'date_time']
NON_SEQ_DELETE_COLS = ['user_name', 'user_id','item_id','org_id', 'seat_id','position_id', 'grade_id', 'grade_name',
               'source', 'heat', 'type', 'i_keywords_label','u_entities_label',
               'position_id', 'user_type', 'content', 'category_id', 'title', 'id', 'date_time']

NONE_SEQ_MODEL_COLS = ['sex', 'org_name','position_name','u_class_label','position_name','category_name','date_y',
                       'date_m', 'date_diff']

IGNORE_COLS = ['target']
NUMERIC_COLS = ['age','date_diff','date_y','date_m']

# cross validation
K_FOLDS = 3

# hyper parameters/nn structure search grids
EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LEARNING_RATE_MIN = 0.001
LEARNING_RATE_MAX = 0.01


EMB_DIM_MIN = 4
EMB_DIM_MAX = 16

DROPOUT_FM_MIN = 0.3
DROPOUT_FM_MAX = 0.8
DEEP_LAYERS_MIN = 16
DEEP_LAYERS_MAX = 32

# distributed training
job_name = 'test_distributed_training'
task_index = 0
ps_hosts = "127.0.0.1:22222"
worker_hosts = "192.168.3.135:22222, 192.168.3.235:22222"
