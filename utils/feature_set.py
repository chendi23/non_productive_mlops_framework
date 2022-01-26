# -*- coding: utf-8 -*-
"""
@Time : 2021/12/6 2:10 下午
@Auth : zcd_zhendeshuai
@File : feature_set.py
@IDE  : PyCharm

"""
from collections import defaultdict, namedtuple
import numpy as np
from copy import deepcopy

import pandas as pd

import config.global_variables as gl


class FeatureDictionary():
    def __init__(self, df, numeric_cols=gl.NUMERIC_COLS, ignore_cols=['target']):
        self.df = df
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.field_size = df.shape[1] - len(ignore_cols)

    def generate_mapping_dict(self, output_name=None):
        col_count = 0
        feature_dict = {}
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                feature_dict[col] = col_count
                col_count += 1
            else:
                us = self.df[col].unique()
                feature_dict[col] = dict(zip(us, range(col_count, col_count+len(us))))
                col_count += len(us)
        self.feature_dim = col_count

        # for logging artifacts
        pd.DataFrame(list(feature_dict.keys())).to_csv(gl.PROCESSED_DATA_DIR+'features_target/'+'features.csv')
        pd.DataFrame(['target']).to_csv(gl.PROCESSED_DATA_DIR+'features_target/'+'target.csv')

        return feature_dict, self.feature_dim, self.field_size


class DataParser():
    def __init__(self, df, feature_dict, feature_dim, field_size):
        self.df = df
        self.feature_dict = feature_dict
        self.feature_dim = feature_dim
        self.feature_size = field_size

    def parse(self, phase='preparing'):
        if 'target' in self.df.columns:
            labels = self.df['target'].values.tolist()
        else:
            labels = []
        dfi = deepcopy(self.df)
        dfv = deepcopy(self.df)

        for col in dfi.columns:
            if col in gl.IGNORE_COLS:
                dfi.drop([col], axis=1, inplace=True)
                dfv.drop([col], axis=1, inplace=True)
            elif col in gl.NUMERIC_COLS:
                dfi[col] = self.feature_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feature_dict[col])
                dfi[col] = dfi[col].fillna(int(np.random.randint(low=min(self.feature_dict[col].values()), high=max(self.feature_dict[col].values())+1)), inplace=False)

                dfv[col] = 1
        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()
        if phase == 'preparing':
            dfi.to_csv(gl.PROCESSED_DATA_DIR+'cleaned_data/'+'dfi.csv')
            dfv.to_csv(gl.PROCESSED_DATA_DIR + 'cleaned_data/' + 'dfv.csv')
            self.df['target'].to_csv(gl.PROCESSED_DATA_DIR + 'cleaned_data/' + 'labels.csv')

        return Xi, Xv, labels


