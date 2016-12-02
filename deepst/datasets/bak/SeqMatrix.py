# -*- coding: utf-8 -*-
"""Sequential matrix data
"""
from __future__ import print_function
# import os
import cPickle as pickle
import numpy as np

from deepst_flow.preprocessing import MinMaxNormalization
from deepst_flow.preprocessing import remove_incomplete_days
# from deepst_flow.config import Config
from deepst_flow.datasets.STMatrix import STMatrix
# from deepst_flow.preprocessing import timestamp2vec
from deepst_flow.datasets import load_stdata
# np.random.seed(1337)  # for reproducibility

# parameters
# DATAPATH = Config().DATAPATH

def load_data(fname=None, T=48, nb_flow=2, len_seq=None, len_test=None, preprocess_name='preprocessing.pkl'):
    assert(len_seq > 0)
    data, timestamps = load_stdata(fname)
    print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    data_train = data[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    X = []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _X, _Y, _timestamps_Y = st.toSeqRecent(len_seq=len_seq)
        X.append(_X)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    X = np.vstack(X)
    Y = np.vstack(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)

    X_train, Y_train = X[:-len_test], Y[:-len_test]
    X_test, Y_test = X[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    print('train shape:', X_train.shape, Y_train.shape,
          'test shape: ', X_test.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test
