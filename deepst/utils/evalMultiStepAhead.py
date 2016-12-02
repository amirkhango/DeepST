from __future__ import print_function
import sys
from deepst_flow.models.gan import generator_model
from deepst_flow.datasets import load_stdata
from deepst_flow.preprocessing import MinMaxNormalization
from deepst_flow.preprocessing import remove_incomplete_days
# import h5py
import numpy as np
from keras.optimizers import Adam
import os
# from keras.callbacks import EarlyStopping
import cPickle as pickle
import time
import pandas as pd
from copy import copy
from deepst_flow.config import Config
from deepst_flow.datasets.STMatrix import STMatrix
from deepst_flow.utils.eval import rmse
np.random.seed(1337)  # for reproducibility
DATAPATH = Config().DATAPATH
print(DATAPATH)


def period_trend(period=1, trend=1):
    model_name = sys.argv[1]
    steps = 24
    Period = 7

    T = 48  # lenofday
    len_seq = 3
    nb_flow = 4
    nb_days = 120
    # divide data into two subsets:
    # Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    len_train = T * (nb_days - 7)
    len_test = T * 7

    data, timestamps = load_stdata(os.path.join(DATAPATH, 'traffic_flow_bj15_nomissing.h5'))
    print(timestamps)
    # remove a certain day which has not 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)

    # minmax_scale
    data_train = data[:len_train]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data = mmn.transform(data)

    st = STMatrix(data, timestamps, T)

    # save TCN and MMS
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:  # [tcn, mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    if period == 1 and trend == 1:
        depends = [1, 2, 3, Period*T, Period*T+1, Period*T+2, Period*T+3]
        len_close = 3
    elif period == 1:
        depends = [1] + [Period * T * j for j in xrange(1, len_seq+1)]
        len_close = 1
    elif trend == 1:
        depends = range(1, 1+len_seq)
        len_close = 3
    else:
        depends = [1]
        len_close = 1
    # else:
    #    print("unknown args")
    #    sys.exit(-1)

    generator = generator_model(nb_flow, len(depends), 32, 32)
    adam = Adam()
    generator.compile(loss='mean_absolute_error', optimizer=adam)
    generator.load_weights(model_name)

    # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
    offset_frame = pd.DateOffset(minutes=24 * 60 // T)
    Y_test = st.data[-(len_test+steps-1):]
    Y_pd_timestamps = st.pd_timestamps[-(len_test+steps-1):]

    X_test = []
    for pd_timestamp in Y_pd_timestamps:
        x = [st.get_matrix(pd_timestamp - j * offset_frame) for j in depends]
        X_test.append(np.vstack(x))
    X_test = np.asarray(X_test)

    Y_true = mmn.inverse_transform(Y_test[-len_test:])

    Y_hats = []
    for k in xrange(1, steps+1):
        print("\n\n==%d-step rmse==" % k)
        ts = time.time()
        Y_hat = generator.predict(X_test)
        Y_hats.append(copy(Y_hat))
        print('Y_hat shape', Y_hat.shape, 'X_test shape:', X_test.shape)
        # eval
        Y_pred = mmn.inverse_transform(Y_hat[-len_test:])
        rmse(Y_true, Y_pred)
        X_test_hat = copy(X_test[1:])
        for j in xrange(1, min(k, len_close) + 1):
            # Y^\hat _t replace
            X_test_hat[:, nb_flow*(j-1):nb_flow*j] = Y_hats[-j][:-j]

        X_test = copy(X_test_hat)
        print("\nelapsed time (eval): ", time.time() - ts)


def period_trend_closeness(len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
    print("start: period_trend_closeness")
    model_name = sys.argv[1]
    steps = 24
    # Period = 7

    T = 48  # lenofday
    # len_seq = 3
    nb_flow = 4
    nb_days = 120
    # divide data into two subsets:
    # Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    len_train = T * (nb_days - 7)
    len_test = T * 7

    data, timestamps = load_stdata(os.path.join(DATAPATH, 'traffic_flow_bj15_nomissing.h5'))
    print(timestamps)
    # remove a certain day which has not 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)

    # minmax_scale
    data_train = data[:len_train]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data = mmn.transform(data)

    st = STMatrix(data, timestamps, T)

    # save TCN and MMS
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:  # [tcn, mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    depends = range(1, len_closeness+1) + \
        [PeriodInterval * T * j for j in xrange(1, len_period+1)] + \
        [TrendInterval * T * j for j in xrange(1, len_trend+1)]

    generator = generator_model(nb_flow, len(depends), 32, 32)
    adam = Adam()
    generator.compile(loss='mean_absolute_error', optimizer=adam)
    generator.load_weights(model_name)

    # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
    offset_frame = pd.DateOffset(minutes=24 * 60 // T)
    Y_test = st.data[-(len_test+steps-1):]
    Y_pd_timestamps = st.pd_timestamps[-(len_test+steps-1):]

    X_test = []
    for pd_timestamp in Y_pd_timestamps:
        x = [st.get_matrix(pd_timestamp - j * offset_frame) for j in depends]
        X_test.append(np.vstack(x))
    X_test = np.asarray(X_test)

    Y_true = mmn.inverse_transform(Y_test[-len_test:])

    Y_hats = []
    for k in xrange(1, steps+1):
        print("\n\n==%d-step rmse==" % k)
        ts = time.time()
        Y_hat = generator.predict(X_test)
        Y_hats.append(copy(Y_hat))
        print('Y_hat shape', Y_hat.shape, 'X_test shape:', X_test.shape)
        # eval
        Y_pred = mmn.inverse_transform(Y_hat[-len_test:])
        rmse(Y_true, Y_pred)
        X_test_hat = copy(X_test[1:])
        for j in xrange(1, min(k, len_closeness) + 1):
            # Y^\hat _t replace
            X_test_hat[:, nb_flow*(j-1):nb_flow*j] = Y_hats[-j][:-j]

        X_test = copy(X_test_hat)
        print("\nelapsed time (eval): ", time.time() - ts)

if __name__ == '__main__':
    if int(sys.argv[2]) == 0:  # period & trend
        period_trend(1, 1)
    elif int(sys.argv[2]) == 1:  # period
        period_trend(1, 0)
    elif int(sys.argv[2]) == 2:  # trend
        period_trend(0, 1)
    elif int(sys.argv[2]) == 3:
        period_trend(0, 0)
    else:
        period_trend_closeness()
        # print("unknown args")
        # sys.exit(-1)
