from __future__ import print_function
import sys
from deepst_flow.models.STConvolution import seqCNN_CPT2
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


def period_trend_closeness(len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
    print("start: period_trend_closeness")
    model_name = sys.argv[1]
    steps = 24
    # Period = 7

    T = 48  # lenofday
    # len_seq = 3
    nb_flow = 2
    # nb_days = 120
    # divide data into two subsets:
    # Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    # len_train = T * (nb_days - 7)
    len_test = T * 7

    data, timestamps = load_stdata(os.path.join(DATAPATH, 'traffic_flow_bj15_nomissing.h5'))
    print(timestamps)
    # remove a certain day which has not 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    # minmax_scale
    data_train = data[-len_test:]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data = mmn.transform(data)

    st = STMatrix(data, timestamps, T)

    # save TCN and MMS
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:  # [tcn, mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    depends = [range(1, len_closeness+1),
               [PeriodInterval * T * j for j in xrange(1, len_period+1)],
               [TrendInterval * T * j for j in xrange(1, len_trend+1)]]
    if len_closeness > 0:
        c_conf = (nb_flow, len_closeness, 32, 32)
    else:
        c_conf = None
    if len_period > 0:
        p_conf = (nb_flow, len_period, 32, 32)
    else:
        p_conf = None
    if len_trend > 0:
        t_conf = (nb_flow, len_trend, 32, 32)
    else:
        t_conf = None
    generator = seqCNN_CPT2(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf)

    adam = Adam()
    generator.compile(loss='mean_absolute_error', optimizer=adam)
    generator.load_weights(model_name)

    # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
    offset_frame = pd.DateOffset(minutes=24 * 60 // T)
    Y_test = st.data[-(len_test+steps-1):]
    Y_pd_timestamps = st.pd_timestamps[-(len_test+steps-1):]

    XC = []
    XP = []
    XT = []
    for pd_timestamp in Y_pd_timestamps:
        # x = [st.get_matrix(pd_timestamp - j * offset_frame) for j in depends]
        # X_test.append(np.vstack(x))
        x_c = [st.get_matrix(pd_timestamp - j * offset_frame) for j in depends[0]]
        x_p = [st.get_matrix(pd_timestamp - j * offset_frame) for j in depends[1]]
        x_t = [st.get_matrix(pd_timestamp - j * offset_frame) for j in depends[2]]
        if len_closeness > 0:
            XC.append(np.vstack(x_c))
        if len_period > 0:
            XP.append(np.vstack(x_p))
        if len_trend > 0:
            XT.append(np.vstack(x_t))
    if len_closeness > 0:
        XC = np.asarray(XC)
    if len_period > 0:
        XP = np.asarray(XP)
    if len_trend > 0:
        XT = np.asarray(XT)
    print(XC.shape, XP.shape, XT.shape)

    # X_test = np.asarray(X_test)
    XAll = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC, XP, XT]):
        if l > 0:
            XAll.append(X_)

    Y_true = mmn.inverse_transform(Y_test[-len_test:])
    Y_hats = []

    # for i in xrange(len(XAll[0])):
    #    x = []
    #    for _X in XAll:
    #        x.append([_X[i]])

    for k in xrange(1, steps+1):
        print("\n\n==%d-step rmse==" % k)
        ts = time.time()
        # k^th predicted sequence
        Y_hat = generator.predict(XAll)
        Y_hats.append(copy(Y_hat))
        print('Y_hat shape', Y_hat.shape)
        # eval
        Y_pred = mmn.inverse_transform(Y_hat[-len_test:])
        rmse(Y_true, Y_pred)
        X_hat = []
        for _X in XAll:
            X_hat.append(copy(_X[1:]))
        # X_hat = [XC[1:], XP[1:], XT[1:]]  # copy(X_test[1:])

        '''
        # for j in xrange(len_closeness-1, 0):
        for j in xrange(1, min(k, len_closeness) + 1):
            # last sequence -j
            if j == 1:
                X_hat[0][:, -1 * nb_flow:] = Y_hats[-j][:-j]
            else:
                X_hat[0][:, nb_flow*(-j):nb_flow*(-j+1)] = Y_hats[-j][:-j]
        '''

        XC_hat = X_hat[0]
        len_replace = min(k, len_closeness)

        for j in xrange(len_replace):
            # XC_hat[:, nb_flow*(j):nb_flow*(j+1)] = Y_hats[-(j+1)][:-(len_replace-j)]
            XC_hat[:, nb_flow*(j):nb_flow*(j+1)] = Y_hats[-(j+1)][:-(j+1)]
            # XC_hat[:, nb_flow*(j):nb_flow*(j+1)] = Y_hats[j][:-(j+1)]
        # for j in xrange(1, + 1):
        #    XC_hat[:, ] =

        # for j in xrange(1, min(k, len_closeness) + 1):
            # Y^\hat _t replace
        #    X_hat[0][:, nb_flow*(j-1):nb_flow*j] = Y_hats[-j][:-j]

        XAll = X_hat
        print("\nelapsed time (eval): ", time.time() - ts)

if __name__ == '__main__':
    period_trend_closeness(len_closeness=int(sys.argv[2]), len_period=int(sys.argv[3]), len_trend=int(sys.argv[4]))
