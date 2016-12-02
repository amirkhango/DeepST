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
from deepst_flow.models.rnn import rnn_model
np.random.seed(1337)  # for reproducibility
DATAPATH = Config().DATAPATH
print(DATAPATH)


def seq_model(len_seq=3):
    model_name = sys.argv[1]
    steps = 24
    # Period = 7

    T = 48  # lenofday
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
    data = data.reshape(data.shape[0], -1)

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

    depends = range(1, 1+len_seq)
    generator = rnn_model(nb_flow, len(depends), 32, 32)
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
    Y_true = Y_true.reshape(len(Y_true), nb_flow, -1)

    Y_hats = []
    for k in xrange(1, steps+1):
        print("\n\n==%d-step rmse==" % k)
        ts = time.time()
        Y_hat = generator.predict(X_test)
        Y_hats.append(copy(Y_hat))
        print('Y_hat shape', Y_hat.shape, 'X_test shape:', X_test.shape)
        # eval
        Y_pred = mmn.inverse_transform(Y_hat[-len_test:])
        Y_pred = Y_pred.reshape(len(Y_pred), nb_flow, -1)
        rmse(Y_true, Y_pred)
        X_test_hat = copy(X_test[1:])
        for j in xrange(1, min(k, len_seq) + 1):
            # Y^\hat _t replace
            X_test_hat[:, j-1] = Y_hats[-j][:-j]

        X_test = copy(X_test_hat)
        print("\nelapsed time (eval): ", time.time() - ts)

if __name__ == '__main__':
    if int(sys.argv[2]) > 0:
        seq_model(len_seq=int(sys.argv[2]))
    else:
        print("unknown args")
        sys.exit(-1)
