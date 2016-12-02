# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import cPickle as pickle
import time
import numpy as np
import h5py

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import TaxiBJ
np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH
CACHEDATA = False
nb_epoch = 50  # 50
nb_epoch_cont = 100  # 2
batch_size = 32
T = 48

lr = float(sys.argv[1])
print("lr: ", lr)
len_closeness = int(sys.argv[2])
len_period = int(sys.argv[3])
len_trend = int(sys.argv[4])
nb_residual_unit = int(sys.argv[5])
len_seq = len_closeness + len_period + len_trend
if len_seq <= 0:
    print('len_seq:', len_seq)
    sys.exit(-1)

nb_flow = 2
# divide data into two subsets: Train & Test
days_test = 7 * 4
len_test = T * days_test
# grid size
map_height, map_width = 32, 32
path_result = 'RET'
path_model = 'MODEL'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


def build_model(external_dim):
    # model = seqCNN_CPT(nb_flow, len_seq, 32, 32)
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def read_cache(fname):
    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()
    return X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test


def load_BJ(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))
    X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test = read_cache(
        fname)
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    h5.create_dataset('external_dim', data=int(external_dim))
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def main():
    # load data
    ts = time.time()

    fname = os.path.join(DATAPATH, 'BJ', 'TaxiBJ_x2_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_BJ(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiBJ.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test)

    print("\ntimestamp_test: ", timestamp_test[0::T])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    ts = time.time()
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f Train rmse: %.6f %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f Test rmse: %.6f %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    ts = time.time()
    # filepath = 'model_lr{}_c{}_p{}_t{}_resnet{}_cont_best.h5'.format(lr, len_closeness, len_period, len_trend, nb_residual_unit)
    fname_param = os.path.join(
        'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='rmse', patience=5, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
                        model_checkpoint], validation_data=(X_test, Y_test))
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    ##
    print('=' * 10)
    print('last model (not the best on training set)')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f Train rmse: %.6f %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f Test rmse: %.6f %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('best model on training set')
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f Train rmse: %.6f %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f Test rmse: %.6f %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))

if __name__ == '__main__':
    main()
