"""
    Usage: python fill_missing_vals.py [fname] [preprocessing_name] [model_name]
"""
from __future__ import print_function
import sys
# sys.path.append("/home/azureuser/workspace/deepst_flow")

from deepst_flow.models.gan import generator_model
# from deepst_flow.datasets import load_bj15
from deepst_flow.datasets import load_stdata
from deepst_flow.preprocessing import TemporalConstrastNormalization, MinMaxNormalization
from deepst_flow.preprocessing import remove_incomplete_days, split_by_time, timeseries2seqs
import h5py
import numpy as np

from keras.optimizers import Adam
import os
from keras.callbacks import EarlyStopping
import cPickle as pickle
import time
import sys
import pandas as pd

np.random.seed(1337)  # for reproducibility
from deepst_flow.config import Config
DATAPATH = Config().DATAPATH

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(-1)

fname = sys.argv[1]
data, timestamps = load_stdata(os.path.join(DATAPATH, '{}.h5'.format(fname)))

T = 48
slot_time = 24. * 60 / 48
# setting
nb_flow = 2
seq_len = 3

data = data[:, :nb_flow]

preprocessing_name = sys.argv[2]
model_name = sys.argv[3]

# load TCN and MMS
fpkl = open(preprocessing_name, 'rb')
mmn = pickle.load(fpkl)
print("Load Normalization Successfully: ", mmn)

# load model
generator = generator_model(nb_flow, seq_len, 32, 32)
adam = Adam(lr=0.0001)
generator.compile(loss='mean_absolute_error', optimizer=adam)
generator.load_weights(model_name)
print("Load Model Successfully")

# working
data = mmn.transform(data)
offset = pd.DateOffset(minutes=24 * 60 // T)

from deepst_flow.utils import string2timestamp, timestamp2string
timestamps = string2timestamp(timestamps, T=T)

new_timestamps = timestamps[:seq_len]
new_data = list(data[:seq_len])

i = seq_len

while i < len(timestamps):
    if new_timestamps[-1] + offset == timestamps[i]:
        new_timestamps.append(timestamps[i])
        new_data.append(data[i])
        i += 1
    else:
        print('insert: ', new_timestamps[-1] + offset)
        new_timestamps.append(new_timestamps[-1] + offset)
        x = np.vstack(new_data[-seq_len:])
        x = x[np.newaxis]
        Y_pred = generator.predict(x)[0]
        new_data.append(Y_pred)

new_data = np.asarray(new_data)
print("shape: ", new_data.shape, "len:", len(new_timestamps))
new_data = mmn.inverse_transform(new_data)

h5 = h5py.File(os.path.join(DATAPATH, '{}_nomissing.h5'.format(fname)), 'w')
h5.create_dataset('data', data=new_data)
h5.create_dataset('date', data=timestamp2string(new_timestamps, T=48))