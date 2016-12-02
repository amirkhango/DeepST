"""

"""
from __future__ import print_function
import h5py
import itertools
import sys
import os
import numpy as np
np.random.seed(1337)  # for reproducibility
import time
from datetime import datetime, timedelta
import pandas as pd
import scipy.sparse as sps
from deepst_flow.config import Config
# DATAPATH = Config().DATAPATH

DATAPATH = "D:/Users/junbzha/data/traffic_flow"
print(DATAPATH)

rootdir = "D:/Users/xiuwen/Project/TrajectoryMap/Data/32_30"
grid_size = 32

def load_data_from_COO_fomat(input_path):
    """timeslot,x,y,inCount,outCount,newCount,endCount"""
    data=np.loadtxt(input_path, delimiter=',')
    I=data[:,1] - 1 # x-axis
    J=data[:,2] - 1 # y-axis
    inflow, outflow = data[:,3], data[:,4]
    inflow = sps.coo_matrix((inflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    outflow = sps.coo_matrix((outflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    # newflow = sps.coo_matrix((newflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    # endflow = sps.coo_matrix((endflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    return np.asarray([inflow, outflow])


def get_file_lines(input_path):
    with open(input_path) as f:
        return len(f.readlines())

def timestamp2string(timestamps):
    return ["%s%02i" % (ts.strftime('%Y%m%d'),
            int(1+ts.to_datetime().hour*2+ts.to_datetime().minute/30)) for ts in timestamps]

def load_data(rootdir=rootdir, start='3/1/2015', end='7/1/2015', freq='30Min', year=13):
    rng = pd.date_range(start=start, end=end, periods=None, freq=freq)
    data = dict()
    data_mat = []
    avail_timestamp = []
    for timestamp in rng:
        hour, minute = timestamp.to_datetime().hour, timestamp.to_datetime().minute
        # print(timestamp, "%s%02i" % (timestamp.strftime('%Y%m%d'), int(1+hour*2+minute/30)))
        # subdir = "%04i%02i" % (timestamp.to_datetime().year, timestamp.to_datetime().month)
        fname = "%s%02i.txt" % (timestamp.strftime('%Y%m%d'), int(1+hour*2+minute/30))
        input_path = os.path.join(rootdir, fname)
        if os.path.exists(input_path) is False:
            print('file cannot be found:', input_path)
            continue
        if get_file_lines(input_path) < grid_size * grid_size * 0.25:
            continue
        avail_timestamp.append(timestamp)
        print("processing", input_path)
        data_tensor = load_data_from_COO_fomat(input_path)
        data_mat.append(data_tensor)
        data[timestamp] = data_tensor

    print("len:", len(avail_timestamp))
    h5 = h5py.File(os.path.join(DATAPATH, 'BJ', 'BJ{}_M{}_T30_Flow.h5'.format(year, grid_size)), 'w')
    h5.create_dataset("date", data=timestamp2string(avail_timestamp))
    h5.create_dataset("data", data=np.asarray(data_mat))
    h5.close()

load_data(start='7/1/2013', end='11/1/2013', year=13)
load_data(start='3/1/2014', end='7/1/2014', year=14)
load_data(start='3/1/2015', end='7/1/2015', year=15)
load_data(start='11/1/2015', end='4/11/2016', year=16)