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
DATAPATH = Config().DATAPATH
print(DATAPATH)

rootdir = "D:/Users/xiuwen/Project/TrajectoryMap/Data/32InOut"
grid_size = 32

'''
for i, t in enumerate(['2015030148']):
    year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:10])-1
    key = pd.Timestamp(datetime(year, month, day, hour=slot//2, minute=(slot % 2)*30))
    print(key)
    print(key - pd.DateOffset(days=1))
    print(key + pd.DateOffset(minutes=30))
    print(key + pd.DateOffset(days=7))
'''

def difference(data, avail_timestamp, offset=pd.DateOffset(minutes=30)):
    diff_data = dict()
    data_mat = []
    avail_timestamp_new = []
    for timestamp in avail_timestamp:
        backshift = timestamp - offset
        if backshift not in data.keys():
            # print("%s is not available" % backshift)
            continue
        avail_timestamp_new.append(timestamp)
        diff_data[timestamp] = data[timestamp] - data[backshift]
        data_mat.append(data[timestamp] - data[backshift])
    return diff_data, data_mat, avail_timestamp_new


def day_difference(data, avail_timestamp, offset=pd.DateOffset(days=1)):
    doubly_diff_data = dict()
    data_mat = []
    avail_timestamp_new = []
    for timestamp in avail_timestamp:
        backshift = timestamp - offset
        if backshift not in data.keys():
            # print("%s is not available" % backshift)
            continue
        avail_timestamp_new.append(timestamp)
        doubly_diff_data[timestamp] = data[timestamp] - data[backshift]
        data_mat.append(data[timestamp] - data[backshift])
    return doubly_diff_data, data_mat, avail_timestamp_new

def week_difference(pd, offset=pd.DateOffset(days=1)):
    pass

def load_data_from_COO_fomat(input_path):
    """timeslot,x,y,inCount,outCount,newCount,endCount"""
    data=np.loadtxt(input_path, delimiter=',')
    I=data[:,1] - 1 # x-axis
    J=data[:,2] - 1 # y-axis
    inflow, outflow, newflow, endflow = data[:,3], data[:,4], data[:,5], data[:,6]
    inflow = sps.coo_matrix((inflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    outflow = sps.coo_matrix((outflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    newflow = sps.coo_matrix((newflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    endflow = sps.coo_matrix((endflow,(I,J)), shape=(grid_size, grid_size) ).toarray()
    return np.asarray([inflow, outflow, newflow, endflow])


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
    h5 = h5py.File(os.path.join(DATAPATH, 'BJ{}_M{}_T30_Flow.h5'.format(year, grid_size)), 'w')
    h5.create_dataset("date", data=timestamp2string(avail_timestamp))
    h5.create_dataset("data", data=np.asarray(data_mat))
    h5.close()
    '''
    diff_data, data_mat, avail_timestamp = difference(data, avail_timestamp)
    h5 = h5py.File('diff_traffic_flow.h5', 'w')
    h5.create_dataset("date", data=timestamp2string(avail_timestamp))
    h5.create_dataset("data", data=np.asarray(data_mat))
    h5.close()

    doubly_diff_data, data_mat, avail_timestamp = day_difference(diff_data, avail_timestamp)
    h5 = h5py.File('doubly_diff_traffic_flow.h5', 'w')
    h5.create_dataset("date", data=timestamp2string(avail_timestamp))
    h5.create_dataset("data", data=np.asarray(data_mat))
    h5.close()
    '''

# load_data(start='7/1/2013', end='11/1/2013', year=13)
load_data(start='3/1/2014', end='7/1/2014', year=14)
load_data(start='3/1/2015', end='7/1/2015', year=15)
load_data(start='11/1/2015', end='4/10/2016', year=16)