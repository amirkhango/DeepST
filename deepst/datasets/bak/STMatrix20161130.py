from __future__ import print_function
import os
import pandas as pd
import numpy as np

from . import load_stdata
from ..config import Config
from ..utils import string2timestamp


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def toSeq(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        X = []
        Y = []
        depends = range(1, len_closeness+1) + \
            [PeriodInterval * self.T * j for j in xrange(1, len_period+1)] + \
            [TrendInterval * self.T * j for j in xrange(1, len_trend+1)]
        # [1, 2, 3, Period*self.T, Period*self.T, Period*self.T+2, Period*self.T+3]
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        '''
        break_points_idx = []
        for i in xrange(1, len(self.pd_timestamps)):
            if self.pd_timestamps[i] - self.pd_timestamps[i-1] != offset_frame:
                break_points_idx.append(i)
        '''

        while i < len(self.pd_timestamps):
            '''
            if self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depends]) is False:
                i += 1
                continue
            '''
            x = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends]
            y = self.get_matrix(self.pd_timestamps[i])
            X.append(np.vstack(x))
            Y.append(y)
            i += 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        print("X shape: ", X.shape, "Y shape:", Y.shape)
        return X, Y

    def toSeq2(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in xrange(1, len_period+1)],
                   [TrendInterval * self.T * j for j in xrange(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y

    def toSeq3(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in xrange(1, len_period+1)],
                   [TrendInterval * self.T * j for j in xrange(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y

    def toSeq4(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in xrange(1, len_period+1)],
                   [TrendInterval * self.T * j for j in xrange(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


    def toSeqRecent(self, len_seq):
        """Recent dependents
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        X = []
        Y = []
        timestamps_Y = []
        closeness_depend = range(1, len_seq+1)

        i = len_seq
        while i < len(self.pd_timestamps):
            Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in closeness_depend])
            if Flag is False:
                i += 1
                continue
            x = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in closeness_depend]
            X.append(np.asarray(x))
            y = self.get_matrix(self.pd_timestamps[i])
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        print("X shape: ", X.shape, "Y shape:", Y.shape)
        return X, Y, timestamps_Y


    def toSeqParallel(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        # parallel version for accelerating
        # offset_week = pd.DateOffset(days=7)
        # from multiprocessing import Pool
        from pathos.multiprocessing import ProcessingPool as Pool
        pool = Pool(16)
        print("number threads: 16")
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in xrange(1, len_period+1)],
                   [TrendInterval * self.T * j for j in xrange(1, len_trend+1)]]

        start_id = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        idx  = range(start_id, len(self.pd_timestamps))
        def func(i):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break                
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])
            if Flag is False:
                return None
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])  
            ret = []
            # if len_closeness > 0:
            ret.append(np.vstack(x_c))
            # if len_period > 0:
            ret.append(np.vstack(x_p))
            # if len_trend > 0:
            ret.append(np.vstack(x_t))
            ret.append(y)
            return ret

        rets = pool.map(func, idx)
        for j, ret in enumerate(rets):
            if ret is None:
                continue
            XC.append(ret[0])
            XP.append(ret[1])
            XT.append(ret[2])
            Y.append(ret[3])
            timestamps_Y.append(self.timestamps[idx[j]])
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


    def toSeq_PeriodTrend(self, length=3, Period=7):
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        X = []
        Y = []
        i = self.T * Period + length
        depends = [1, 2, 3, Period*self.T, Period*self.T+1, Period*self.T+2, Period*self.T+3]
        while i < len(self.pd_timestamps):
            x = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends]
            y = self.get_matrix(self.pd_timestamps[i])
            X.append(np.vstack(x))
            Y.append(y)
            i += 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        print("X shape: ", X.shape, "Y shape:", Y.shape)
        return X, Y

    def toSeq_Period(self, length=3, Period=7):
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        X = []
        Y = []
        i = self.T * Period * length
        depends = [1] + [Period * self.T * j for j in xrange(1, length+1)]  # [Period*self.T, Period*self.T, Period*self.T, Period*self.T]
        while i < len(self.pd_timestamps):
            x = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends]
            y = self.get_matrix(self.pd_timestamps[i])
            X.append(np.vstack(x))
            Y.append(y)
            i += 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        print("X shape: ", X.shape, "Y shape:", Y.shape)
        return X, Y

    def toSeq_Trend(self, length=3):
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        X = []
        Y = []
        i = length
        depends = range(1, 1+length)
        print(depends)
        while i < len(self.pd_timestamps):
            x = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends]
            y = self.get_matrix(self.pd_timestamps[i])
            X.append(np.vstack(x))
            Y.append(y)
            i += 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        print("X shape: ", X.shape, "Y shape:", Y.shape)
        return X, Y

if __name__ == '__main__':
    DATAPATH = Config().DATAPATH
    print(DATAPATH)
    data, timestamps = load_stdata(os.path.join(DATAPATH, 'traffic_flow_bj15_nomissing.h5'))
    st = STMatrix(data, timestamps)
    # st = STMatrix(os.path.join(DATAPATH, 'traffic_flow_bj15.h5'))
    print(st.data.shape, len(st.timestamps))
