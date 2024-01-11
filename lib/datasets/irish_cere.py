import os

import numpy as np
import pandas as pd

from .pd_dataset import PandasDataset
from .. import datasets_path
from ..utils import sample_mask, compute_mean, correntropy, extract_weighted_k_nearest_neighbors


class IrishCERE(PandasDataset):

    def __init__(self, freq='30T'):
        self.eval_mask = None
        df, mask = self.load()
        super().__init__(dataframe=df, u=None, name='cere', mask=mask, freq=freq, aggr='nearest')

    def load(self, impute_nans=True):
        path = os.path.join(datasets_path['cere'], 'cere_sme.h5')
        df = pd.read_hdf(path)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='30T')
        '''difference: ['2010-03-28 00:30:00', '2010-03-28 01:00:00']'''
        df = df.reindex(index=date_range.strftime("%Y-%m-%d %H:%M:%S"))
        df.index = pd.to_datetime(df.index)
        mask = (~np.isnan(df.values)).astype(int)  # 原 df 中 nan 的地方是 0 ，非 nan 是 1
        if impute_nans:
            df = df.fillna(compute_mean(df))
        return df, mask

    def get_similarity(self, k=10, force_symmetric=False, sparse=False, **kwargs):
        # 以周为单位的重采样
        df_weekly = self.df.resample('W').sum()
        # 计算时间序列之间的相似性
        # 初始化相似度矩阵
        num_meters = len(df_weekly.columns)
        similarity_matrix = np.zeros((num_meters, num_meters))
        sigma = np.std(df_weekly.values.ravel())
        for i in range(num_meters):
            for j in range(i, num_meters):  # 因为相似度矩阵是对称的，只需计算上三角部分
                data_i = df_weekly.iloc[:, i].values  # 获取第i个传感器的数据
                data_j = df_weekly.iloc[:, j].values  # 获取第j个传感器的数据
                similarity = correntropy(data_i, data_j, sigma)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # 填充相似度矩阵的对称部分

        # 提取k-nearest neighbor图，构建邻接矩阵
        adj = extract_weighted_k_nearest_neighbors(similarity_matrix, 10)
        return adj


class MissingValuesCERE(IrishCERE):
    SEED = 223344

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesCERE, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=4,  # 2 h
                                max_seq=2 * 24 * 2,  # 2 days
                                rng=self.rng)
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]
