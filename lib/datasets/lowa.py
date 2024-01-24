import os

import numpy as np
import pandas as pd

from .pd_dataset import PandasDataset
from .. import datasets_path
from ..utils import sample_mask, compute_mean, correntropy, extract_weighted_k_nearest_neighbors


class Lowa(PandasDataset):
    def __init__(self, freq='60T'):
        self.eval_mask = None
        df, dist, mask = self.load()
        self.dist = dist
        super().__init__(dataframe=df, u=None, name='lowa', mask=mask, freq=freq, aggr='nearest')

    def load(self, drop_all_zeros_columns=True):
        path = os.path.join(datasets_path['lowa'], 'lowa.h5')
        df = pd.read_hdf(path)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='60T')
        df = df.reindex(index=date_range.strftime("%Y-%m-%d %H:%M:%S"))
        df.index = pd.to_datetime(df.index)

        # 加载dist
        buses = df.columns.tolist()
        buses = [int(bus) for bus in buses]
        dist = self.load_distance_matrix(buses)

        # 原数据中有 19.25% 的数据为 0，绝大多数情况都是某一个bus对应的整列为0
        # 整列的0根据提供的信息，发现这些地方没有customers，所以0就是采集值
        # 在处理时候，直接忽略这些列，把剩下的相连
        # 非整列0不需要额外的预处理
        if drop_all_zeros_columns:
            # 删除全是0的列
            all_zeros_columns = df.columns[(df == 0.).all()]
            whether_drop = (df == 0.0).all(axis=0).values
            df = df.drop(columns=all_zeros_columns)
            mask = np.ones_like(df)  # mask: 全1
            # 处理dist
            drop_indices = np.where(whether_drop)[0]
            dist = self.update_distance_matrix(dist, drop_indices)
        else:
            # mask: 原df中整列为0的地方是0，其他是1
            mask = np.ones_like(df)
            mask[:, df.eq(0.).all()] = 0

        return df, dist, mask

    def update_distance_matrix(self, dist, drop_indices):
        n = dist.shape[0]
        inf = np.inf

        # 对于需要删除的每个点，更新距离
        for idx in drop_indices:
            # 获取与该点直接相连的所有点（排除与自身连接的情况）
            connected_points = [i for i in range(n) if i != idx and dist[idx, i] != inf]

            # 更新这些点之间的距离
            for i in connected_points:
                for j in connected_points:
                    if i != j:
                        # 只有当两个点都与被删除的点有连接时，才更新距离
                        dist[i, j] = dist[i, idx] + dist[idx, j]

        # 删除对应的行和列
        dist = np.delete(dist, drop_indices, axis=0)
        dist = np.delete(dist, drop_indices, axis=1)
        return dist

    def load_distance_matrix(self, buses):
        path = os.path.join(datasets_path['lowa'], 'lowa_dist.npy')
        try:
            dist = np.load(path)
        except:
            path = os.path.join(datasets_path['lowa'], 'lowa_dist.h5')
            df_dist = pd.read_hdf(path)
            num_buses = len(buses)
            dist = np.ones((num_buses, num_buses), dtype=np.float32) * np.inf
            for index, row in df_dist.iterrows():
                bus1_idx = buses.index(row['Bus A'])
                bus2_idx = buses.index(row['Bus B'])
                dist[bus1_idx, bus2_idx] = row['Length(ft.)']
                dist[bus2_idx, bus1_idx] = row['Length(ft.)']
            np.save(path, dist)
        return dist

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        finite_dist = self.dist.reshape(-1)
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        sigma = finite_dist.std()
        adj = np.exp(-np.square(self.dist / sigma))
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj


class MissingValuesLowa(Lowa):
    SEED = 223344

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesLowa, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        # 根据p和p_noise先挖一些空，挖空的地方是1
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=1,  # 1 h
                                max_seq=24,  # 1 day
                                rng=self.rng)
        # 挖空 & 原来 df 有值 是1，作为评估真正挖空的为1
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):  # 1: 最原始的数据中可用 & 没有被挖掉
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
