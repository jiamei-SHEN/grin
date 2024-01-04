import numpy as np
import pandas as pd
import torch


class PandasDataset:
    def __init__(self, dataframe: pd.DataFrame, u: pd.DataFrame = None, name='pd-dataset', mask=None, freq=None,
                 aggr='sum', **kwargs):
        """
        Initialize a tsl dataset from a pandas dataframe.


        :param dataframe: dataframe containing the data, shape: n_steps, n_nodes
        :param u: dataframe with exog variables 外生变量？？
        :param name: optional name of the dataset
        :param mask: mask for valid data (1:valid, 0:not valid)
        :param freq: force a frequency (possibly by resampling)
        :param aggr: aggregation method after resampling
        """
        super().__init__()
        self.name = name

        # set dataset dataframe
        self.df = dataframe

        # set optional exog_variable dataframe 接下来的代码将处理可选的外生变量数据框
        # make sure to consider only the overlapping part of the two dataframes 代码将仅考虑两个数据框中重叠的部分
        # assumption u.index \in df.index 假设外生变量数据框 u 的时间索引应该包含在主数据框 dataframe 的时间索引中
        idx = sorted(self.df.index)
        self.start = idx[0]
        self.end = idx[-1]
        # 确保外生变量数据与主数据集的时间轴一致，以便进行后续的操作和分析。
        # 如果外生变量的时间索引不在主数据框的时间索引范围内，那么它将被截断以适应主数据框的时间范围。
        if u is not None:
            self.u = u[self.start:self.end]
        else:
            self.u = None

        if mask is not None:
            mask = np.asarray(mask).astype('uint8')
        self._mask = mask  # 私有属性

        if freq is not None:
            self.resample_(freq=freq, aggr=aggr)  # 重新采样时间序列数据集
        else:
            self.freq = self.df.index.inferred_freq
            # make sure that all the dataframes are aligned
            self.resample_(self.freq, aggr=aggr)

        assert 'T' in self.freq  # T表示分钟 比如有个调用freq=60T
        self.samples_per_day = int(60 / int(self.freq[:-1]) * 24)

    def __repr__(self):
        return "{}(nodes={}, length={})".format(self.__class__.__name__, self.n_nodes, self.length)

    @property
    def has_mask(self):
        return self._mask is not None

    @property
    def has_u(self):
        return self.u is not None

    def resample_(self, freq, aggr):
        resampler = self.df.resample(freq)
        idx = self.df.index
        if aggr == 'sum':
            self.df = resampler.sum()
        elif aggr == 'mean':
            self.df = resampler.mean()
        elif aggr == 'nearest':
            self.df = resampler.nearest()
        else:
            raise ValueError(f'{aggr} if not a valid aggregation method.')

        if self.has_mask:
            resampler = pd.DataFrame(self._mask, index=idx).resample(freq)
            self._mask = resampler.min().to_numpy()

        if self.has_u:
            resampler = self.u.resample(freq)
            self.u = resampler.nearest()
        self.freq = freq

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    @property
    def length(self):
        return self.df.values.shape[0]

    @property
    def n_nodes(self):
        return self.df.values.shape[1]

    @property
    def mask(self):
        if self._mask is None:
            return np.ones_like(self.df.values).astype('uint8')
        return self._mask

    def numpy(self, return_idx=False):
        if return_idx:
            return self.numpy(), self.df.index
        return self.df.values

    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)

    def __len__(self):
        return self.length

    @staticmethod
    def build():
        raise NotImplementedError

    def load_raw(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
