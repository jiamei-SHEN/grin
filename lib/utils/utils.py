import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import haversine_distances


def sample_mask(shape, p=0.002, p_noise=0., max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)  # 去重
        idxs = np.clip(idxs, 0, shape[0] - 1)  # 防止索引越界
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')  # 挖空的地方是1


def compute_mean(x, index=None):
    """Compute the mean values for each datetime. The mean is first computed hourly over the week of the year.
    Further NaN values are computed using hourly mean over the same month through the years. If other NaN are present,
    they are removed using the mean of the sole hours. Hoping reasonably that there is at least a non-NaN entry of the
    same hour of the NaN datetime in all the dataset."""
    if isinstance(x, np.ndarray) and index is not None:
        shape = x.shape
        x = x.reshape((shape[0], -1))
        df_mean = pd.DataFrame(x, index=index)
    else:
        df_mean = x.copy()
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week, df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean


def geographical_distance(x=None, to_rad=True):
    """
    Compute the as-the-crow-flies distance between every pair of samples in `x`. The first dimension of each point is
    assumed to be the latitude, the second is the longitude. The inputs is assumed to be in degrees. If it is not the
    case, `to_rad` must be set to False. The dimension of the data must be 2.
    计算 `x` 中每对样本之间的距离。每个点的第一个维度假定为纬度，第二个维度为经度。
    输入的单位假定为度。如果不是，则必须将 `to_rad` 设置为 False。数据的维度必须是 2。

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray
        array_like structure of shape (n_samples_2, 2).
    to_rad : bool
        whether to convert inputs to radians (provided that they are in degrees).

    Returns
    -------
    distances :
        The distance between the points in kilometers.
    """
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1 if it is present in the DataFrame and
    absent in the `infer_from` month.
    在评估掩码中，如果某个值在数据框中存在但在infer_from月份（就是上个月或下个月）中不存在，则该值为1

    @param pd.DataFrame df: the DataFrame.
    @param str infer_from: denotes from which month the evaluation value must be inferred.
    Can be either `previous` or `next`.
    @return: pd.DataFrame eval_mask: the evaluation mask for the DataFrame
    """
    # 创建一个与输入数据框 df 具有相同形状的掩码 mask，对应位置为NaN，则掩码值为0，否则为1。
    mask = (~df.isna()).astype('uint8')
    # 创建一个与输入数据框相同形状的评估掩码 eval_mask，并将所有值初始化为0
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns, data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('infer_from can only be one of %s' % ['previous', 'next'])
    # 提取数据框中存在的不同年份和月份的组合，并对它们进行排序
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    '''months = [(2014, 5),
                 (2014, 6),
                 (2014, 7),
                 (2014, 8),
                 (2014, 9),
                 (2014, 10),
                 (2014, 11),
                 (2014, 12),
                 (2015, 1),
                 (2015, 2),
                 (2015, 3),
                 (2015, 4)]'''
    length = len(months)
    for i in range(length):
        j = (i + offset) % length  # 选择相邻月份
        year_i, month_i = months[i]  # 获取当前月份（i）和相邻月份（j）的年份和月份
        year_j, month_j = months[j]
        # 从数据框 mask 中提取相邻月份（j）的掩码数据
        mask_j = mask[(mask.index.year == year_j) & (mask.index.month == month_j)]
        # mask_j 的1,0值不变，把第一列时间换成 mask_i 对应的时间作为 mask_i 的值
        mask_i = mask_j.shift(1, pd.DateOffset(months=12 * (year_i - year_j) + (month_i - month_j)))
        # 移除 mask_i 中重复的索引行，并保留第一个出现的行
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        # 保留 mask_i 中也存在于 mask 中的时间戳行
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        # 根据 mask_i 数据框的有效性信息来更新 eval_mask 数据框中相应时间戳的值
        # 在数据框中存在但在infer_from月份中不存在，则该值为1
        eval_mask.loc[mask_i.index] = ~mask_i.loc[mask_i.index] & mask.loc[mask_i.index]
    return eval_mask


def prediction_dataframe(y, index, columns=None, aggregate_by='mean'):
    """Aggregate batched predictions in a single DataFrame.

    @param (list or np.ndarray) y: the list of predictions.
    @param (list or np.ndarray) index: the list of time indexes coupled with the predictions.
    @param (list or pd.Index) columns: the columns of the returned DataFrame.
    @param (str or list) aggregate_by: how to aggregate the predictions in case there are more than one for a step.
    - `mean`: take the mean of the predictions
    - `central`: take the prediction at the central position, assuming that the predictions are ordered chronologically
    - `smooth_central`: average the predictions weighted by a gaussian signal with std=1
    - `last`: take the last prediction
    @return: pd.DataFrame df: the evaluation mask for the DataFrame
    """
    dfs = [pd.DataFrame(data=data.reshape(data.shape[:2]), index=idx, columns=columns) for data, idx in zip(y, index)]
    df = pd.concat(dfs)
    preds_by_step = df.groupby(df.index)
    # aggregate according passed methods
    aggr_methods = ensure_list(aggregate_by)
    dfs = []
    for aggr_by in aggr_methods:
        if aggr_by == 'mean':
            dfs.append(preds_by_step.mean())
        elif aggr_by == 'central':
            dfs.append(preds_by_step.aggregate(lambda x: x[int(len(x) // 2)]))
        elif aggr_by == 'smooth_central':
            from scipy.signal import gaussian
            dfs.append(preds_by_step.aggregate(lambda x: np.average(x, weights=gaussian(len(x), 1))))
        elif aggr_by == 'last':
            dfs.append(preds_by_step.aggregate(lambda x: x[0]))  # first imputation has missing value in last position
        else:
            raise ValueError('aggregate_by can only be one of %s' % ['mean', 'central' 'smooth_central', 'last'])
    if isinstance(aggregate_by, str):
        return dfs[0]
    return dfs


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def missing_val_lens(mask):
    m = np.concatenate([np.zeros((1, mask.shape[1])),
                        (~mask.astype('bool')).astype('int'),
                        np.zeros((1, mask.shape[1]))])
    mdiff = np.diff(m, axis=0)
    lens = []
    for c in range(m.shape[1]):
        mj, = mdiff[:, c].nonzero()
        diff = np.diff(mj)[::2]
        lens.extend(list(diff))
    return lens


def disjoint_months(dataset, months=None, synch_mode='window'):
    idxs = np.arange(len(dataset))
    months = ensure_list(months)
    # divide indices according to window or horizon
    if synch_mode == 'window':
        start, end = 0, dataset.window - 1
    elif synch_mode == 'horizon':
        start, end = dataset.horizon_offset, dataset.horizon_offset + dataset.horizon - 1
    else:
        raise ValueError('synch_mode can only be one of %s' % ['window', 'horizon'])
    # after idxs
    start_in_months = np.in1d(dataset.index[dataset._indices + start].month, months)
    end_in_months = np.in1d(dataset.index[dataset._indices + end].month, months)
    idxs_in_months = start_in_months & end_in_months
    after_idxs = idxs[idxs_in_months]
    # previous idxs
    months = np.setdiff1d(np.arange(1, 13), months)
    start_in_months = np.in1d(dataset.index[dataset._indices + start].month, months)
    end_in_months = np.in1d(dataset.index[dataset._indices + end].month, months)
    idxs_in_months = start_in_months & end_in_months
    prev_idxs = idxs[idxs_in_months]
    return prev_idxs, after_idxs


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    # 这里 theta 的平方对应论文中的 gama
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.  # mask中对应 True 的为 0.
    return weights


def correntropy(x, y, sigma):
    """
    计算两个时间序列x和y之间的correntropy。

    Parameters
    ----------
    x : np.ndarray
        第一个时间序列.
    y : np.ndarray
        第二个时间序列.
    sigma : float
        核函数的宽度参数.

    Returns
    -------
    float
        两个时间序列之间的correntropy值.
    """
    return np.mean(np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2)))


def extract_weighted_k_nearest_neighbors(similarity_matrix, k=10):
    num_meters = similarity_matrix.shape[0]
    adj = np.zeros((num_meters, num_meters), dtype=float)

    # 遍历每个meter
    for i in range(num_meters):
        # 对相似度进行排序并获取索引
        sorted_indices = np.argsort(similarity_matrix[i])
        # 选择前k个最近邻传感器的索引（不包括自身）
        k_nearest_indices = sorted_indices[1:k + 1]  # 跳过自身，选择前k个

        # 将邻接矩阵中对应位置设为相似度值，表示连接权重
        # adjacency_matrix[i, k_nearest_indices] = similarity_matrix[i, k_nearest_indices]
        # adjacency_matrix[i, k_nearest_indices] = 1

        # 计算最大和最小相似度值
        max_similarity = np.max(similarity_matrix[i, k_nearest_indices])
        min_similarity = np.min(similarity_matrix[i, k_nearest_indices])
        if max_similarity == min_similarity:
            adj[i, k_nearest_indices] = 0.5
        else:
            # 将邻接矩阵中对应位置设为相似度值的线性归一化
            adj[i, k_nearest_indices] = (similarity_matrix[i, k_nearest_indices] - min_similarity) / (
                        max_similarity - min_similarity)

    return adj
