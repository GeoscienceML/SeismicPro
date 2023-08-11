"""Implements optimized functions for various gather normalizations"""

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def clip_inplace(data, data_min, data_max):
    """Limit the `data` values. May change `data` inplace.

    `data` values outside [`data_min`, `data_max`] interval are clipped to the interval edges.

    Parameters
    ----------
    data : np.ndarray
        Data to clip.
    data_min : int, float
        Minimum value of the interval.
    data_max : int, float
        Maximum value of the interval.

    Returns
    -------
    data : np.ndarray
        Clipped data with the same shape.
    """
    data_shape = data.shape
    data = data.reshape(-1)  # may return a copy but usually a view
    for i in range(len(data)):  # pylint: disable=consider-using-enumerate
        data[i] = min(max(data[i], data_min), data_max)
    return data.reshape(data_shape)

@njit(nogil=True, parallel=True)
def get_tracewise_mean_std(data):
    """ Calculate mean and standard deviation tracewise.

    Parameters
    ----------
    data : 2d np.ndarray
        Data to calculate statistics on.

    Returns
    -------
    mean, std : tuple of 2d np.ndarrays of `data.dtype`
        Calculated statistics broadcastable to `data.shape`
    
    """
    n_traces = data.shape[0]
    mean = np.empty((n_traces, 1), dtype=data.dtype)
    std = np.empty((n_traces, 1), dtype=data.dtype)
    for i in prange(n_traces): # pylint: disable=not-an-iterable
        mean[i] = np.nanmean(data[i])
        std[i] = np.nanstd(data[i])
    return mean, std

@njit(nogil=True)
def scale_standard(data, mean, std, eps):
    r"""Scale `data` using the following formula:

    :math:`S = \frac{data - mean}{std + eps}`

    Parameters
    ----------
    data : np.ndarray
        Data to scale.
    mean : np.ndarray of `data.dtype`
        Global mean value. Must be broadcastable to `data.shape`.
    std : np.ndarray of `data.dtype`
        Global standard deviation. Must be broadcastable to `data.shape`.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchanged shape.
    """
    return (data - mean) / (std + eps)

@njit(nogil=True, parallel=True)
def get_tracewise_quantile(data, q):
    """Compute the `q`-th quantile of the data tracewise.

    Parameters
    ----------
    data : 2d np.ndarray
        Data to compute quantiles on.
    q : 1d np.ndarray of floats
        Quantiles to compute, which must be between 0 and 1 inclusive.
    
    Returns
    -------
    q : 2d np.ndarray of floats
        The array with `q`-th quantile values.
    """
    n_traces, n_quantiles = len(data), len(q)
    values = np.empty((n_quantiles, n_traces), dtype=np.float64)
    for i in prange(n_traces): # pylint: disable=not-an-iterable
        values[:, i] = np.nanquantile(data[i], q=q)
    return values.astype(data.dtype)


@njit(nogil=True)
def scale_maxabs(data, min_value, max_value, clip, eps):
    r"""Scale `data` inplace using the following formula:

    :math:`S = \frac{data}{max(|min_value|, |max_value|) + eps}`

    Parameters
    ----------
    data : 2d np.ndarray
        Data to scale.
    min_value : float or 1d array-like
        Minimum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must
        be broadcastable to `data.shape`.
    max_value : float or 1d array-like
        Maximum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must
        be broadcastable to `data.shape`.
    clip : bool
        Whether to clip scaled data to the [-1, 1] range.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchanged shape.
    """
    max_abs = np.maximum(np.abs(min_value), np.abs(max_value))
    max_abs += eps
    # Use np.atleast_2d(array).T to make the array 2-dimensional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data /= np.atleast_2d(np.asarray(max_abs)).T
    if clip:
        data = clip_inplace(data, np.float32(-1), np.float32(1))
    return data


@njit(nogil=True)
def scale_minmax(data, min_value, max_value, clip, eps):
    r"""Scale `data` inplace using the following formula:

    :math:`S = \frac{data - min_value}{max_value - min_value + eps}`

    Parameters
    ----------
    data : 2d np.ndarray
        Data to scale.
    min_value : float or 1d array-like
        Minimum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must
        be broadcastable to `data.shape`.
    max_value : float or 1d array-like
        Maximum value. Dummy trailing axes are added to the array to have at least 2 dimensions, the result must
        be broadcastable to `data.shape`.
    clip : bool
        Whether to clip scaled data to the [0, 1] range.
    eps : float
        A constant to be added to the denominator to avoid division by zero.

    Returns
    -------
    data : np.ndarray
        Scaled data with unchanged shape.
    """
    # Use np.atleast_2d(array).T to make the array 2-dimensional by adding dummy trailing axes
    # for further broadcasting to work tracewise
    data -= np.atleast_2d(np.asarray(min_value)).T
    data /= np.atleast_2d(np.asarray(max_value - min_value + eps)).T
    if clip:
        data = clip_inplace(data, np.float32(0), np.float32(1))
    return data
