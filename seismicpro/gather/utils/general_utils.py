"""General gather processing utils"""

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def convert_times_to_mask(times, n_samples, sample_interval, delay):
    """Convert `times` to indices of nearest samples in an array of `n_samples` evenly spaced elements starting from
    `delay` with a step of `sample_interval`. Return a boolean mask with shape (n_times, n_samples) with `False` values
    before calculated time index for each row and `True` values after.

    Examples
    --------
    >>> times = np.array([0, 4, 6])
    >>> convert_times_to_mask(times, n_samples=5, sample_interval=2, delay=0)
    array([[ True,  True,  True,  True,  True],
           [False, False,  True,  True,  True],
           [False, False, False,  True,  True]])

    Parameters
    ----------
    times : 1d np.ndarray
        Time values to construct the mask. Measured in milliseconds.
    n_samples : int
        The number of samples to consider.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask with shape (n_times, n_samples).
    """
    times_indices = np.rint((times - delay) / sample_interval)
    return np.arange(n_samples) >= times_indices.reshape(-1, 1)


@njit(nogil=True, parallel=True)
def convert_mask_to_pick(mask, threshold, sample_interval, delay):
    """Convert a first breaks `mask` into an array of arrival times.

    The mask has shape (n_traces, n_samples), each its value represents a probability of corresponding index along the
    trace to follow the first break. A naive approach is to define the first break time index as the location of the
    first trace value exceeding the `threshold`. Unfortunately, it results in noisy predictions, so the following
    conversion procedure is proposed as it appears to be more stable:
    1. Binarize the mask according to the specified `threshold`,
    2. Find the longest sequence of ones in the binarized mask for each trace and save indices of the first elements of
       the found sequences,
    3. Multiply these indices by `sample_interval` and add `delay` to obtain times of first breaks.

    Examples
    --------
    >>> mask = np.array([[  1, 1, 1, 1, 1],
    ...                  [  0, 0, 1, 1, 1],
    ...                  [0.6, 0, 0, 1, 1]])
    >>> convert_mask_to_pick(mask, samples, threshold=0.5, sample_interval=2, delay=0)
    array([0, 4, 6])

    Parameters
    ----------
    mask : 2d np.ndarray
        An array with shape (n_traces, n_samples), with each value representing a probability of corresponding index
        along the trace to follow the first break.
    threshold : float
        A threshold for trace mask value to refer its index to be either pre- or post-first break.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.

    Returns
    -------
    times : np.ndarray with length n_traces
        Start time of the longest sequence with `mask` values greater than the `threshold` for each trace. Measured in
        milliseconds.
    """
    picking_times = np.empty(len(mask), dtype=np.int32)
    for i in prange(len(mask)):  # pylint: disable=not-an-iterable
        trace = mask[i]
        max_len, curr_len, picking_ix = 0, 0, 0
        for j, sample in enumerate(trace):
            # Count length of current sequence of ones
            if sample >= threshold:
                curr_len += 1
            else:
                # If the new longest sequence found
                if curr_len > max_len:
                    max_len = curr_len
                    picking_ix = j
                curr_len = 0
        # If the longest sequence found in the end of the trace
        if curr_len > max_len:
            picking_ix = len(trace)
            max_len = curr_len
        picking_times[i] = delay + sample_interval * (picking_ix - max_len)
    return picking_times


@njit(nogil=True)
def mute_gather(gather_data, muting_times, sample_interval, delay, fill_value):
    """Fill area before `muting_times` with `fill_value`.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to mute.
    muting_times : 1d np.ndarray
        Time values up to which muting is performed. Its length must match `gather_data.shape[0]`. Measured in
        milliseconds.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.
    fill_value : float
         A value to fill the muted part of the gather with.

    Returns
    -------
    gather_data : 2d np.ndarray
        Muted gather data.
    """
    mask = convert_times_to_mask(muting_times, gather_data.shape[1], sample_interval, delay)
    data_shape = gather_data.shape
    gather_data = gather_data.reshape(-1)
    mask = mask.reshape(-1)
    gather_data[~mask] = fill_value
    return gather_data.reshape(data_shape)
