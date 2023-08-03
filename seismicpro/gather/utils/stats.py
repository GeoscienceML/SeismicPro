"""Implements function for calculating traces statistics in predefined windows"""
import numpy as np
from numba import njit, prange


@njit(nogil=True, parallel=True)
def compute_rms(traces, start_ixs, end_ixs):
    """Compute traces RMS in windows defined by `start_ixs` and `end_ixs`."""
    temp = np.empty(len(traces), dtype=np.float32)
    for i in prange(len(traces)):  # pylint: disable=not-an-iterable
        temp[i] = np.nanmean(traces[i, start_ixs[i]: end_ixs[i]]**2)**.5
    return temp


@njit(nogil=True, parallel=True)
def compute_abs(traces, start_ixs, end_ixs):
    """Compute mean absolute amplitudes in windows defined by `start_ixs` and `end_ixs`."""
    temp = np.empty(len(traces), dtype=np.float32)
    for i in prange(len(traces)):  # pylint: disable=not-an-iterable
        temp[i] = np.nanmean(np.abs(traces[i, start_ixs[i]: end_ixs[i]]))
    return temp
