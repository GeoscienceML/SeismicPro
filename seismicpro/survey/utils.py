"""General survey processing utils"""

import numpy as np
from numba import njit, prange


@njit(nogil=True, parallel=True)
def calculate_trace_stats(trace):
    """Calculate min, max, mean and var of trace amplitudes."""
    trace_min = np.float32(np.inf)
    trace_max = np.float32(-np.inf)

    # Traces are generally centered around zero so variance is calculated in a single pass by accumulating sum and
    # sum of squares of trace amplitudes as float64 for numerical stability
    trace_sum = np.float64(0)
    trace_sum_sq = np.float64(0)

    # min, max and += are supported numba reductions and can be safely used in a prange
    for i in prange(len(trace)):  # pylint: disable=not-an-iterable
        sample = trace[i]
        sample64 = np.float64(sample)
        trace_min = min(sample, trace_min)
        trace_max = max(sample, trace_max)
        trace_sum += sample64
        trace_sum_sq += sample64**2
    trace_mean = trace_sum / len(trace)
    trace_var = trace_sum_sq / len(trace) - trace_mean**2
    return trace_min, trace_max, trace_mean, trace_var
