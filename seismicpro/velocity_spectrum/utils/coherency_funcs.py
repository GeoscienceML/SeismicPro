""" Functions for estimating hodograph coherency. """

# pylint: disable=not-an-iterable, missing-function-docstring
import numpy as np
from numba import prange, jit_module


def stacked_amplitude(corrected_gather, amplify_factor=0, abs=True):
    numerator = np.empty_like(corrected_gather[0])
    denominator = np.ones_like(corrected_gather[0])
    for i in prange(corrected_gather.shape[1]):
        numerator[i] = np.nansum(corrected_gather[:, i])
        if abs:
            numerator[i] = np.abs(numerator[i])
        n = max(np.sum(~np.isnan(corrected_gather[:, i])), np.int64(1))
        numerator[i] = numerator[i] * ((amplify_factor / np.sqrt(n)) + ((1 - amplify_factor) / n))
    return numerator, denominator


def normalized_stacked_amplitude(corrected_gather):
    numerator = np.empty_like(corrected_gather[0])
    denominator = np.empty_like(corrected_gather[0])
    for i in prange(corrected_gather.shape[1]):
        numerator[i] = np.abs(np.nansum(corrected_gather[:, i]))
        denominator[i] = np.nansum(np.abs(corrected_gather[:, i]))
    return numerator, denominator


def semblance(corrected_gather):
    numerator = np.empty_like(corrected_gather[0])
    denominator = np.empty_like(corrected_gather[0])
    for i in prange(corrected_gather.shape[1]):
        numerator[i] = (np.nansum(corrected_gather[:, i]) ** 2) / max(np.sum(~np.isnan(corrected_gather[:, i])), 1)
        denominator[i] = np.nansum(corrected_gather[:, i] ** 2)
    return numerator, denominator


def crosscorrelation(corrected_gather):
    numerator = np.empty_like(corrected_gather[0])
    denominator = np.ones_like(corrected_gather[0])
    for i in prange(corrected_gather.shape[1]):
        numerator[i] = ((np.nansum(corrected_gather[:, i]) ** 2) - np.nansum(corrected_gather[:, i] ** 2)) / 2
    return numerator, denominator


def energy_normalized_crosscorrelation(corrected_gather):
    numerator = np.empty_like(corrected_gather[0])
    denominator = np.empty_like(corrected_gather[0])
    for i in prange(corrected_gather.shape[1]):
        input_enerty =  np.nansum(corrected_gather[:, i] ** 2)
        output_energy = np.nansum(corrected_gather[:, i]) ** 2
        numerator[i] = (output_energy - input_enerty) / max(np.sum(~np.isnan(corrected_gather[:, i])) - 1, 1)
        denominator[i] = input_enerty
    return numerator, denominator


ALL_FASTMATH_FLAGS  = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
jit_module(nopython=True, nogil=True, parallel=True, fastmath=ALL_FASTMATH_FLAGS - {'nnan'})
