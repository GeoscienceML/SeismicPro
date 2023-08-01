"""Utility functions for tutorials"""

import numpy as np
from scipy import signal


def generate_trace(reflection_event_times=(20, 1400, 2400), reflection_event_amplitude=(6.0, -12.0, 8.0),
                   nmo_velocity=(1.6, 2.0, 2.4), wavelet_length=50, wavelet_width=5, **kwargs):
    """Generates a seismic trace with Ricker impulse using reflectivity parameters and trace's headers.

    Parameters
    ----------
    reflection_event_time : 1d np.ndarray, defaults to (20, 1400, 2400)
        Zero-offset times of reflection events measured in ms.
    reflection_event_amplitude : 1d np.ndarray, defaults to (6.0, -12.0, 8.0)
        Amplitudes of reflection events.
    nmo_velocity : 1d np.ndarray, defaults to (1.6, 2.0, 2.4)
        NMO velocities for the reflection events, m/ms.
    wavelet_length : int, defaults to 50
        Overall length of the vector with Ricker wavelet. Equivalent of `points` parameter of `scipy.signal.ricker`.
    wavelet_width : int, defaults to 5
         Width parameter of the wavelet itself. Equivalent of `a` parameter of `scipy.signal.ricker`.
    kwargs : dict
        Dict with trace header values. This function uses TRACE_SAMPLE_COUNT, TRACE_SAMPLE_INTERVAL and
        offset.

    Returns
    -------
    trace : 1d-ndarray
        Generated seismic trace.
    """
    n_samples = kwargs.get('TRACE_SAMPLE_COUNT')
    sample_rate = kwargs.get('TRACE_SAMPLE_INTERVAL')
    offset = kwargs.get('offset')
    sample_rate = sample_rate / 1000  # cast sample rate (dt) from microseconds to milliseconds
    times = np.array(reflection_event_times) / sample_rate  # cast to samples
    reflections = np.array(reflection_event_amplitude)
    velocities = np.array(nmo_velocity)

    equal_lengths = (len(times) == len(reflections) == len(velocities))
    if not equal_lengths:
        raise ValueError("reflection_event_times, reflection_event_amplitude and nmo_velocity"
                         "should have equal lengths")

    # Inversed normal moveout
    shifted_times = ((times**2 + offset**2 / (velocities * sample_rate)**2)**0.5).astype(int)
    ref_series = np.zeros(max(n_samples, max(shifted_times)) + 1)

    # Tweak reflection event amplitudes to make Survey and Gathers statistics differ
    reflections = np.random.normal(reflections, np.abs(reflections) / 5)

    ref_series[shifted_times] = reflections
    ref_series[min(shifted_times):] += np.random.normal(1, 0.5, size=len(ref_series)-min(shifted_times))

    # Generate "seismic" signal by convolving reflectivity series with the "impulse" wavelet
    trace = np.convolve(ref_series, signal.ricker(wavelet_length, wavelet_width), mode='same')[:n_samples]
    trace = trace.astype(np.float32)

    return trace
