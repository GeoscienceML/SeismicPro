"""Implements gather quality control metrics."""

import numpy as np
from ..metrics import PipelineMetric


class SignalLeakage(PipelineMetric):
    """Calculate signal leakage after ground-roll attenuation.

    The metric is based on the assumption that a vertical velocity spectrum calculated for the difference between
    processed and source gathers should not have pronounced energy maxima.
    """
    is_lower_better = True
    min_value = 0
    max_value = None
    views = ("plot_diff_gather", "plot_diff_velocity_spectrum")
    args_to_unpack = ("gather_before", "gather_after")

    @staticmethod
    def get_diff_gather(gather_before, gather_after):
        """Construct a new gather whose amplitudes are element-wise differences of amplitudes from `gather_after` and
        `gather_before`."""
        if ((gather_before.shape != gather_after.shape) or (gather_before.delay != gather_after.delay) or
            (gather_before.sample_interval != gather_after.sample_interval)):
            raise ValueError("Both gathers should have the same shape and samples")
        gather_diff = gather_after.copy(ignore=["data", "headers", "samples"])
        gather_diff.data = gather_after.data - gather_before.data
        return gather_diff

    def __call__(self, gather_before, gather_after, velocities=None):
        """Calculate signal leakage when moving from `gather_before` to `gather_after`."""
        gather_diff = self.get_diff_gather(gather_before, gather_after)
        spectrum_diff = gather_diff.calculate_vertical_velocity_spectrum(velocities)
        spectrum_before = gather_before.calculate_vertical_velocity_spectrum(velocities)
        # TODO: update calculation logic, probably sum of semblance values of gather diff along stacking velocity,
        # picked for gather_before / gather_after will perform way better
        signal_leakage = (spectrum_diff.velocity_spectrum.ptp(axis=1) /
                          (1 + 1e-6 - spectrum_before.velocity_spectrum.ptp(axis=1)))
        return max(0, np.max(signal_leakage))

    def plot_diff_gather(self, gather_before, gather_after, velocities=None, *, ax, **kwargs):
        """Plot the difference between `gather_after` and `gather_before`."""
        _ = velocities
        gather_diff = self.get_diff_gather(gather_before, gather_after)
        gather_diff.plot(ax=ax, **kwargs)

    def plot_diff_velocity_spectrum(self, gather_before, gather_after, velocities=None, *, ax, **kwargs):
        """Plot a velocity spectrum of the difference between `gather_after` and `gather_before`."""
        gather_diff = self.get_diff_gather(gather_before, gather_after)
        spectrum_diff = gather_diff.calculate_vertical_velocity_spectrum(velocities)
        spectrum_diff.plot(ax=ax, **{"title": "", **kwargs})
