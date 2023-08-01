"""Implements metrics for quality control of stacking velocities.

These metrics are supposed to be used in :func:`~stacking_velocity_field.StackingVelocityField.qc` method which
iterates over groups of neighboring stacking velocities and automatically provides metrics with all required context
for interactive plotting.

In order to define your own metric you need to inherit a new class from `StackingVelocityMetric` and do the following:
* Set an `is_window_metric` class attribute to `True` or `False` depending on whether your metric needs all stacking
  velocities in a spatial window or only the central one in its `calc` method. In the first case, the central velocity
  will be the first one in the stacked 2d array of velocities.
* Optionally define all other class attributes of `Metric` for future convenience.
* Redefine `calc` method, which must accept two arguments: stacking velocities and times they are estimated for. If
  `is_window_metric` is `False`, stacking velocities will be a 1d array, otherwise it will be a 2d array with shape
  `(n_velocities, n_times)`. Times are always represented as a 1d array. `calc` must return a single metric value.
* Optionally redefine `plot` method which will be used to plot stacking velocities on click on a metric map in
  interactive mode. It should accept an instance of `matplotlib.axes.Axes` to plot on and the same arguments that were
  passed to `calc` during metric calculation. By default it plots all stacking velocities in the window in blue.

If you want the created metric to be calculated by :func:`~stacking_velocity_field.StackingVelocityField.qc` method by
default, it should also be appended to a `VELOCITY_QC_METRICS` list.
"""

import numpy as np
from numba import njit
from matplotlib import patches

from ..metrics import Metric, ScatterMapPlot, MetricMap
from ..utils import calculate_axis_limits, set_ticks, set_text_formatting


class StackingVelocityScatterMapPlot(ScatterMapPlot):
    """Equivalent to `ScatterMapPlot` class except for the `click` method, which also highlights a spatial window in
    which the metric was calculated."""
    def __init__(self, *args, plot_window=True, **kwargs):
        self.plot_window = plot_window
        self.window = None
        super().__init__(*args, **kwargs)

    def click(self, coords):
        """Process the click and highlight a spatial window in which the metric was calculated."""
        coords = super().click(coords)
        if self.window is not None:
            self.window.remove()
        if self.metric_map.metric.is_window_metric and self.plot_window:
            self.window = patches.Circle(coords, self.metric_map.metric.coords_neighbors.radius,
                                         color="blue", alpha=0.3)
            self.main.ax.add_patch(self.window)
        return coords


class StackingVelocityMetricMap(MetricMap):
    """Equivalent to `MetricMap` class except for the interactive scatter plot class, which highlights a spatial window
    in which the metric was calculated on click."""
    interactive_scatter_map_class = StackingVelocityScatterMapPlot


class StackingVelocityMetric(Metric):
    """Base metric class for quality control of stacking velocities."""
    is_window_metric = True
    map_class = StackingVelocityMetricMap
    views = "plot_on_click"

    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.times = None
        self.velocities = None
        self.velocity_limits = None
        self.coords_neighbors = None

    def __str__(self):
        is_window_str = f"Is window metric:          {self.is_window_metric}"
        return self._get_general_info() + "\n" + is_window_str + "\n\n" + self._get_plot_info()

    def __call__(self, velocities, times):
        """Calculate the metric. Selects only central velocity for non-window metrics and redirects the call to a
        static `calc` method which may be njitted."""
        if self.is_window_metric:
            return self.calc(velocities, times)
        return self.calc(velocities[0], times)

    @staticmethod
    def calc(*args, **kwargs):
        """Calculate the metric. Must be overridden in child classes."""
        _ = args, kwargs
        raise NotImplementedError

    def bind_context(self, metric_map, times, velocities, coords_neighbors):
        """Process metric evaluation context: memorize stacking velocities of all items, times for which they were
        calculated and `coords_neighbors` which allows reconstructing windows used for metric calculation."""
        _ = metric_map
        self.times = times
        self.velocities = velocities
        self.velocity_limits = calculate_axis_limits(velocities)
        self.coords_neighbors = coords_neighbors

    def get_window_velocities(self, coords):
        """Return all stacking velocities in a spatial window around given `coords`."""
        _, window_indices = self.coords_neighbors.radius_neighbors([coords], return_distance=True, sort_results=True)
        window_indices = window_indices[0]
        if not self.is_window_metric:
            window_indices = window_indices[0]
        return self.velocities[window_indices]

    @staticmethod
    def plot(ax, window_velocities, times, **kwargs):
        """Plot all stacking velocities in a spatial window."""
        for vel in np.atleast_2d(window_velocities):
            ax.plot(vel, times, color="tab:blue", **kwargs)

    def plot_on_click(self, ax, coords, index, x_ticker=None, y_ticker=None, **kwargs):
        """Plot all stacking velocities used by `calc` during metric calculation."""
        _ = index  # Equals to coords and thus ignored
        window_velocities = self.get_window_velocities(coords)
        (x_ticker, y_ticker), kwargs = set_text_formatting(x_ticker, y_ticker, **kwargs)
        self.plot(ax, window_velocities, self.times, **kwargs)
        ax.invert_yaxis()
        set_ticks(ax, "x", "Stacking velocity, m/s", **x_ticker)
        set_ticks(ax, "y", "Time", **y_ticker)
        ax.set_xlim(*self.velocity_limits)


class HasInversions(StackingVelocityMetric):
    """Check if a stacking velocity decreases at some time."""
    min_value = 0
    max_value = 1
    is_lower_better = True
    is_window_metric = False

    @staticmethod
    @njit(nogil=True)
    def calc(stacking_velocity, times):
        """Return whether the stacking velocity decreases at some time."""
        _ = times
        for cur_vel, next_vel in zip(stacking_velocity[:-1], stacking_velocity[1:]):
            if cur_vel > next_vel:
                return True
        return False

    def plot(self, ax, stacking_velocity, times, **kwargs):
        """Plot the stacking velocity and highlight segments with velocity inversions in red."""
        super().plot(ax, stacking_velocity, times, **kwargs)

        # Highlight sections with velocity inversion
        decreasing_pos = np.where(np.diff(stacking_velocity) < 0)[0]
        if len(decreasing_pos):
            # Process each continuous decreasing section independently
            for section in np.split(decreasing_pos, np.where(np.diff(decreasing_pos) != 1)[0] + 1):
                section_slice = slice(section[0], section[-1] + 2)
                ax.plot(stacking_velocity[section_slice], times[section_slice], color="tab:red", **kwargs)


class MaxAccelerationDeviation(StackingVelocityMetric):
    """Calculate maximal absolute deviation of instantaneous acceleration from the mean acceleration over all times."""
    min_value = 0
    max_value = None
    is_lower_better = None
    is_window_metric = False

    @staticmethod
    @njit(nogil=True)
    def calc(stacking_velocity, times):
        """Return the maximal deviation of instantaneous acceleration from the mean acceleration over all times."""
        mean_acc = (stacking_velocity[-1] - stacking_velocity[0]) / (times[-1] - times[0])
        max_deviation = 0
        for i in range(len(times) - 1):
            instant_acc = (stacking_velocity[i + 1] - stacking_velocity[i]) / (times[i + 1] - times[i])
            deviation = abs(instant_acc - mean_acc)
            if deviation > max_deviation:
                max_deviation = deviation
        return max_deviation

    def plot(self, ax, stacking_velocity, times, **kwargs):
        """Plot the stacking velocity and a mean-acceleration line in dashed red."""
        super().plot(ax, stacking_velocity, times, **kwargs)

        # Plot a mean-acceleration line
        ax.plot([stacking_velocity[0], stacking_velocity[-1]], [times[0], times[-1]], "--", color="tab:red")


class MaxStandardDeviation(StackingVelocityMetric):
    """Calculate maximal spatial velocity standard deviation in a window over all times."""
    min_value = 0
    max_value = None
    is_lower_better = True
    is_window_metric = True

    @staticmethod
    @njit(nogil=True)
    def calc(window, times):
        """Return the maximal spatial velocity standard deviation in a window over all times."""
        _ = times
        if window.shape[0] == 0:
            return 0

        max_std = 0
        for i in range(window.shape[1]):
            current_std = window[:, i].std()
            max_std = max(max_std, current_std)
        return max_std


class MaxRelativeVariation(StackingVelocityMetric):
    """Calculate maximal absolute relative difference between central stacking velocity and the average of all
    remaining velocities in the window over all times."""
    min_value = 0
    max_value = None
    is_lower_better = True
    is_window_metric = True

    @staticmethod
    @njit(nogil=True)
    def calc(window, times):
        """Return the maximal absolute relative difference between central stacking velocity and the average of all
        remaining velocities in the window over all times."""
        _ = times
        if window.shape[0] <= 1:
            return 0

        max_rel_var = 0
        for i in range(window.shape[1]):
            current_rel_var = abs(np.mean(window[1:, i]) - window[0, i]) / window[0, i]
            max_rel_var = max(max_rel_var, current_rel_var)
        return max_rel_var

    def plot(self, ax, window, times, **kwargs):
        """Plot all stacking velocities in spatial window and highlight the central one in red."""
        super().plot(ax, window[1:], times, **kwargs)
        ax.plot(window[0], times, color="tab:red", **kwargs)


VELOCITY_QC_METRICS = [HasInversions, MaxAccelerationDeviation, MaxStandardDeviation, MaxRelativeVariation]
