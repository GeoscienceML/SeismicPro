# pylint: disable=not-an-iterable
"""Implements a utility metric class for headers metric maps construction and a bunch of metrics for survey quality
control.

Quality control metrics are supposed to be used in :func:`~survey.qc` method, which iterates over traces or group of
traces and automatically provides metrics with all required context for interactive plotting.

To define your own metric, you need to inherit a new class from either `TracewiseMetric` or
`MuteTracewiseMetric`, or `BaseWindowRMSMetric` depending on the purpose, and do the following:
* Redefine `get_values` or `numba_get_values` method. Since all metrics are calculated in threads, it is important to
  release GIL as early as possible. Try not to use `get_values` at all, or only use it to calculate the part that can't
  be done without releasing the GIL. The main part should be implemented in `numba_get_values` with njit decorator and
  flag `nogil=True`.
* Redefine `description` property, which returns a string description of the traces the metric is supposed to detected.
* Set a `threshold` class attribute to a number above or below which the trace will be considered bad. If an
  `is_lower_better` class attribute is `True`, the values greater or equal to the `threshold` will be considered bad
  and lower or equal otherwise.
* Optionally redefine `preprocess` method, which accepts the gather and applies any preprocess procedures such as
  muting or scaling. This gather will later be used in `get_values` and displayed on the right side of the interactive
  plot.
* Optionally define all other class attributes of `Metric` for future convenience.
* Optionally redefine `plot` method which will be used to plot gather with tracewise metric on top when a metric map is
  clicked in interactive mode. It should accept an instance of `matplotlib.axes.Axes` to plot on, `coords` and `index`
  for gather that will be plotted, all arguments defined in `self.get_views` and kwargs for gather.plot. By default the
  method plots the gather with tracewise metric values on top. All traces considered as bad by the metric are colored
  red on the gather plot.

If you want the created metric to be calculated by :func:`~survey.qc` method by default, it should also be appended to
a `DEFAULT_TRACEWISE_METRICS` list.
"""

import warnings
from textwrap import wrap
from functools import partial

import numpy as np
from numba import njit
from matplotlib import patches

from ..metrics import Metric


class SurveyAttribute(Metric):
    """A utility metric class that reindexes given survey by `index_cols` and allows for plotting gathers by their
    indices. Does not implement any calculation logic."""
    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.survey = None

    @property
    def header_cols(self):
        """Column names in `survey.headers` for which metric was created."""
        return self.name

    def bind_context(self, metric_map, survey):
        """Process metric evaluation context: memorize the parent survey, reindexed by `index_cols`."""
        self.survey = survey.reindex(metric_map.index_cols)

    def plot(self, ax, coords, index, sort_by=None, **kwargs):
        """Plot a gather by its `index`. Optionally sort it."""
        _ = coords
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        gather.plot(ax=ax, **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        """Return a single view, that plots a gather sorted by `sort_by` by click coordinates."""
        return [partial(self.plot, sort_by=sort_by)], kwargs


class TracewiseMetric(SurveyAttribute):
    """Base class for tracewise metrics with plotters and aggregations. Child classes should redefine `get_values` or
    `numba_get_values` and `description` methods, and optionally `preprocess`."""
    threshold = None
    top_y_ax_scale = "linear"

    def __call__(self, gather):
        """Compute metric value for each trace of the gather by sequentially applying `self.preprocess`,
        `self.get_values` and `self.aggregate` methods."""
        # Ignore warnings related to empty slices or dividing by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gather = self.preprocess(gather)
            values = self.get_values(gather)
            return self.aggregate(values)

    @property
    def description(self):
        """String description of the tracewise metric. Mainly used in `self.describe` when describing the number of bad
        traces detected by the metric."""
        return NotImplementedError

    def describe(self, metric_values, line_width=60):
        """Provide a description about the number of bad values for the passed metric values in a string format. Each
        line in the resulting string will not exceed `line_width`."""
        bin_values = self.binarize(metric_values)
        # Process multiline descriptions and set the distance from the last line to the result to `line_width` symbols
        desc_list = wrap(self.description, width=line_width-1)
        desc_list[-1] = f"{desc_list[-1]+':':<{line_width}}{bin_values.sum()} ({100 * bin_values.mean():.3f}%)"
        return '\n'.join(desc_list)

    def preprocess(self, gather):
        """Preprocess gather before either calling `self.get_values` method to calculate metric or to plot the gather.
        Identity by default."""
        return gather

    def get_values(self, gather):  # get_values, compute_metric
        """Compute QC indicator.

        There are two possible outputs for the provided gather:
            1. Samplewise indicator with the same shape as `gather`.
            2. Tracewise indicator with a shape of (`gather.n_traces`,).

        The method redirects the call to njitted static `numba_get_values` method. Either this method or
        `numba_get_values` must be overridden in child classes.
        """
        return self.numba_get_values(gather.data)

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        """Compute QC indicator."""
        raise NotImplementedError

    def aggregate(self, values):
        """If `values` is 2d array-like, selects the worst values from each row depending on `self.is_lower_better`.
        If `values` is 1d array-like, returns `values` unchanged."""
        if self.is_lower_better is None:
            agg_fn = np.nanmean
        elif self.is_lower_better:
            agg_fn = np.nanmax
        else:
            agg_fn = np.nanmin
        return values if values.ndim == 1 else agg_fn(values, axis=1)

    def binarize(self, values, threshold=None):
        """Binarize given `values` by the provided threshold.

        Parameters
        ----------
        values : 1d ndarray or 2d ndarray
            An array with computed metric values to be converted to a binary mask.
        threshold : int, float, array-like with 2 elements, optional, defaults to None
            A value to use as a threshold for binarizing the input `values`.
            If a single number and `self.is_lower_better` is `True`, `values` greater or equal than the `threshold`
            will be treated as bad and marked as `True`, otherwise, if `self.is_lower_better` is `False`, values lower
            or equal than the `threshold` will be treated as bad and marked as `True`.
            If array-like, two numbers indicate the boundaries within which the metric values are marked as `False`,
            outside inclusive - as `True`.
            If `None`, `self.threshold` will be used.

        Returns
        -------
        bin_mask : 1d ndarray or 2d ndarray
            Binary mask obtained by comparing the `values` with threshold.

        Raises
        ------
        ValueError
            If `threshold` is not provided and `self.threshold` is None.
            If `threshold` is a single number and `self.is_lower_better` is None.
            If `threshold` is iterable but does not contain exactly 2 elements.
        """
        threshold = self.threshold if threshold is None else threshold
        if threshold is None:
            raise ValueError("Either `threshold` or `self.threshold` must be defined")

        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `self.is_lower_better` is None")
            bin_fn = np.greater_equal if self.is_lower_better else np.less_equal
            bin_mask = bin_fn(values, threshold)
        elif len(threshold) != 2:
            msg = f"If `threshold` is array-like, it should contain exactly 2 elements, not {len(threshold)}"
            raise ValueError(msg)
        else:
            bin_mask = (values <= threshold[0]) | (values >= threshold[1])
        bin_mask[np.isnan(values)] = False
        return bin_mask

    def plot(self, ax, coords, index, sort_by=None, threshold=None, top_y_ax_scale=None,  bad_only=False, **kwargs):
        """Plot gather by its `index` with highlighted traces with `bad` metric values based on a `threshold` and a
        `self.is_lower_better`.

        Tracewise metric values will be shown on top of the gather plot. Also, the area with `good` metric values will
        be highlighted in blue.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes of the figure to plot on.
        coords : array-like with 2 elements
            Gather coordinates.
        index : array-like with 2 elements
            Gather index.
        sort_by : str or array-like, optional, defaults to None
            Headers names to sort the gather by.
        threshold : int, float, array-like with 2 elements, optional, defaults to None
            Threshold used to binarize the metric values.
            If None, `self.threshold` will be used. See `self.binarize` for more details.
        top_y_ax_scale : str, optional, defaults to None
            Scale type for top header plot, see `matplotlib.axes.Axes.set_yscale` for available options. Linear
            scaling is applied by default.
        bad_only : bool, optional, defaults to False
            Show only traces that are considered bad based on provided threshold and `self.is_lower_better`.
        kwargs : misc, optional
            Additional keyword arguments to the `gather.plot`.
        """
        _ = coords
        threshold = self.threshold if threshold is None else threshold
        top_y_ax_scale = self.top_y_ax_scale if top_y_ax_scale is None else top_y_ax_scale

        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(sort_by)
        gather = self.preprocess(gather)

        result = self.get_values(gather)
        tracewise_result = self.aggregate(result)
        bin_mask = self.binarize(result, threshold)

        mode = kwargs.pop("mode", "wiggle")
        masks_dict = {"masks": bin_mask, "alpha": 0.8, "label": self.name or "metric", **kwargs.pop("masks", {})}

        if bad_only:
            bad_mask = self.aggregate(bin_mask) == 0
            gather.data[bad_mask] = np.nan
            tracewise_result[bad_mask] = np.nan
            # Don't need to plot 1d `bin_mask` if only bad traces will be plotted.
            if bin_mask.ndim == 1:
                masks_dict = None

        gather.plot(ax=ax, mode=mode, top_header=tracewise_result, masks=masks_dict, **kwargs)
        top_ax = ax.figure.axes[1]
        top_ax.set_yscale(top_y_ax_scale)
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)

    def _plot_threshold(self, ax, threshold):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if isinstance(threshold, (int, float, np.number)):
            if self.is_lower_better is None:
                raise ValueError("`threshold` cannot be single number if `is_lower_better` is None")
            threshold = [threshold, y_max] if self.is_lower_better else [y_min, threshold]
        ax.fill_between(np.arange(x_min, x_max), *threshold, alpha=0.3, color="blue")

    def get_views(self, sort_by=None, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `sort_by` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_y_ax_scale`. The first view
        plots full gather with highlighted bad traces selected based on a `threshold` and a `self.is_lower_better`.
        The second view displays only bad traces."""
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class DeadTrace(TracewiseMetric):
    """Detect constant traces.

    `get_values` returns 1d binary mask where each constant trace is marked with one.
    """
    min_value = 0
    max_value = 1
    is_lower_better = True
    threshold = 0.5

    @property
    def description(self):
        """String description of the tracewise metric."""
        return "Number of dead traces"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.isclose(max(traces[i]), min(traces[i]))
        return res

    def get_views(self, sort_by=None, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return single view, that plots a gather sorted by `sort_by` with a metric values shown on top of the gather
        plot and highlighted bad traces selected based on a `threshold` and a `self.is_lower_better`. The y-axis of
        the metric plot is scaled by `top_y_ax_scale`."""
        plot_kwargs = {"sort_by": sort_by, "threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs)], kwargs


class TraceAbsMean(TracewiseMetric):
    """Calculate absolute value of the trace's mean divided by trace's std.

    `get_values` returns 1d array with computed metric values for the gather.
    """
    is_lower_better = True
    threshold = 0.1

    @property
    def description(self):
        """String description of the tracewise metric."""
        return f"Traces with mean divided by std greater than {self.threshold}"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.abs(traces[i].mean() / (traces[i].std() + 1e-10))
        return res


class TraceMaxAbs(TracewiseMetric):
    """Find a maximum absolute amplitude value divided by trace's std.

    `get_values` returns 1d array with computed metric values for the gather.
    """
    is_lower_better = True
    threshold = 15

    @property
    def description(self):
        """String description of tracewise metric"""
        return f"Traces with max abs to std ratio greater than {self.threshold}"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            res[i] = np.max(np.abs(traces[i])) / (traces[i].std() + 1e-10)
        return res


class MaxClipsLen(TracewiseMetric):
    """Calculate the maximum number of clipped samples in a row for each trace.

    `get_values` returns a 2d array matching the shape of the input gather. It's values are either 0 for samples that
    are not clipped or a positive integer representing the length of the clipped sequence for a particular sample
    otherwise.
    """
    min_value = 1
    max_value = None
    is_lower_better = True
    threshold = 3

    @property
    def description(self):
        """String description of tracewise metric."""
        return f"Traces with more than {self.threshold} clipped samples in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        def _update_counters(sample, value, counter, container):
            if np.isclose(sample, value):
                counter += 1
            else:
                if counter > 1:
                    container[:] = counter
                counter = 0
            return counter

        maxes = np.zeros_like(traces)
        mins = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            trace = traces[i]
            max_val = max(trace)
            max_counter = 0
            min_val = min(trace)
            min_counter = 0
            for j in range(trace.shape[0]):  # pylint: disable=consider-using-enumerate
                max_counter = _update_counters(trace[j], max_val, max_counter, maxes[i, j - max_counter: j])
                min_counter = _update_counters(trace[j], min_val, min_counter, mins[i, j - min_counter: j])

            if max_counter > 1:
                maxes[i, -max_counter:] = max_counter
            if min_counter > 1:
                mins[i, -min_counter:] = min_counter
        return maxes + mins


class MaxConstLen(TracewiseMetric):
    """Calculate the maximum number of identical values in a row for each trace.

    `get_values` returns a 2d array matching the shape of the input gather. Each its value is the length of a sequence
    of identical values around each particular sample.
    """
    is_lower_better = True
    threshold = 4

    @property
    def description(self):
        """String description of tracewise metric"""
        return f"Traces with more than {self.threshold} identical values in a row"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        indicator = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            trace = traces[i]
            counter = 1
            for j in range(1, trace.shape[0]):  # pylint: disable=consider-using-enumerate
                if np.isclose(trace[j], trace[j - 1]):
                    counter += 1
                else:
                    if counter > 1:
                        indicator[i, j - counter: j] = counter
                    counter = 1

            if counter > 1:
                indicator[i, -counter:] = counter
        return indicator


class MuteTracewiseMetric(TracewiseMetric):  # pylint: disable=abstract-method
    """Base class for tracewise metric with implemented `self.preprocess` method which applies muting and standard
    scaling to the input gather. Child classes should redefine `get_values` or `numba_get_values` methods.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to None
        A metric name.
    """
    def __init__(self, muter, name=None):
        super().__init__(name=name)
        self.muter = muter

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', muter='{self.muter}')"

    def preprocess(self, gather):
        """Apply muting with np.nan as a fill value and standard scaling to provided gather."""
        return gather.copy().mute(muter=self.muter, fill_value=np.nan).scale_standard()


class Spikes(MuteTracewiseMetric):
    """Detect spikes: drastic and abnormal changes in trace amplitudes.

    `get_values` returns a 2d array matching the shape of the input gather. Each its value is the absolute difference
    between the corresponding trace sample and the rolling trace mean in the 3 sample window.

    The metric is highly dependent on a muter being used; if the muter is not strong enough, the metric will overreact
    to the first breaks.
    """
    min_value = 0
    max_value = None
    is_lower_better = True
    threshold = 2

    @property
    def description(self):
        """String description of tracewise metric."""
        return "Traces with spikes"

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces):
        res = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            nan_indices = np.nonzero(np.isnan(traces[i]))[0]
            j = nan_indices[-1] + 1 if len(nan_indices) > 0 else 0
            res[i, j+1: -1] = np.abs(traces[i, j+2:] + traces[i, j: -2] - 2*traces[i, j+1 :-1]) / 3
        return res


class Autocorrelation(MuteTracewiseMetric):
    """Compute the lag-1 autocorrelation for each trace.

    `get_values` returns a 1d array with autocorrelation value for each trace if the fraction of `nan`s in it is no
    greater than `max_nan_fraction` or `np.nan` otherwise.

    The metric is highly dependent on a muter being used; if the muter is not strong enough, the metric will overreact
    to the first breaks.

    Parameters
    ----------
    muter : Muter
        A muter to use.
    name : str, optional, defaults to "Autocorrelation"
        A metric name.
    max_nan_fraction : float, optional, defaults to 0.95
        The maximum proportion of nan values allowed in a trace.
    """
    min_value = -1
    max_value = 1
    is_lower_better = False
    threshold = 0.8

    def __init__(self, muter, name=None, max_nan_fraction=0.95):
        super().__init__(muter=muter, name=name)
        self.max_nan_fraction = max_nan_fraction

    def __repr__(self):
        """String representation of the metric."""
        kwargs_str = f"(name='{self.name}', muter='{self.muter}', max_nan_fraction='{self.max_nan_fraction}')"
        return f"{type(self).__name__}" + kwargs_str

    @property
    def description(self):
        """String description of tracewise metric."""
        return f"Traces with autocorrelation less than {self.threshold}"

    def get_values(self, gather):
        return self.numba_get_values(gather.data, max_nan_fraction=self.max_nan_fraction)

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces, max_nan_fraction):
        res = np.empty_like(traces[:, 0])
        for i in range(traces.shape[0]):
            if np.isnan(traces[i]).mean() > max_nan_fraction:
                res[i] = np.nan
            else:
                res[i] = np.nanmean(traces[i, 1:] * traces[i, :-1])
        return res


class BaseWindowRMSMetric(TracewiseMetric):  # pylint: disable=abstract-method
    """Base class for a tracewise metric that computes RMS amplitude in a window for each trace.

    Child classes should redefine `get_values` or `numba_get_values` methods.
    """

    def __call__(self, gather, return_rms=True):
        """Compute the metric by applying `self.preprocess` and `self.get_values` to the provided gather.
        If `return_rms` is True, the RMS value for the provided gather will be returned.
        Otherwise, two 1d arrays will be returned:
            1. Sum of squares of amplitudes in the defined window for each trace,
            2. Number of amplitudes in a specified window for each trace.
        """
        # Ignore warnings related to empty slices or dividing by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gather = self.preprocess(gather)
            squares, nums = self.get_values(gather)
        if return_rms:
            return self.compute_rms(squares, nums)
        return squares, nums

    def _get_threshold_description(self):
        """String description of the threshold for Window RMS metrics."""
        if isinstance(self.threshold, (int, float, np.number)):
            msg = ("greater then or equal to" if self.is_lower_better else "less then or equal to") +\
                   f" {self.threshold}"
            return msg
        return f"between {self.threshold[0]} and {self.threshold[1]}"

    @property
    def header_cols(self):
        """Column names in `survey.headers` to store the metrics results in."""
        return [self.name+"_sum", self.name+"_n"]

    @staticmethod
    def compute_rms(squares, nums):
        """Compute the RMS using provided squares of amplitudes and the number of amplitudes used for square
        calculation."""
        return np.sqrt(np.sum(squares) / np.sum(nums))

    @staticmethod
    @njit(nogil=True)
    def compute_stats_by_ixs(data, start_ixs, end_ixs):
        """Compute the sum of squares and the number of elements in a window specified by `start_ixs` and `end_ixs`
        for each trace in the provided data."""
        sum_squares = np.empty_like(data[:, 0])
        nums = np.empty_like(data[:, 0])

        for i, (trace, start_ix, end_ix) in enumerate(zip(data, start_ixs, end_ixs)):
            sum_squares[i] = sum(trace[start_ix: end_ix] ** 2)
            nums[i] = len(trace[start_ix: end_ix])
        return sum_squares, nums

    def construct_map(self, coords, values, *, coords_cols=None, index=None, index_cols=None, agg=None, bin_size=None,
                      calculate_immediately=True, tracewise=False):
        """Construct a metric map with computed RMS values for gathers indexed by `index`.

        There are two available options for RMS computation controlled by `tracewise` flag:
        1. If `tracewise` is False (default behavior), the RMS is computed gatherwise,
        2. If `tracewise` is True, the RMS is computed tracewise and then aggregated by gathers.
        """
        if tracewise:
            return super().construct_map(coords, np.sqrt(values.iloc[:, 0] / values.iloc[:, 1]),
                                         coords_cols=coords_cols, index=index, index_cols=index_cols, agg=agg,
                                         bin_size=bin_size, calculate_immediately=calculate_immediately)

        sum_square_map = super().construct_map(coords, values.iloc[:, 0], coords_cols=coords_cols, index=index,
                                               index_cols=index_cols, agg="sum")
        nums_map = super().construct_map(coords, values.iloc[:, 1], coords_cols=coords_cols, index=index,
                                         index_cols=index_cols, agg="sum")
        sum_square_df = sum_square_map.index_data[[*sum_square_map.index_cols, sum_square_map.metric_name]]
        sum_df = sum_square_df.merge(nums_map.index_data, on=nums_map.index_cols)
        sum_df[self.name] = np.sqrt(sum_df[self.name+"_x"] / sum_df[self.name+"_y"])
        return super().construct_map(sum_df[nums_map.coords_cols], sum_df[self.name],
                                     index=sum_df[nums_map.index_cols], agg=agg, bin_size=bin_size,
                                     calculate_immediately=calculate_immediately)

    def plot(self, ax, coords, index, threshold=None, top_y_ax_scale=None, bad_only=False, color="lime", **kwargs):  # pylint: disable=arguments-renamed
        """Plot the gather sorted by offset with tracewise indicator on the top of the gather plot. Any metric related
        information might be displayed over the gather plot using `self.add_window_on_plot`."""
        threshold = self.threshold if threshold is None else threshold
        top_y_ax_scale = self.top_y_ax_scale if top_y_ax_scale is None else top_y_ax_scale
        _ = coords
        gather = self.survey.get_gather(index).sort("offset")
        squares, nums = self(gather, return_rms=False)
        tracewise_metric = np.sqrt(squares / nums)
        tracewise_metric[tracewise_metric == 0] = np.nan
        if bad_only:
            bin_mask = self.binarize(tracewise_metric, threshold)
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        gather.plot(ax=ax, top_header=tracewise_metric, **kwargs)
        top_ax = ax.figure.axes[1]
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)
        top_ax.set_yscale(top_y_ax_scale)
        self.add_window_on_plot(ax=ax, gather=gather, color=color)
        if len(ax.get_legend_handles_labels()[0]):
            # Define legend position to speed up plotting for huge gathers
            ax.legend(loc='upper right')

    def add_window_on_plot(self, ax, gather, color=None):
        """Plot any additional metric related graphs over the gather plot."""
        _ = self, ax, gather, color

    def get_views(self, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `offset` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_y_ax_scale`. The first view
        plots full gather with highlighted `bad` traces selected based on a `threshold` and a `self.is_lower_better`.
        The second view displays only `bad` traces."""
        plot_kwargs = {"threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


class WindowRMS(BaseWindowRMSMetric):
    """Compute traces RMS in a rectangular window defined by provided ranges of offsets and times.

    Parameters
    ----------
    offsets : array-like with 2 ints
        Offset range to use for calculation, measured in meters.
    times : array-like with 2 ints
        Time range to use for calculation, measured in ms.
    name : str, optional, defaults to "WindowRMS"
        A metric name.
    """
    is_lower_better = None
    threshold = None

    def __init__(self, offsets, times, name=None):
        if len(offsets) != 2:
            raise ValueError(f"`offsets` must contain 2 elements, not {len(offsets)}")

        if len(times) != 2:
            raise ValueError(f"`times` must contain 2 elements, not {len(times)}")

        super().__init__(name=name)
        self.offsets = np.array(offsets)
        self.times = np.array(times)

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(name='{self.name}', offsets='{self.offsets}', times='{self.times}')"

    @property
    def description(self):
        """String description of the tracewise metric."""
        msg = f"Traces with RMS {self._get_threshold_description()} computed in a window within a range "\
              f"{self.offsets}m in offsets and {self.times}ms in times"
        return msg

    def describe(self, metric_values, line_width=60):
        """Provide a description about the number of bad values for the passed metric values in a string format. Each
        line in the resulting string will not exceed `line_width`. If `self.threshold` is None, the average RMS for
        the passed values will be showed."""
        metric_value = np.sqrt(metric_values[:, 0] / metric_values[:, 1])
        if self.threshold is None:
            description = "Mean traces RMS computed in a window within a range \n"
            description += f"{f'{self.offsets}m in offsets and {self.times}ms in times:':<{line_width}}"
            return description + f"{np.nanmean(metric_value):<.3f}"
        return super().describe(metric_value, line_width=line_width)

    def get_values(self, gather):
        times_ixs = self._get_time_ixs(gather)
        return self.numba_get_values(traces=gather.data, times_ixs=times_ixs, gather_offsets=gather.offsets,
                                     metric_offsets=self.offsets, compute_stats_by_ixs=self.compute_stats_by_ixs)

    def _get_time_ixs(self, gather):
        times_ixs = np.clip(gather.times_to_indices(self.times, round=True), 0, gather.n_times - 1)
        # Include the next index to mimic the behavior of conventional software
        times_ixs[1] += 1
        return times_ixs

    @staticmethod
    @njit(nogil=True)
    def numba_get_values(traces, times_ixs, gather_offsets, metric_offsets, compute_stats_by_ixs):
        window_ixs = (gather_offsets >= metric_offsets[0]) & (gather_offsets <= metric_offsets[1])
        start_ixs = np.full(sum(window_ixs), fill_value=times_ixs[0], dtype=np.int16)
        end_ixs = np.full(sum(window_ixs), fill_value=times_ixs[1], dtype=np.int16)
        squares = np.zeros_like(traces[:, 0])
        nums = np.zeros_like(traces[:, 0])
        window_squares, window_nums = compute_stats_by_ixs(traces[window_ixs], start_ixs=start_ixs, end_ixs=end_ixs)
        squares[window_ixs] = window_squares
        nums[window_ixs] = window_nums
        return squares, nums

    def add_window_on_plot(self, ax, gather, color="lime", legend=None):
        """Plot a rectangle path over the gather plot in a place where RMS was computed."""
        times_ixs = self._get_time_ixs(gather)
        offs_ind = np.nonzero((gather.offsets >= self.offsets[0]) & (gather.offsets <= self.offsets[1]))[0]
        if len(offs_ind) > 0:
            n_rec = (offs_ind[0], times_ixs[0]), len(offs_ind), (times_ixs[1] - times_ixs[0])
            ax.add_patch(patches.Rectangle(*n_rec, linewidth=2, color=color, label=legend, alpha=0.5))


class AdaptiveWindowRMS(BaseWindowRMSMetric):
    """Compute traces RMS in sliding window along a provided refractor velocity.

    For each gather, the RMS value will be calculated within a window of size `window_size` centered around the
    refractor velocity shifted by `shift`.

    Only traces that contain at least one sample in the provided window are considered, otherwise the metric is nan.

    Parameters
    ----------
    window_size : int
        Length of the window for computing RMS amplitudes, measured in ms.
    shift : int
        The distance to shift the sliding window from the refractor velocity, measured in ms.
    refractor_velocity: RefractorVelocity
        Refractor velocity object to find times along witch the RMS will be calculated.
    name : str, optional, defaults to "AdaptiveWindowRMS"
        A metric name.
    """
    is_lower_better = False
    threshold = None

    def __init__(self, window_size, shift, refractor_velocity, name=None):
        super().__init__(name=name)
        self.window_size = window_size
        self.shift = shift
        self.refractor_velocity = refractor_velocity

    def __repr__(self):
        """String representation of the metric."""
        repr_str = f"(name='{self.name}', window_size='{self.window_size}', shift='{self.shift}', "\
                   f"refractor_velocity='{self.refractor_velocity}')"
        return f"{type(self).__name__}" + repr_str

    @property
    def description(self):
        """String description of the tracewise metric."""
        direction = "below" if self.shift > 0 else "above"
        msg = f"Traces with RMS {self._get_threshold_description()} computed in a {self.window_size} ms window around"\
              f" RV shifted {direction} by {abs(self.shift)} ms "
        return msg + self._get_threshold_description()

    def describe(self, metric_values, line_width=60):
        """Provide a description about the number of bad values for the passed metric values in a string format. Each
        line in the resulting string will not exceed `line_width`. If `self.threshold` is None, the average RMS for
        the passed values will be showed."""
        metric_value = np.sqrt(metric_values[:, 0] / metric_values[:, 1])
        if self.threshold is None:
            direction =  "below" if self.shift > 0 else "above"
            description = f"Mean of traces RMS computed in a {self.window_size} ms window \n"
            description += f"{f'around RV shifted {direction} by {abs(self.shift)} ms:':<{line_width}}"
            return description + f"{np.nanmean(metric_value):<.3f}"
        return super().describe(metric_value, line_width=line_width)

    def get_values(self, gather):
        fbp_times = self.refractor_velocity(gather.offsets)
        kwargs = {"traces": gather.data, "window_size": self.window_size, "shift": self.shift,
                  "samples": gather.samples, "delay": gather.delay, "sample_interval": gather.sample_interval,
                  "fbp_times": fbp_times, "_get_indices": self._get_indices,
                  "compute_stats_by_ixs": self.compute_stats_by_ixs}
        return self.numba_get_values(**kwargs)

    @staticmethod
    @njit(nogil=True)
    # pylint: disable-next=redefined-outer-name
    def numba_get_values(traces, window_size, shift, samples, delay, sample_interval, fbp_times, _get_indices,
                         compute_stats_by_ixs):
        start_ixs, end_ixs = _get_indices(window_size=window_size, shift=shift, samples=samples, delay=delay,
                                          sample_interval=sample_interval, fbp_times=fbp_times)
        return compute_stats_by_ixs(traces, start_ixs=start_ixs, end_ixs=end_ixs)

    @staticmethod
    @njit(nogil=True)
    # pylint: disable-next=redefined-outer-name
    def _get_indices(window_size, shift, samples, delay, sample_interval, fbp_times):
        """Calculates the start and end indices of a window of size `window_size` centered around the refractor
        velocity shifted by `shift`."""
        mid_times = fbp_times + shift
        n_samples = len(samples)

        start_times = mid_times - (window_size - window_size // 2)
        start_ixs = (start_times - delay) / sample_interval
        start_ixs = np.clip(np.rint(start_ixs), 0, n_samples - 1).astype(np.int32)

        end_times = mid_times + (window_size // 2)
        end_ixs = (end_times - delay) / sample_interval
        end_ixs = np.clip(np.rint(end_ixs), 0, n_samples - 1).astype(np.int32)
        return start_ixs, end_ixs

    def add_window_on_plot(self, ax, gather, color="lime", legend=None):
        """Plot two parallel lines over the gather plot along the window where RMS was computed."""
        fbp_times = self.refractor_velocity(gather.offsets)
        indices = self._get_indices(window_size=self.window_size, shift=self.shift, samples=gather.samples,
                                    delay=gather.delay, sample_interval=gather.sample_interval, fbp_times=fbp_times)
        indices = np.where(indices[0] == indices[1], np.nan, indices)
        ax.fill_between(np.arange(gather.n_traces), indices[0], indices[1], color=color, label=legend, alpha=0.5)


class MetricsRatio(TracewiseMetric):  # pylint: disable=abstract-method
    """Calculate the ratio of two window RMS metrics.

    By default, the RMS amplitudes for `self.numerator` and `self.denominator` are first calculated for each gather and
    then used to calculate the ratio. If `tracewise` flag in :func:`MetricsRatio.construct_map` set to `True`, the RMS
    ratio of `self.numerator` and `self.denominator` is calculated independently for each trace and then aggregated by
    gathers.

    Parameters
    ----------
    numerator : BaseWindowRMSMetric or its subclass
        Metric instance whose values will be divided by the `denominator` metric values.
    denominator : BaseWindowRMSMetric or its subclass
        Metric instance whose values will be used as a divisor for a `numerator` metric.
    name : str, optional, defaults to "numerator.name to denominator.name ratio"
        A metric name.
    """
    is_lower_better = False
    threshold = None

    def __init__(self, numerator, denominator, name=None):
        for metric in [numerator, denominator]:
            if not isinstance(metric, BaseWindowRMSMetric):
                msg = f"Metric ratio can be computed only for BaseWindowRMSMetric instances or its subclasses, but \
                       given metric has type: {type(metric)}."
                raise ValueError(msg)

        name = f"{numerator.name} to {denominator.name} ratio" if name is None else name
        super().__init__(name=name)

        self.numerator = numerator
        self.denominator = denominator

    def __repr__(self):
        """String representation of the metric."""
        kwargs_str = f"(name='{self.name}', numerator='{self.numerator}', denominator='{self.denominator}')"
        return f"{type(self).__name__}" + kwargs_str

    @property
    def header_cols(self):
        """Column names in `survey.headers` to store the metrics results in."""
        return self.numerator.header_cols + self.denominator.header_cols

    def construct_map(self, coords, values, *, coords_cols=None, index=None, index_cols=None, agg=None, bin_size=None,
                      calculate_immediately=True, tracewise=False):
        """Construct a metric map with `self.numerator` and `self.denominator` ratio for gathers indexed by `index`.

        There are two available options for ratio computation controlled by `tracewise` flag:
        1. If `tracewise` is False (default behavior), the RMS amplitudes for `self.numerator` and `self.denominator`
        are first calculated for each gather and then used to calculate the ratio,
        2. If `tracewise` is True, the ratio of `self.numerator` and `self.denominator` computed independently for each
        trace and then aggregated by gathers.
        """
        if tracewise:
            numerator_values = values[self.numerator.header_cols].to_numpy()
            denominator_values = values[self.denominator.header_cols].to_numpy()
            numerator_rms = np.sqrt(numerator_values[:, 0] / numerator_values[:, 1])
            denominator_rms = np.sqrt(denominator_values[:, 0] / denominator_values[:, 1])
            rms = numerator_rms / denominator_rms
            if np.isnan(rms).sum() == len(rms):
                msg = f"The ratio of `{self.numerator.name}` and `{self.denominator.name}` cannot be computed since"\
                       " they were calculated in disjoint windows"
                raise ValueError(msg)
            return super().construct_map(coords, rms, coords_cols=coords_cols, index=index, index_cols=index_cols,
                                         agg=agg, bin_size=bin_size, calculate_immediately=calculate_immediately)

        mmaps_1 = self.numerator.construct_map(coords, values[self.numerator.header_cols], coords_cols=coords_cols,
                                               index=index, index_cols=index_cols, agg=agg)
        mmaps_2 = self.denominator.construct_map(coords, values[self.denominator.header_cols], coords_cols=coords_cols,
                                                 index=index, index_cols=index_cols, agg=agg)
        numerator_df = mmaps_1.index_data[[*mmaps_1.index_cols, mmaps_1.metric_name]]
        ratio_df = numerator_df.merge(mmaps_2.index_data, on=mmaps_2.index_cols)
        ratio_df[self.name] = ratio_df[self.numerator.name] / ratio_df[self.denominator.name]
        coords = ratio_df[mmaps_1.coords_cols]
        values = ratio_df[self.name]
        index = ratio_df[mmaps_1.index_cols]
        return super().construct_map(coords, values, index=index, agg=agg, bin_size=bin_size,
                                     calculate_immediately=calculate_immediately)

    def plot(self, ax, coords, index, threshold=None, top_y_ax_scale=None, bad_only=False, **kwargs):
        """Plot the gather sorted by offset with two rectangles over the gather plot. The lime-colored rectangle
        represents the window where the `self.numerator` metric was calculated, while the magenta-colored rectangle
        represents the `self.denominator` window. Additionally, tracewise ratio of `self.numerator` by
        `self.denominator` is displayed on the top of the gather plot."""
        threshold = self.threshold if threshold is None else threshold
        top_y_ax_scale = self.top_y_ax_scale if top_y_ax_scale is None else top_y_ax_scale
        _ = coords
        gather = self.survey.get_gather(index).sort("offset")

        squares_numerator, nums_numerator = self.numerator(gather, return_rms=False)
        squares_denominator, nums_denominator = self.denominator(gather, return_rms=False)

        tracewise_numerator = np.sqrt(squares_numerator / nums_numerator)
        tracewise_denominator = np.sqrt(squares_denominator / nums_denominator)
        tracewise_metric = tracewise_numerator / tracewise_denominator
        tracewise_metric[tracewise_metric == 0] = np.nan

        if bad_only:
            bin_mask = self.binarize(tracewise_metric, threshold)
            gather.data[self.aggregate(bin_mask) == 0] = np.nan

        gather.plot(ax=ax, top_header=tracewise_metric, **kwargs)
        top_ax = ax.figure.axes[1]
        if threshold is not None:
            self._plot_threshold(ax=top_ax, threshold=threshold)
        top_ax.set_yscale(top_y_ax_scale)

        self.numerator.add_window_on_plot(ax=ax, gather=gather, color="lime", legend=f"{self.numerator.name} window")
        self.denominator.add_window_on_plot(ax=ax, gather=gather, color="magenta",
                                            legend=f"{self.denominator.name} window")
        if len(ax.get_legend_handles_labels()[0]):
            # Define legend position to speed up plotting for huge gathers
            ax.legend(loc='upper right')

    def get_views(self, threshold=None, top_y_ax_scale=None, **kwargs):
        """Return two plotters of the metric views. Each view plots a gather sorted by `offset` with a metric values
        shown on top of the gather plot. The y-axis of the metric plot is scaled by `top_y_ax_scale`. The first view
        plots full gather with highlighted `bad` traces selected based on a `threshold` and a `self.is_lower_better`.
        The second view displays only `bad` traces."""
        plot_kwargs = {"threshold": threshold, "top_y_ax_scale": top_y_ax_scale}
        return [partial(self.plot, **plot_kwargs), partial(self.plot, bad_only=True, **plot_kwargs)], kwargs


DEFAULT_TRACEWISE_METRICS = [DeadTrace, TraceAbsMean, TraceMaxAbs, MaxClipsLen, MaxConstLen]
