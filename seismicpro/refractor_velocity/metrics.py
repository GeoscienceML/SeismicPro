"""Implements metrics for quality control of first breaks given the near-surface velocity model."""

from math import ceil, floor
from functools import partial, lru_cache

import numpy as np
from numba import njit

from ..gather.utils.normalization import scale_maxabs, get_tracewise_quantile
from ..gather.utils.correction import get_hodograph
from ..metrics import Metric
from ..utils import get_first_defined, set_ticks, GEOGRAPHIC_COORDS, get_coords_cols


class RefractorVelocityMetric(Metric):
    """Base metric class for quality control of refractor velocity field.
    Implements the following logic: `calc` method returns iterable of tracewise metric values,
    then `__call__` is used for gatherwise metric aggregation.
    Default views are `plot_refractor_velocity` utilizing `RefractorVelocity.plot` and `plot_gather`,
    which is parametrized for plotting metric values on top and above the `Gather.plot`.
    Parameters needed for metric calculation and view plotting should be set as attributes, e.g. `first_breaks_header`.

    Parameters
    ----------
    first_breaks_header : str, optional
        Column name from `survey.headers` where times of first break are stored.
        If not provided, must be set before the metric call via `set_defaults`.
    correct_uphole : bool, optional, defaults to None
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
        emulating the case when sources are located on the surface.
        If not provided, must be set before the metric call via `set_defaults`.
    name : str, optional
        Metric name, overrides default name if given.
    """

    views = ("plot_gather", "plot_refractor_velocity")

    def __init__(self, first_breaks_header=None, correct_uphole=None, name=None):
        super().__init__(name=name)
        self.survey = None
        self.field = None
        self.max_offset = None
        self.is_geographic = None
        self.first_breaks_header = first_breaks_header
        self.correct_uphole = correct_uphole

    def set_defaults(self, **kwargs):
        """Set parameters needed for metric calculation as attributes.
        Does not update attributes that have already been set to anything but None.
        
        Parameters
        ----------
        kwargs : misc
            Dict with attributes to set.
        """
        for attr, attr_value in kwargs.items():
            if getattr(self, attr, None) is None:
                setattr(self, attr, attr_value)

    def bind_context(self, metric_map, survey, field):
        """Process interactive plot evaluation context."""
        _ = metric_map
        self.survey = survey
        self.field = field
        self.max_offset = survey["offset"].max()
        self.is_geographic = get_coords_cols(survey.indexed_by) in GEOGRAPHIC_COORDS

    def calc(self, gather, refractor_velocity):
        """Calculate the metric. Must be overridden in child classes."""
        _ = gather, refractor_velocity
        raise NotImplementedError

    def __call__(self, gather, refractor_velocity):
        """Aggregate the metric. If not overriden, takes mean value of `calc`."""
        return np.mean(self.calc(gather, refractor_velocity))

    def get_views(self, sort_by=None, threshold=None, **kwargs):
        """Return plotters of the metric views and add kwargs for `plot_gather` for interactive map plotting."""
        views_list = [partial(self.plot_gather, sort_by=sort_by, threshold=threshold), self.plot_refractor_velocity]
        return views_list, kwargs

    def binarize(self, gather, metric_values, threshold=None):
        """Get binarized mask from metric_values."""
        _ = gather
        # Also handles self.is_lower_better=None case
        invert_mask = -1 if self.is_lower_better is False else 1
        mask_threshold = get_first_defined(threshold,
                                           self.vmax if invert_mask == 1 else self.vmin,
                                           metric_values.mean())
        return {"masks": metric_values * invert_mask, "threshold": mask_threshold * invert_mask}

    @staticmethod
    @njit(nogil=True)
    def _make_windows(times, data, offsets, n_samples, sample_interval, delay):
        res = np.empty((data.shape[0], n_samples), dtype=data.dtype)
        for i in range(n_samples):
            dt = (-n_samples / 2 + i) * sample_interval
            res[:, i] = get_hodograph(data, offsets, sample_interval, delay, times + dt, fill_value=0)
        return res

    def plot_gather(self, coords, ax, index, sort_by=None, threshold=None, mask=True, top_header=True, **kwargs):
        """Base view for gather plotting. Plot the gather by its index in bounded survey and its first breaks.
        By default also recalculates metric in order to display `top_header` with metric values above gather traces
        and mask traces according to threshold. Threshold is either acquired from `threshold` argument if given,
        metric's colorbar margin if defined, or simply by mean metric value.

        Parameters
        ----------
        sort_by : str or iterable of str, optional
            Headers names to sort the gather by.
        threshold : int or float, optional
            A value to use as a threshold for binarizing the metric values. If `self.is_lower_better` is `True`,
            metric values greater or equal than the `threshold` will be treated as bad and vice versa.
        mask : bool, optional, defaults to True
            Whether to mask traces according to `threshold` on top of the gather plot.
        top_header : bool, optional, defaults to True
            Whether to show a header with metric values above the gather plot.
        kwargs : misc, optional
            Additional keyword arguments to `gather.plot`
        """
        _ = coords
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        plotting_kwargs = {"mode": "wiggle", "event_headers": self.first_breaks_header}
        if top_header or mask:
            refractor_velocity = self.field(gather.coords, is_geographic=self.is_geographic)
            metric_values = self.calc(gather=gather, refractor_velocity=refractor_velocity)
            if mask:
                plotting_kwargs['masks'] = self.binarize(gather, metric_values, threshold)
            if top_header:
                plotting_kwargs["top_header"] = metric_values
        plotting_kwargs.update(kwargs)
        gather.plot(ax=ax, **plotting_kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        """Plot the refractor velocity curve."""
        refractor_velocity = self.field(coords, is_geographic=self.is_geographic)
        gather = self.survey.get_gather(index)
        refractor_velocity.times = gather[self.first_breaks_header]
        if self.correct_uphole:
            refractor_velocity.times = refractor_velocity.times + gather["SourceUpholeTime"]
        refractor_velocity.offsets = gather["offset"]
        refractor_velocity.plot(ax=ax, max_offset=self.max_offset, **kwargs)


class FirstBreaksOutliers(RefractorVelocityMetric):
    """The first break outliers metric.
    A first break time is considered to be an outlier if it differs from the expected arrival time defined by
    an offset-traveltime curve by more than a given threshold. Evaluates the fraction of outliers in the gather.

    Parameters
    ----------
    threshold_times : float, optional, defaults to 50
        Threshold for the first breaks outliers metric calculation. Measured in milliseconds.
    first_breaks_header : str, optional, defaults to None
        Column name from `survey.headers` where times of first break are stored.
        If not provided, must be set before the metric call via `set_defaults`.
    correct_uphole : bool, optional, defaults to None
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
        emulating the case when sources are located on the surface.
        If not provided, must be set before the metric call via `set_defaults`.
    name : str, optional
        Metric name, overrides default name if given.
    """

    vmin = 0
    vmax = 0.05
    is_lower_better = True

    def __init__(self, threshold_times=50, first_breaks_header=None, correct_uphole=None, name=None):
        super().__init__(first_breaks_header, correct_uphole, name)
        self.threshold_times = threshold_times

    @staticmethod
    @njit(nogil=True)
    def _calc(rv_times, picking_times, threshold_times):
        """Calculate the first break outliers."""
        return np.abs(rv_times - picking_times) > threshold_times

    def calc(self, gather, refractor_velocity):
        """Calculate the first break outliers.
        Returns whether first break of each trace in the gather differs from those estimated by
        a near-surface velocity model by more than `threshold_times`.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity
            Near-surface velocity model to estimate the expected first break times at `gather` offsets.

        Returns
        -------
        metric : np.ndarray of bool
            Array indicating whether each trace in the gather represents an outlier.
        """
        rv_times = refractor_velocity(gather["offset"])
        picking_times = gather[self.first_breaks_header]
        if self.correct_uphole:
            picking_times = picking_times + gather["SourceUpholeTime"]
        return self._calc(rv_times, picking_times, self.threshold_times)

    def plot_gather(self, *args, **kwargs):
        """Plot the gather with highlighted outliers on top of the gather plot."""
        kwargs["top_header"] = kwargs.pop("top_header", False)
        super().plot_gather(*args, **kwargs)

    def plot_refractor_velocity(self, *args, **kwargs):
        """Plot the refractor velocity curve and show the threshold area used for metric calculation."""
        kwargs["threshold_times"] = kwargs.pop("threshold_times", self.threshold_times)
        return super().plot_refractor_velocity(*args, **kwargs)


class FirstBreaksAmplitudes(RefractorVelocityMetric):
    """Mean amplitude of the signal in the moment of first break after maxabs scaling.
    
    Parameters
    ----------
    first_breaks_header : str, optional, defaults to None
        Column name from `survey.headers` where times of first break are stored.
        If not provided, must be set before the metric call via `set_defaults`.
    correct_uphole : bool, optional, defaults to None
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
        emulating the case when sources are located on the surface.
        If not provided, must be set before the metric call via `set_defaults`.
    name : str, optional
        Metric name, overrides default name if given.
    """

    is_lower_better = None

    @staticmethod
    @njit(nogil=True)
    def _calc(gather_data, offsets, sample_interval, delay, picking_times):
        """Calculate signal amplitudes at first break times."""
        min_value, max_value = get_tracewise_quantile(gather_data, np.array([0, 1]))
        min_value, max_value = np.atleast_2d(min_value), np.atleast_2d(max_value)
        gather_data = scale_maxabs(gather_data, min_value=min_value, max_value=max_value, clip=False, eps=1e-10)
        return get_hodograph(gather_data, offsets, sample_interval, delay, picking_times, fill_value=np.nan)

    def calc(self, gather, refractor_velocity=None):
        """Return signal amplitudes at first break times.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity, optional
            Argument is not used. Preserved for quality control consistency.
        Returns
        -------
        metric : np.ndarray of float
            Signal amplitudes for each trace in the gather.
        """
        _ = refractor_velocity
        return self._calc(gather.data, gather["offset"], gather.sample_interval,
                          gather.delay, gather[self.first_breaks_header])

    def plot_gather(self, *args, **kwargs):
        """Plot the gather with amplitude values on top of the gather plot."""
        kwargs["mask"] = kwargs.pop("mask", False)
        super().plot_gather(*args, **kwargs)


class FirstBreaksPhases(RefractorVelocityMetric):
    """Mean absolute deviation of the signal phase from target value in the moment of first break.

    Parameters
    ----------
    target : float in range (-pi, pi] or str from {'max', 'min', 'transition'}, optional, defaults to 'max'
        Target phase value in the moment of first break: 0, pi, pi / 2 for `max`, `min` and `transition` respectively.
    window_size : int, defaults to 40
        Size of the window around the first break to attenuate during metric calculation. Measured in ms.
        Should be approximately equal to phase length around the first break.
    first_breaks_header : str, optional, defaults to None
        Column name from `survey.headers` where times of first break are stored.
        If not provided, must be set before the metric call via `set_defaults`.
    correct_uphole : bool, optional, defaults to None
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
        emulating the case when sources are located on the surface.
        If not provided, must be set before the metric call via `set_defaults`.
    name : str, optional
        Metric name, overrides default name if given.
    """

    vmin = 0
    vmax = np.pi / 2
    is_lower_better = True

    def __init__(self, target="max", window_size=40, first_breaks_header=None, correct_uphole=None, name=None):
        if isinstance(target, str):
            if target not in {"max", "min", "transition"}:
                raise KeyError("`target` should be one of {'max', 'min', 'transition'} or float.")
            target = {"max": 0, "min": np.pi, "transition": np.pi / 2}[target]
        self.target = target
        self.window_size = window_size
        super().__init__(first_breaks_header, correct_uphole, name)

    def calc(self, gather, refractor_velocity=None):
        """Return absolute deviation of the signal phase from target value in the moment of first break.        

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity, optional
            Argument is not used. Preserved for quality control consistency.

        Returns
        -------
        metric : np.ndarray of float
            Signal phase value at first break time for each trace in the gather.
        """
        _ = refractor_velocity
        # Shift traces to zero mean for correct Hilbert transform
        data = gather.data - gather.data.mean(axis=1).reshape(-1, 1)
        n_samples = ceil(self.window_size / gather.sample_interval) | 1
        windows = self._make_windows(gather[self.first_breaks_header], data, gather.offsets,
                                     n_samples, gather.sample_interval, gather.delay)
        filt = self._get_filter(n_samples)
        res = self._calc_phase_in_windows(windows, filt, self.target)
        return res

    @lru_cache
    def _get_filter(self, n_samples):
        half_n_samples = floor(n_samples / 2)
        filt_samples = np.arange(half_n_samples) + 1
        half_filt = 2 * np.sin(np.pi * filt_samples / 2)**2 / (np.pi * filt_samples)
        filt = np.concatenate((-half_filt[::-1], np.asarray([0.]), half_filt)) * np.hamming(2 * half_n_samples + 1)
        return np.ascontiguousarray(filt[::-1], dtype=np.float32)

    @staticmethod
    @njit(nogil=True)
    def _calc_phase_in_windows(windows, filt, target):
        n_traces, n_samples = windows.shape
        half_n_samples = floor(n_samples / 2)
        hilbert = np.empty(n_traces, dtype=windows.dtype)
        for i in range(n_traces):
            hilbert[i] = np.dot(windows[i], filt)
        fb_phases = np.arctan2(hilbert, windows[:, half_n_samples])
        # Map angles to range (target - pi, target + pi]
        if target > 0:
            fb_phases = np.where(fb_phases > target - np.pi, fb_phases, fb_phases + (2 * np.pi))
        else:
            fb_phases = np.where(fb_phases < target + np.pi, fb_phases, fb_phases - (2 * np.pi))
        return fb_phases - target

    def __call__(self, gather, refractor_velocity):
        """Return mean absolute deviation of the signal phase from target value in the moment of first break
        in the gather."""
        deltas = self.calc(gather, refractor_velocity)
        return np.mean(np.abs(deltas))

    def binarize(self, gather, metric_values, threshold=None):
        """Get binarized mask from metric_values."""
        _ = gather
        return super().binarize(gather, np.abs(metric_values), threshold)


class FirstBreaksCorrelations(RefractorVelocityMetric):
    """Mean Pearson correlation coefficient of trace with mean hodograph in window around the first break.

    Parameters
    ----------
    window_size : int, optional, defaults to 40
        Size of the window to calculate the correlation coefficient in. Measured in milliseconds.
    first_breaks_header : str, optional, defaults to None
        Column name from `survey.headers` where times of first break are stored.
        If not provided, must be set before the metric call via `set_defaults`.
    correct_uphole : bool, optional, defaults to None
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
        emulating the case when sources are located on the surface.
        If not provided, must be set before the metric call via `set_defaults`.
    name : str, optional
        Metric name, overrides default name if given.
    """

    views = ("plot_gather_window", "plot_mean_hodograph")
    vmin = 0
    vmax = 1
    is_lower_better = False

    def __init__(self, window_size=40, first_breaks_header=None, correct_uphole=None, name=None):
        self.window_size = window_size
        super().__init__(first_breaks_header, correct_uphole, name)

    @staticmethod
    @njit(nogil=True)
    def _calc(traces_windows):
        """Calculate signal correlation with mean hodograph"""
        n_traces, n_samples = traces_windows.shape
        mean_hodograph = np.empty((1, n_samples), dtype=traces_windows.dtype)
        for i in range(n_samples):
            mean_hodograph[:, i] = np.nanmean(traces_windows[:, i])
        mean_hodograph_scaled = (mean_hodograph - np.mean(mean_hodograph)) / np.std(mean_hodograph)

        corrs = np.empty(n_traces, dtype=traces_windows.dtype)
        for i in range(n_traces):
            trace_mean = np.nanmean(traces_windows[i])
            trace_std = np.nanstd(traces_windows[i])
            trace_window_scaled = (traces_windows[i] - trace_mean) / trace_std
            corrs[i] = np.nanmean(trace_window_scaled * mean_hodograph_scaled)
        return corrs

    def calc(self, gather, refractor_velocity=None):
        """Return signal correlation with mean hodograph in the given window around first break times
        for a scaled gather.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity, optional
            Argument is not used. Preserved for quality control consistency.

        Returns
        -------
        metric : np.ndarray of float
            Window correlation with mean hodograph for each trace in the gather.
        """
        _ = refractor_velocity
        n_samples = ceil(self.window_size / gather.sample_interval) | 1
        traces_windows = self._make_windows(gather[self.first_breaks_header], gather.data, gather["offset"],
                                            n_samples, gather.sample_interval, gather.delay)
        return self._calc(traces_windows)

    def get_views(self, sort_by=None, threshold=None, **kwargs):
        """Return plotters of the metric views and parse `plot_gather_window` kwargs for interactive map plotting."""
        views_list = [partial(self.plot_gather_window, sort_by=sort_by, threshold=threshold), self.plot_mean_hodograph]
        return views_list, kwargs

    def plot_gather_window(self, coords, ax, index, sort_by=None, threshold=None, mask=True,
                           top_header=True, **kwargs):
        """Plot traces around first break times in winow_size orbit and highlight ones negatively correlated
        with mean hodograph.
        """
        _ = coords
        gather = self.survey.get_gather(index)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        plotting_kwargs = {"mode": "wiggle"}
        if top_header or mask:
            metric_values = self.calc(gather=gather, refractor_velocity=None)
            if top_header:
                kwargs["top_header"] = metric_values
            if mask:
                mask_threshold = get_first_defined(threshold, self.vmin, metric_values.mean())
                plotting_kwargs["masks"] = {"masks": metric_values * -1, "threshold": mask_threshold * -1}
        plotting_kwargs.update(kwargs)
        n_samples = ceil(self.window_size / gather.sample_interval) | 1
        traces_windows = self._make_windows(gather[self.first_breaks_header], gather.data, gather["offset"],
                                            n_samples, gather.sample_interval, gather.delay)
        from ..gather import Gather # pylint: disable=import-outside-toplevel
        windows_gather = Gather(gather.headers, traces_windows, gather.sample_interval,
                                gather.survey, delay=0)
        windows_gather.plot(ax=ax, **plotting_kwargs)
        set_ticks(ax, axis="y", label="Window time, ms")

    def plot_mean_hodograph(self, coords, ax, index, **kwargs):
        """Plot mean trace in the scaled gather around the first break with length of the given window size."""
        _ = coords
        gather = self.survey.get_gather(index)
        n_samples = ceil(self.window_size / gather.sample_interval) | 1
        traces_windows = self._make_windows(gather[self.first_breaks_header], gather.data, gather["offset"],
                                            n_samples, gather.sample_interval, gather.delay)
        mean_hodograph = np.nanmean(traces_windows, axis=0)
        mean_hodograph_scaled = ((mean_hodograph - mean_hodograph.mean()) / mean_hodograph.std()).reshape(1, -1)
        fb_time_mean = np.mean(gather[self.first_breaks_header], dtype=np.int32)
        from ..gather import Gather # pylint: disable=import-outside-toplevel
        mean_hodograph_gather = Gather({'MeanHodohraph': [0]}, mean_hodograph_scaled,
                                       gather.sample_interval, gather.survey,
                                       delay=fb_time_mean - self.window_size // 2)
        mean_hodograph_gather.plot(mode="wiggle", ax=ax, **kwargs)
        set_ticks(ax, axis="x", label="Amplitude", num=3)


class DivergencePoint(RefractorVelocityMetric):
    """The divergence point metric for first breaks.
    Find an offset after that first breaks are most likely to diverge from expected time.
    Such an offset is defined as the last that preserves tolerable mean absolute deviation from expected time.

    Parameters
    ----------
    tol : float, optional, defaults to 10
        Tolerable mean absolute deviation from expected time. Measured in milliseconds.
    first_breaks_header : str, optional, defaults to None
        Column name from `survey.headers` where times of first break are stored.
        If not provided, must be set before the metric call via `set_defaults`.
    correct_uphole : bool, optional, defaults to None
        Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
        emulating the case when sources are located on the surface.
        If not provided, must be set before the metric call via `set_defaults`.
    name : str, optional
        Metric name, overrides default name if given.
    """

    is_lower_better = False

    def __init__(self, tol=10, first_breaks_header=None, correct_uphole=None, name=None):
        super().__init__(first_breaks_header, correct_uphole, name)
        self.tol = tol

    def bind_context(self, *args, **kwargs):
        """Set map attributes according to provided context."""
        super().bind_context(*args, **kwargs)
        self.vmax = self.survey["offset"].max()
        self.vmin = self.survey["offset"].min()

    @staticmethod
    @njit(nogil=True)
    def _calc(offsets, deviations, tol):
        """Calculate divergence offset for the gather."""
        sorted_offsets_idx = np.argsort(offsets)
        sorted_offsets = offsets[sorted_offsets_idx]
        sorted_deviations = deviations[sorted_offsets_idx]
        cum_means = np.cumsum(sorted_deviations) / np.arange(1, len(sorted_deviations) + 1)
        less_tol = np.nonzero(cum_means <= tol)[0]
        div_idx = less_tol[-1] if len(less_tol) > 0 else -1
        div_offset = sorted_offsets[div_idx]
        return div_offset

    def calc(self, gather, refractor_velocity):
        """Calculate divergence offset for the gather.

        Parameters
        ----------
        gather : Gather
            A seismic gather to get offsets and times of first breaks from.
        refractor_velocity : RefractorVelocity
            Near-surface velocity model to estimate the expected first break times at `gather` offsets.

        Returns
        -------
        metric : np.ndarray of bool
            Array indicating whether first break of each trace in the gather is diverged.
        """
        times = gather[self.first_breaks_header]
        if self.correct_uphole:
            times = times + gather["SourceUpholeTime"]
        offsets = gather["offset"]
        rv_times = refractor_velocity(offsets)
        deviations = np.abs(rv_times - times)
        div_offset = self._calc(offsets, deviations, self.tol)
        metric = np.full_like(offsets, div_offset)
        return metric

    def binarize(self, gather, metric_values, threshold=None):
        """Calculate whether traces are after the divergence offset."""
        threshold = metric_values[0] + 1 if threshold is None else threshold
        return {"masks": gather["offset"], "threshold": threshold}

    def get_views(self, sort_by="offset", **kwargs):
        """Return plotters of the metric views and add kwargs for `plot_gather` for interactive map plotting."""
        return super().get_views(sort_by=sort_by, **kwargs)

    def plot_gather(self, *args, **kwargs):
        """Plot the gather with highlighted traces after the divergence offset."""
        kwargs["top_header"] = kwargs.pop("top_header", False)
        super().plot_gather(*args, **kwargs)

    def plot_refractor_velocity(self, coords, ax, index, **kwargs):
        """Plot the refractor velocity curve, show the divergence offset
        and threshold area used for metric calculation."""
        gather = self.survey.get_gather(index)
        rv = self.field(coords, is_geographic=self.is_geographic)
        divergence_offset = self.calc(gather, rv)[0]
        title = f"Divergence point: {divergence_offset} m"
        super().plot_refractor_velocity(coords, ax, index, title=title, **kwargs)
        ax.axvline(x=divergence_offset, color="k", linestyle="--")

REFRACTOR_VELOCITY_QC_METRICS = [FirstBreaksOutliers, FirstBreaksPhases, FirstBreaksCorrelations,
                                 FirstBreaksAmplitudes, DivergencePoint]
