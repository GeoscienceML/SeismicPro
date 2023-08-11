"""Implements Gather class that represents a group of seismic traces that share some common acquisition parameter"""

import os
import math
import warnings
from itertools import cycle
from textwrap import dedent

import cv2
import scipy
import segyio
import numpy as np
from scipy.signal import firwin
from matplotlib.path import Path
from matplotlib.patches import Polygon, PathPatch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cropped_gather import CroppedGather
from .plot_corrections import NMOCorrectionPlot, LMOCorrectionPlot
from .utils import correction, normalization, gain
from .utils import convert_times_to_mask, convert_mask_to_pick, mute_gather, make_origins
from ..utils import (to_list, get_coords_cols, get_first_defined, set_ticks, format_subplot_yticklabels,
                     set_text_formatting, add_colorbar, piecewise_polynomial, Coordinates)
from ..containers import TraceContainer, SamplesContainer
from ..muter import Muter, MuterField
from ..velocity_spectrum import VerticalVelocitySpectrum, ResidualVelocitySpectrum
from ..stacking_velocity import StackingVelocity, StackingVelocityField
from ..refractor_velocity import RefractorVelocity, RefractorVelocityField
from ..decorators import batch_method, plotter
from ..const import HDR_FIRST_BREAK, HDR_TRACE_POS, DEFAULT_STACKING_VELOCITY
from ..velocity_spectrum.utils.coherency_funcs import stacked_amplitude


class Gather(TraceContainer, SamplesContainer):
    """A class representing a single seismic gather.

    A gather is a collection of seismic traces that share some common acquisition parameter (usually common values of
    trace headers used as an index in their survey). Unlike `Survey`, `Gather` instances store loaded seismic traces
    along with the corresponding subset of the parent survey trace headers.

    `Gather` instance is generally created by calling one of the following methods of a `Survey`, `SeismicIndex` or
    `SeismicDataset`:
    1. `sample_gather` - to get a randomly selected gather,
    2. `get_gather` - to get a particular gather by its index value.

    Most of the methods change gather data inplace, thus `Gather.copy` may come in handy to keep the original gather
    intact.

    Examples
    --------
    Load a randomly selected common source gather, sort it by offset and plot:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> gather.plot()

    Parameters
    ----------
    headers : pd.DataFrame
        Headers of gather traces. Must be a subset of parent survey trace headers.
    data : 2d np.ndarray
        Trace data of the gather with (n_traces, n_samples) layout.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    survey : Survey
        A survey that generated the gather.
    delay : float, optional, defaults to 0
        Delay recording time of seismic traces. Measured in milliseconds.

    Attributes
    ----------
    headers : pd.DataFrame
        Headers of gather traces.
    data : 2d np.ndarray
        Trace data of the gather with (n_traces, n_samples) layout.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.
    survey : Survey
        A survey that generated the gather.
    sort_by : None or str or list of str
        Headers that were used for gather sorting. If `None`, no sorting was performed.
    """
    def __init__(self, headers, data, sample_interval, survey, delay=0):
        if sample_interval <= 0:
            raise ValueError("Sample interval must be positive")
        if delay < 0:
            raise ValueError("Delay must be non-negative")
        self.headers = headers
        self.data = data
        self.samples = self.create_samples(data.shape[1], sample_interval, delay)
        self.sample_interval = sample_interval
        self.delay = delay
        self.survey = survey
        self.sort_by = None

    @property
    def index(self):
        """int or tuple of int or None: Common value of `Survey`'s `header_index` that define traces of the gather.
        `None` if the gather is combined.
        """
        indices = self.headers.index.drop_duplicates()
        if len(indices) != 1:
            return None
        return indices[0]

    @property
    def offsets(self):
        """1d np.ndarray of floats: The distance between source and receiver for each trace. Measured in meters."""
        return self["offset"]

    @property
    def shape(self):
        """tuple with 2 elements: The number of traces in the gather and trace length in samples."""
        return self.data.shape

    @property
    def coords(self):
        """Coordinates or None: Spatial coordinates of the gather. Headers to extract coordinates from are determined
        automatically by the `indexed_by` attribute of the gather. `None` if the gather is indexed by unsupported
        headers or required coords headers were not loaded or coordinates are non-unique for traces of the gather."""
        try:  # Possibly unknown coordinates for indexed_by, required coords headers may be not loaded
            coords_cols = get_coords_cols(self.indexed_by, self.survey.source_id_cols, self.survey.receiver_id_cols)
            coords = self[coords_cols]
        except KeyError:
            return None
        if (coords != coords[0]).any():  # Non-unique coordinates
            return None
        return Coordinates(coords[0], names=coords_cols)

    def __getitem__(self, key):
        """Either select values of gather headers by their names or create a new `Gather` with specified traces and
        samples depending on the key type.

        Notes
        -----
        1. Only basic indexing and slicing is supported along the time axis in order to preserve constant sample rate.
        2. If the traces are no longer sorted after `__getitem__`, `sort_by` attribute of the resulting `Gather` is set
           to `None`.
        3. If headers selection is performed, the returned array will be 1d if a single header is selected and 2d
           otherwise.

        Parameters
        ----------
        key : str, list of str, int, list, tuple, slice
            If str or list of str, gather headers to get as an `np.ndarray`. The returned array is 1d if a single
            header is selected and 2d otherwise.
            Otherwise, indices of traces and samples to get. In this case, `__getitem__` behavior almost coincides with
            that of `np.ndarray`, except that only basic indexing and slicing is supported along the time axis in order
            to preserve constant sample rate.

        Returns
        -------
        result : np.ndarray or Gather
            Values of gather headers or a gather with the specified subset of traces and samples.

        Raises
        ------
        ValueError
            If the resulting gather is empty or the number of data dimensions has changed.
        """
        # If key is str or array of str, treat it as names of headers columns
        if all(isinstance(item, str) for item in to_list(key)):
            return super().__getitem__(key)

        # Split key into indexers of traces and samples
        key = (key,) if not isinstance(key, tuple) else key
        key = key + (slice(None),) if len(key) == 1 else key
        if len(key) != 2 or any(indexer is None for indexer in key):
            raise KeyError("Data ndim must not change")
        traces_indexer, samples_indexer = key

        def int_to_slice(ix, size):
            """Convert an integer index to a slice to be further used for array indexing, that will return a view and
            preserve the number of array dimensions."""
            if ix < 0:
                ix += size
            if (ix < 0) or (ix > size - 1):
                raise IndexError("gather index out of range")
            return slice(ix, ix + 1)

        # Cast samples indexer to a slice so that possible advanced indexing is performed only along traces axis
        if isinstance(samples_indexer, (int, np.integer)):
            samples_indexer = int_to_slice(samples_indexer, self.n_samples)
        if not isinstance(samples_indexer, slice):
            raise KeyError("Only basic indexing and slicing is supported along the time axis")
        delay_ix, _, samples_step = samples_indexer.indices(self.n_samples)
        if samples_step < 0:
            raise ValueError("Negative step is not allowed for samples slicing")

        # Cast a single trace indexer to a slice
        if isinstance(traces_indexer, (int, np.integer)):
            traces_indexer = int_to_slice(traces_indexer, self.n_traces)

        # Index data and make it C-contiguous since otherwise some numba functions may fail
        data = np.require(self.data[traces_indexer, samples_indexer], requirements="C")
        if data.size == 0:  # e.g. after empty slicing
            raise ValueError("Empty gather after indexation")
        headers = self.headers.iloc[traces_indexer]
        sample_interval = self.sample_interval * samples_step
        gather = Gather(headers, data, sample_interval, delay=self.samples[delay_ix], survey=self.survey)

        # Preserve gather sorting if needed
        if self.sort_by is not None:
            if isinstance(traces_indexer, slice):
                if traces_indexer.step is None or traces_indexer.step > 0:
                    gather.sort_by = self.sort_by
            elif (np.diff(traces_indexer) >= 0).all():
                gather.sort_by = self.sort_by
        return gather

    def __str__(self):
        """Print gather metadata including information about its survey, headers and traces."""
        # Calculate offset range
        offsets = self.headers.get('offset')
        offset_range = f'[{np.min(offsets)} m, {np.max(offsets)} m]' if offsets is not None else "Unknown"

        # Count the number of zero/constant traces
        n_dead_traces = np.isclose(np.max(self.data, axis=1), np.min(self.data, axis=1)).sum()

        try:
            sample_interval_str = f"{self.sample_interval} ms"
            sample_rate_str = f"{self.sample_rate} Hz"
        except ValueError:
            sample_interval_str = "Irregular"
            sample_rate_str = "Irregular"

        msg = f"""
        Parent survey path:          {self.survey.path}
        Parent survey name:          {self.survey.name}

        Number of traces:            {self.n_traces}
        Trace length:                {self.n_samples} samples
        Sample interval:             {sample_interval_str}
        Sample rate:                 {sample_rate_str}
        Times range:                 [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:               {offset_range}

        Indexed by:                  {', '.join(to_list(self.indexed_by))}
        Index value:                 {get_first_defined(self.index, "Combined")}
        Gather coordinates:          {get_first_defined(self.coords, "Unknown")}
        Gather sorting:              {self.sort_by}

        Gather statistics:
        Number of dead traces:       {n_dead_traces}
        mean | std:                  {np.mean(self.data):>10.2f} | {np.std(self.data):<10.2f}
         min | max:                  {np.min(self.data):>10.2f} | {np.max(self.data):<10.2f}
         q01 | q99:                  {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """
        return dedent(msg).strip()

    def info(self):
        """Print gather metadata including information about its survey, headers and traces."""
        print(self)

    @batch_method(target='threads', copy_src=False)
    def copy(self, ignore=None):
        """Perform a deepcopy of all gather attributes except for `survey` and those specified in ignore, which are
        kept unchanged.

        Parameters
        ----------
        ignore : str or array of str, defaults to None
            Attributes that won't be copied.

        Returns
        -------
        copy : Gather
            Copy of the gather.
        """
        ignore = set() if ignore is None else set(to_list(ignore))
        return super().copy(ignore | {"survey"})

    @batch_method(target="for", copy_src=False)
    def get_item(self, *args):
        """A pipeline interface for `self.__getitem__` method."""
        return self[args if len(args) > 1 else args[0]].copy()

    def _post_index(self, key):
        """Index gather data by provided `key`."""
        self.data = self.data[key]

    # Target set to `for` to avoid race condition when the same trace appears in two gathers (ex. supergathers)
    @batch_method(target='for', use_lock=True)
    def store_headers_to_survey(self, columns):
        """Save given headers from the gather to its survey.

        Notes
        -----
        The correct result is guaranteed only if the `self.survey` has not been modified after `self` creation.

        Parameters
        ----------
        columns : str or list of str
            Column names from `self.headers` that will be stored to `self.survey` headers.

        Returns
        -------
        self : Gather
            Gather unchanged.
        """
        columns = to_list(columns)

        headers = self.survey.headers
        pos = self[HDR_TRACE_POS]
        for column in columns:
            column_data = self[column] # Here we also check that column is in self.headers
            if column not in headers:
                headers[column] = np.nan

            if not np.can_cast(column_data, headers.dtypes[column]):
                headers[column] = headers[column].astype(column_data.dtype)

            # FIXME: Workaround for a pandas bug https://github.com/pandas-dev/pandas/issues/48998
            # iloc may call unnecessary copy of the whole column before setitem
            headers[column].array[pos] = column_data
        return self

    #------------------------------------------------------------------------#
    #                              Dump methods                              #
    #------------------------------------------------------------------------#

    @batch_method(target='for', force=True)
    def dump(self, path, name=None, retain_parent_segy_headers=True):
        """Save the gather to a `.sgy` file.

        Notes
        -----
        1. All textual and almost all binary headers are copied from the parent SEG-Y file unchanged except for the
           following binary header fields that are inferred by the current gather:
           1) Sample rate, bytes 3217-3218, called `Interval` in `segyio`,
           2) Number of samples per data trace, bytes 3221-3222, called `Samples` in `segyio`,
           3) Extended number of samples per data trace, bytes 3269-3272, called `ExtSamples` in `segyio`.
        2. Bytes 117-118 of trace header (called `TRACE_SAMPLE_INTERVAL` in `segyio`) for each trace is filled with
           sample rate of the current gather.

        Parameters
        ----------
        path : str
            A directory to dump the gather in.
        name : str, optional, defaults to None
            The name of the file. If `None`, the concatenation of the survey name and the value of gather index will
            be used.
        retain_parent_segy_headers : bool, optional, defaults to True
            Whether to copy the headers that weren't loaded during `Survey` creation from the parent SEG-Y file.

        Returns
        -------
        self : Gather
            Gather unchanged.

        Raises
        ------
        ValueError
            If empty `name` was specified.
        """
        parent_handler = self.survey.loader.file_handler

        if name is None:
            # Use the first value of gather index to handle combined case
            name = "_".join(map(str, [self.survey.name] + to_list(self.headers.index.values[0])))
        if name == "":
            raise ValueError("Argument `name` can not be empty.")
        if not os.path.splitext(name)[1]:
            name += ".sgy"
        full_path = os.path.join(path, name)

        os.makedirs(path, exist_ok=True)
        # Create segyio spec. We choose only specs that relate to unstructured data.
        spec = segyio.spec()
        spec.samples = self.samples
        spec.ext_headers = parent_handler.ext_headers
        spec.format = parent_handler.format
        spec.tracecount = self.n_traces

        sample_interval = np.int32(self.sample_interval * 1000) # Convert to microseconds
        # Remember ordinal numbers of traces in the parent SEG-Y file to further copy their headers
        trace_ids = self["TRACE_SEQUENCE_FILE"] - 1

        # Keep only headers, defined by SEG-Y standard.
        used_header_names = self.available_headers & set(segyio.tracefield.keys)
        used_header_names = to_list(used_header_names)

        # Transform header's names into byte number based on the SEG-Y standard.
        used_header_bytes = [segyio.tracefield.keys[header_name] for header_name in used_header_names]

        with segyio.create(full_path, spec) as dump_handler:
            # Copy the binary header from the parent SEG-Y file and update it with samples data of the gather.
            # TODO: Check if other bin headers matter
            dump_handler.bin = parent_handler.bin
            dump_handler.bin[segyio.BinField.Interval] = sample_interval
            dump_handler.bin[segyio.BinField.Samples] = self.n_samples
            dump_handler.bin[segyio.BinField.ExtSamples] = self.n_samples

            # Copy textual headers from the parent SEG-Y file.
            for i in range(spec.ext_headers + 1):
                dump_handler.text[i] = parent_handler.text[i]

            # Dump traces and their headers. Optionally copy headers from the parent SEG-Y file.
            dump_handler.trace = self.data
            for i, trace_headers in enumerate(self[used_header_names]):
                if retain_parent_segy_headers:
                    dump_handler.header[i] = parent_handler.header[trace_ids[i]]
                dump_handler.header[i].update({**dict(zip(used_header_bytes, trace_headers)),
                                               segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval,
                                               segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1})
        return self

    #------------------------------------------------------------------------#
    #                         Normalization methods                          #
    #------------------------------------------------------------------------#

    def get_quantile(self, q, tracewise=False, use_global=False):
        """Calculate the `q`-th quantile of the gather or fetch the global quantile from the parent survey.

        Notes
        -----
        The `tracewise` mode is only available when `use_global` is `False`.

        Parameters
        ----------
        q : float or array-like of floats
            Quantile or a sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        tracewise : bool, optional, default False
            If `True`, the quantiles are computed for each trace independently, otherwise for the entire gather.
        use_global : bool, optional, default False
            If `True`, the survey's quantiles are used, otherwise the quantiles are computed for the gather data.

        Returns
        -------
        q : float or array-like of floats
            The `q`-th quantile values.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        if use_global:
            return self.survey.get_quantile(q)
        q = np.array(q, dtype=np.float32)
        if not tracewise:
            quantiles = np.nanquantile(self.data, q=q)
        else:
            quantiles = normalization.get_tracewise_quantile(self.data, q=np.atleast_1d(q))
            # return the same type as q: either single float or array-like
            if q.ndim == 0:
                quantiles = quantiles[0]
        return quantiles.astype(self.data.dtype)

    @batch_method(target='threads')
    def scale_standard(self, tracewise=True, use_global=False, eps=1e-10):
        r"""Standardize the gather by removing the mean and scaling to unit variance.

        The standard score of a gather `g` is calculated as:
        :math:`G = \frac{g - m}{s + eps}`,
        where:
        `m` - the mean of the gather or global average if `use_global=True`,
        `s` - the standard deviation of the gather or global standard deviation if `use_global=True`,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Standardization is performed inplace.

        Parameters
        ----------
        tracewise : bool, optional, defaults to True
            If `True`, mean and standard deviation are calculated for each trace independently. Otherwise they are
            calculated for the entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's mean and std are used, otherwise gather statistics are computed.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Standardized gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        if use_global:
            if not self.survey.has_stats:
                raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
            mean = np.atleast_2d(self.survey.mean)
            std = np.atleast_2d(self.survey.std)
        elif not tracewise:
            mean = np.nanmean(self.data, keepdims=True)
            std = np.nanstd(self.data, keepdims=True)
        else:
            mean, std = normalization.get_tracewise_mean_std(self.data)
        self.data = normalization.scale_standard(self.data, mean, std, np.float32(eps))
        return self

    @batch_method(target='threads')
    def scale_maxabs(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        r"""Scale the gather by its maximum absolute value.

        Maxabs scale of the gather `g` is calculated as:
        :math: `G = \frac{g}{m + eps}`,
        where:
        `m` - the maximum of absolute values of `q_min`-th and `q_max`-th quantiles,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Quantiles are used to minimize the effect of amplitude outliers on the scaling result. Default 0 and 1
        quantiles represent the minimum and maximum values of the gather respectively and result in usual max-abs
        scaler behavior.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Maxabs scaling is performed inplace.

        Parameters
        ----------
        q_min : float, optional, defaults to 0
            A quantile to be used as a gather minimum during scaling.
        q_max : float, optional, defaults to 1
            A quantile to be used as a gather maximum during scaling.
        tracewise : bool, optional, defaults to True
            If `True`, quantiles are calculated for each trace independently. Otherwise they are calculated for the
            entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's quantiles are used, otherwise gather quantiles are computed.
        clip : bool, optional, defaults to False
            Whether to clip the scaled gather to the [-1, 1] range.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Scaled gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        self.data = normalization.scale_maxabs(self.data, min_value, max_value, clip, np.float32(eps))
        return self

    @batch_method(target='threads')
    def scale_minmax(self, q_min=0, q_max=1, tracewise=True, use_global=False, clip=False, eps=1e-10):
        r"""Linearly scale the gather to a [0, 1] range.

        The transformation of the gather `g` is given by:
        :math:`G=\frac{g - min}{max - min + eps}`
        where:
        `min` and `max` - `q_min`-th and `q_max`-th quantiles respectively,
        `eps` - a constant that is added to the denominator to avoid division by zero.

        Notes
        -----
        1. The presence of NaN values in the gather will lead to incorrect behavior of the scaler.
        2. Minmax scaling is performed inplace.

        Parameters
        ----------
        q_min : float, optional, defaults to 0
            A quantile to be used as a gather minimum during scaling.
        q_max : float, optional, defaults to 1
            A quantile to be used as a gather maximum during scaling.
        tracewise : bool, optional, defaults to True
            If `True`, quantiles are calculated for each trace independently. Otherwise they are calculated for the
            entire gather.
        use_global : bool, optional, defaults to False
            If `True`, parent survey's quantiles are used, otherwise gather quantiles are computed.
        clip : bool, optional, defaults to False
            Whether to clip the scaled gather to the [0, 1] range.
        eps : float, optional, defaults to 1e-10
            A constant to be added to the denominator to avoid division by zero.

        Returns
        -------
        self : Gather
            Scaled gather.

        Raises
        ------
        ValueError
            If `use_global` is `True` but global statistics were not calculated.
        """
        min_value, max_value = self.get_quantile([q_min, q_max], tracewise=tracewise, use_global=use_global)
        self.data = normalization.scale_minmax(self.data, min_value, max_value, clip, np.float32(eps))
        return self

    #------------------------------------------------------------------------#
    #                    First-breaks processing methods                     #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", copy_src=False)
    def pick_to_mask(self, first_breaks_header=HDR_FIRST_BREAK):
        """Convert first break times to a binary mask with the same shape as `gather.data` containing zeros before the
        first arrivals and ones after for each trace.

        Parameters
        ----------
        first_breaks_header : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            A column of `self.headers` that contains first arrival times, measured in milliseconds.

        Returns
        -------
        gather : Gather
            A new `Gather` with calculated first breaks mask in its `data` attribute.
        """
        mask = convert_times_to_mask(times=self[first_breaks_header], n_samples=self.n_samples,
                                     sample_interval=self.sample_interval, delay=self.delay)
        gather = self.copy(ignore='data')
        gather.data = mask.astype(np.float32)
        return gather

    @batch_method(target='threads', args_to_unpack='save_to')
    def mask_to_pick(self, threshold=0.5, first_breaks_header=HDR_FIRST_BREAK, save_to=None):
        """Convert a first break mask saved in `data` into times of first arrivals.

        For a given trace each value of the mask represents the probability that the corresponding time sample follows
        the first break.

        Notes
        -----
        A detailed description of conversion heuristic used can be found in :func:`~general_utils.convert_mask_to_pick`
        docs.

        Parameters
        ----------
        threshold : float, optional, defaults to 0.5
            A threshold for trace mask value to refer its index to be either pre- or post-first break.
        first_breaks_header : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Headers column to save first break times to.
        save_to : Gather or str, optional
            An extra `Gather` to save first break times to. Generally used to conveniently pass first break times from
            a `Gather` instance with a first break mask to an original `Gather`.
            May be `str` if called in a pipeline: in this case it defines a component with gathers to save first break
            times to.

        Returns
        -------
        self : Gather
            A gather with first break times in headers column defined by `first_breaks_header`.
        """
        picking_times = convert_mask_to_pick(mask=self.data, threshold=threshold, sample_interval=self.sample_interval,
                                             delay=self.delay)
        self[first_breaks_header] = picking_times
        if save_to is not None:
            save_to[first_breaks_header] = picking_times
        return self

    @batch_method(target="for", copy_src=False)  # pylint: disable-next=too-many-arguments
    def calculate_refractor_velocity(self, init=None, bounds=None, n_refractors=None, max_offset=None,
                                     min_velocity_step=1, min_refractor_size=1, loss="L1", huber_coef=20, tol=1e-5,
                                     first_breaks_header=HDR_FIRST_BREAK, correct_uphole=None, **kwargs):
        """Fit a near-surface velocity model by offsets of traces and times of their first breaks.

        Notes
        -----
        Please refer to the :class:`~refractor_velocity.RefractorVelocity` docs for more details about the velocity
        model, its computation algorithm and available parameters. At least one of `init`, `bounds` or `n_refractors`
        should be passed.

        Examples
        --------
        >>> refractor_velocity = gather.calculate_refractor_velocity(n_refractors=2)

        Parameters
        ----------
        init : dict, optional
            Initial values of model parameters.
        bounds : dict, optional
            Lower and upper bounds of model parameters.
        n_refractors : int, optional
            The number of refractors described by the model.
        max_offset : float, optional
            Maximum offset reliably described by the model. Inferred automatically by `offsets`, `init` and `bounds`
            provided but should be preferably explicitly passed.
        min_velocity_step : int, or 1d array-like with shape (n_refractors - 1,), optional, defaults to 1
            Minimum difference between velocities of two adjacent refractors. Default value ensures that velocities are
            strictly increasing.
        min_refractor_size : int, or 1d array-like with shape (n_refractors,), optional, defaults to 1
            Minimum offset range covered by each refractor. Default value ensures that refractors do not degenerate
            into single points.
        loss : str, optional, defaults to "L1"
            Loss function to be minimized. Should be one of "MSE", "huber", "L1", "soft_L1", or "cauchy".
        huber_coef : float, optional, default to 20
            Coefficient for Huber loss function.
        tol : float, optional, defaults to 1e-5
            Precision goal for the value of loss in the stopping criterion.
        first_breaks_header : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `self.headers` where times of first break are stored.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
            emulating the case when sources are located on the surface. If not given, correction is performed if
            "SourceUpholeTime" header is loaded.
        kwargs : misc, optional
            Additional `SLSQP` options, see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html for
            more details.

        Returns
        -------
        rv : RefractorVelocity
            Constructed near-surface velocity model.
        """
        times = self[first_breaks_header]
        if correct_uphole is None:
            correct_uphole = "SourceUpholeTime" in self.available_headers
        if correct_uphole:
            times = times + self["SourceUpholeTime"]
        return RefractorVelocity.from_first_breaks(self.offsets, times, init, bounds, n_refractors, max_offset,
                                                   min_velocity_step, min_refractor_size, loss, huber_coef, tol,
                                                   coords=self.coords, is_uphole_corrected=correct_uphole, **kwargs)

    #------------------------------------------------------------------------#
    #                         Gather muting methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", args_to_unpack="muter")
    def mute(self, muter, fill_value=np.nan):
        """Mute the gather using given `muter` which defines an offset-time boundary above which gather amplitudes will
        be set to `fill_value`.

        Parameters
        ----------
        muter : Muter, MuterField or str
            A muter to use. `Muter` instance is used directly. If `MuterField` instance is passed, a `Muter`
            corresponding to gather coordinates is fetched from it.
            May be `str` if called in a pipeline: in this case it defines a component with muters to apply.
        fill_value : float, optional, defaults to np.nan
            A value to fill the muted part of the gather with.

        Returns
        -------
        self : Gather
            Muted gather.
        """
        if isinstance(muter, MuterField):
            muter = muter(self.coords)
        if not isinstance(muter, Muter):
            raise ValueError("muter must be of Muter or MuterField type")
        self.data = mute_gather(gather_data=self.data, muting_times=muter(self.offsets),
                                sample_interval=self.sample_interval, delay=self.delay, fill_value=fill_value)
        return self

    #------------------------------------------------------------------------#
    #             Vertical Velocity Spectrum calculation methods             #
    #------------------------------------------------------------------------#

    @batch_method(target="for", args_to_unpack="stacking_velocity", copy_src=False)
    def calculate_vertical_velocity_spectrum(self, velocities=None, stacking_velocity=None, relative_margin=0.2,
                                             velocity_step=50, window_size=50, mode='semblance',
                                             max_stretch_factor=np.inf, interpolate=True):
        """Calculate vertical velocity spectrum for the gather.

        Notes
        -----
        A detailed description of vertical velocity spectrum and its computation algorithm can be found in
        :func:`~velocity_spectrum.VerticalVelocitySpectrum` docs.

        Examples
        --------
        Calculate vertical velocity spectrum with default parameters: velocities evenly spaced around default stacking
        velocity, 50 ms temporal window size, semblance coherency measure and no muting of hodograph stretching:
        >>> spectrum = gather.calculate_vertical_velocity_spectrum()

        Calculate vertical velocity spectrum for 200 velocities from 2000 to 6000 m/s, temporal window size of 128 ms,
        crosscorrelation coherency measure and muting of stretching effects greater than 65%:
        >>> spectrum = gather.calculate_vertical_velocity_spectrum(velocities=np.linspace(2000, 6000, 200), mode='CC',
                                                                   window_size=128, max_stretch_factor=0.65)

        Parameters
        ----------
        velocities : 1d np.ndarray, optional, defaults to None
            An array of stacking velocities to calculate the velocity spectrum for. Measured in meters/seconds. If not
            provided, `stacking_velocity` is evaluated for gather times to estimate the velocity range being examined.
            The resulting velocities are then evenly sampled from this range being additionally extended by
            `relative_margin` * 100% in both directions with a step of `velocity_step`.
        stacking_velocity : StackingVelocity or StackingVelocityField or str, optional,
                            defaults to DEFAULT_STACKING_VELOCITY
            Stacking velocity around which vertical velocity spectrum is calculated if `velocities` are not given.
            `StackingVelocity` instance is used directly. If `StackingVelocityField` instance is passed, a
            `StackingVelocity` corresponding to gather coordinates is fetched from it. May be `str` if called in a
            pipeline: in this case it defines a component with stacking velocities to use.
        relative_margin : float, optional, defaults to 0.2
            Relative velocity margin to additionally extend the velocity range obtained from `stacking_velocity`: an
            interval [`min_velocity`, `max_velocity`] is mapped to [(1 - `relative_margin`) * `min_velocity`,
            (1 + `relative_margin`) * `max_velocity`].
        velocity_step : float, optional, defaults to 50
            A step between two adjacent velocities for which vertical velocity spectrum is calculated if `velocities`
            are not passed. Measured in meters/seconds.
        window_size : int, optional, defaults to 50
            Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
            the resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
        mode: str, optional, defaults to 'semblance'
            The measure for estimating hodograph coherency.
            The available options are:
                `semblance` or `NE`,
                `stacked_amplitude` or `S`,
                `normalized_stacked_amplitude` or `NS`,
                `crosscorrelation` or `CC`,
                `energy_normalized_crosscorrelation` or `ENCC`.
        max_stretch_factor : float, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO
            correction. This mute is applied after NMO correction for each provided velocity and before coherency
            calculation. The lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
            Reasonably good value is 0.65.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude
            at the nearest time sample is used.

        Returns
        -------
        vertical_velocity_spectrum : VerticalVelocitySpectrum
            Calculated vertical velocity spectrum.
        """
        return VerticalVelocitySpectrum(gather=self, velocities=velocities, stacking_velocity=stacking_velocity,
                                        relative_margin=relative_margin, velocity_step=velocity_step,
                                        window_size=window_size, mode=mode, max_stretch_factor=max_stretch_factor,
                                        interpolate=interpolate)

    @batch_method(target="for", args_to_unpack="stacking_velocity", copy_src=False)
    def calculate_residual_velocity_spectrum(self, stacking_velocity, relative_margin=0.2, velocity_step=25,
                                             window_size=50, mode="semblance", max_stretch_factor=np.inf,
                                             interpolate=True):
        """Calculate residual velocity spectrum for the gather and provided stacking velocity.

        Notes
        -----
        A detailed description of residual velocity spectrum and its computation algorithm can be found in
        :func:`~velocity_spectrum.ResidualVelocitySpectrum` docs.

        Examples
        --------
        Calculate residual velocity spectrum for a gather and a stacking velocity, loaded from a file:
        >>> velocity = StackingVelocity.from_file(velocity_path)
        >>> spectrum = gather.calculate_residual_velocity_spectrum(velocity, velocity_step=100, window_size=32)

        Parameters
        ----------
        stacking_velocity : StackingVelocity or StackingVelocityField or str
            Stacking velocity around which residual velocity spectrum is calculated. `StackingVelocity` instance is
            used directly. If `StackingVelocityField` instance is passed, a `StackingVelocity` corresponding to gather
            coordinates is fetched from it. May be `str` if called in a pipeline: in this case it defines a component
            with stacking velocities to use.
        relative_margin : float, optional, defaults to 0.2
            Relative velocity margin, that determines the velocity range for velocity spectrum calculation for each
            time `t` as `stacking_velocity(t)` * (1 +- `relative_margin`).
        velocity_step : float, optional, defaults to 25
            A step between two adjacent velocities for which residual velocity spectrum is calculated. Measured in
            meters/seconds.
        window_size : int, optional, defaults to 50
            Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
            the resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
        mode: str, optional, defaults to 'semblance'
            The measure for estimating hodograph coherency.
            The available options are:
                `semblance` or `NE`,
                `stacked_amplitude` or `S`,
                `normalized_stacked_amplitude` or `NS`,
                `crosscorrelation` or `CC`,
                `energy_normalized_crosscorrelation` or `ENCC`.
        max_stretch_factor : float, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO
            correction. This mute is applied after NMO correction for each provided velocity and before coherency
            calculation. The lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
            Reasonably good value is 0.65.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude
            at the nearest time sample is used.

        Returns
        -------
        residual_velocity_spectrum : ResidualVelocitySpectrum
            Calculated residual velocity spectrum.
        """
        return ResidualVelocitySpectrum(gather=self, stacking_velocity=stacking_velocity,
                                        relative_margin=relative_margin, velocity_step=velocity_step,
                                        window_size=window_size, mode=mode, max_stretch_factor=max_stretch_factor,
                                        interpolate=interpolate)

    #------------------------------------------------------------------------#
    #                           Gather corrections                           #
    #------------------------------------------------------------------------#

    @batch_method(target="threads", args_to_unpack="refractor_velocity")
    def apply_lmo(self, refractor_velocity, delay=100, fill_value=np.nan, event_headers=None, correct_uphole=None):
        """Perform gather linear moveout correction using the given near-surface velocity model.

        Parameters
        ----------
        refractor_velocity : int, float, RefractorVelocity, RefractorVelocityField or str
            Near-surface velocity model to perform LMO correction with. `RefractorVelocity` instance is used directly.
            If `RefractorVelocityField` instance is passed, a `RefractorVelocity` corresponding to gather coordinates
            is fetched from it. If `int` or `float` then constant-velocity correction is performed.
            May be `str` if called in a pipeline: in this case it defines a component with refractor velocities to use.
        delay : float, optional, defaults to 100
            An extra delay in milliseconds introduced in each trace, positive values result in shifting gather traces
            down. Used to center the first breaks hodograph around the delay value instead of 0.
        fill_value : float, optional, defaults to np.nan
            Value used to fill the amplitudes outside the gather bounds after moveout.
        event_headers : str, list, or None, optional, defaults to None
            Headers columns which will be LMO-corrected inplace.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to estimated delay
            times. If enabled, centers first breaks around `delay` for uphole surveys. If not given, correction is
            performed if "SourceUpholeTime" header is loaded and given `refractor_velocity` was also uphole corrected.

        Returns
        -------
        self : Gather
            LMO-corrected gather.

        Raises
        ------
        ValueError
            If wrong type of `refractor_velocity` is passed.
        """
        if isinstance(refractor_velocity, (int, float)):
            refractor_velocity = RefractorVelocity.from_constant_velocity(refractor_velocity)
        if isinstance(refractor_velocity, RefractorVelocityField):
            refractor_velocity = refractor_velocity(self.coords)
        if not isinstance(refractor_velocity, RefractorVelocity):
            raise ValueError("refractor_velocity must be of int, float, RefractorVelocity or RefractorVelocityField "
                             "type")

        trace_delays = delay - refractor_velocity(self.offsets)
        if correct_uphole is None:
            correct_uphole = "SourceUpholeTime" in self.available_headers and refractor_velocity.is_uphole_corrected
        if correct_uphole:
            trace_delays += self["SourceUpholeTime"]
        trace_delays_samples = np.rint(trace_delays / self.sample_interval).astype(np.int32)
        self.data = correction.apply_lmo(self.data, trace_delays_samples, fill_value)
        if event_headers is not None:
            self[to_list(event_headers)] += trace_delays.reshape(-1, 1)
        return self

    @batch_method(target="threads", args_to_unpack="stacking_velocity")
    def apply_nmo(self, stacking_velocity, max_stretch_factor=np.inf, interpolate=True, fill_value=np.nan):
        """Perform gather normal moveout correction using the given stacking velocity.

        Notes
        -----
        A detailed description of NMO correction can be found in :func:`~utils.correction.apply_nmo` docs.

        Parameters
        ----------
        stacking_velocity : int, float, StackingVelocity, StackingVelocityField or str
            Stacking velocities to perform NMO correction with. `StackingVelocity` instance is used directly. If
            `StackingVelocityField` instance is passed, a `StackingVelocity` corresponding to gather coordinates is
            fetched from it. If `int` or `float` then constant-velocity correction is performed. May be `str` if called
            in a pipeline: in this case it defines a component with stacking velocities to use.
        max_stretch_factor : float, optional, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO
            correction. The lower the value, the stronger the mute. In case np.inf (default) only areas where time
            reversal occurred are muted. Reasonably good value is 0.65.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude
            at the nearest time sample is used.
        fill_value : float, optional, defaults to np.nan
            Fill value to use if hodograph time is outside the gather bounds.

        Returns
        -------
        self : Gather
            NMO-corrected gather.

        Raises
        ------
        ValueError
            If wrong type of `stacking_velocity` is passed.
        """
        if isinstance(stacking_velocity, (int, float, np.number)):
            stacking_velocity = np.float32(stacking_velocity / 1000)  # from m/s to m/ms
            self.data = correction.apply_constant_velocity_nmo(self.data, self.offsets, self.sample_interval,
                                                               self.delay, self.times, stacking_velocity,
                                                               max_stretch_factor, interpolate, fill_value)
            return self

        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(self.coords)
        if not isinstance(stacking_velocity, StackingVelocity):
            raise ValueError("stacking_velocity must be of int, float, StackingVelocity or StackingVelocityField type")

        velocities = stacking_velocity(self.times) / 1000  # from m/s to m/ms
        velocities_grad = np.gradient(velocities, self.sample_interval)
        self.data = correction.apply_nmo(self.data, self.offsets, self.sample_interval, self.delay, self.times,
                                         velocities, velocities_grad, max_stretch_factor, interpolate, fill_value)
        return self

    #------------------------------------------------------------------------#
    #                       General processing methods                       #
    #------------------------------------------------------------------------#

    @batch_method(target="threads")
    def sort(self, by):
        """Sort gather by specified headers.

        Parameters
        ----------
        by : str or iterable of str
            Headers names to sort the gather by.

        Returns
        -------
        self : Gather
            Gather sorted by given headers. Sets `sort_by` attribute to `by`.
        """
        by = to_list(by)
        if by == to_list(self.sort_by)[:len(by)]:
            return self

        order = np.lexsort(self[by[::-1]].T)
        self.sort_by = by[0] if len(by) == 1 else by
        self.data = self.data[order]
        self.headers = self.headers.iloc[order]
        return self

    @batch_method(target="threads")
    def get_central_gather(self):
        """Get a central CDP gather from a supergather.

        A supergather has `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D` headers columns, whose values equal to
        values of `INLINE_3D` and `CROSSLINE_3D` only for traces from the central CDP gather. Read more about
        supergather generation in :func:`~Survey.generate_supergathers` docs.

        Returns
        -------
        self : Gather
            `self` with only traces from the central CDP gather kept. Updates `self.headers` and `self.data` inplace.
        """
        line_cols = self["INLINE_3D", "CROSSLINE_3D", "SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        mask = (line_cols[:, :2] == line_cols[:, 2:]).all(axis=1)
        self.headers = self.headers.loc[mask]
        self.data = self.data[mask]
        return self

    @batch_method(target="threads")
    def stack(self, amplify_factor=0):
        """Stack a gather by calculating mean value of all non-nan amplitudes for each time over the offset axis.

        The resulting gather will contain a single trace with `headers` matching those of the first input trace.

        Parameters
        ----------
        amplify_factor : float in range [0, 1], optional, defaults to 0
            Amplifying factor which affects the normalization of the sum of hodographs amplitudes.
            The amplitudes sum is multiplied by amplify_factor/sqrt(N) + (1 - amplify_factor)/N, where N is the number
            of live (non muted) amplitudes. Acts as the coherency amplifier for long hodographs. Note that in case
            amplify_factor=0 (default), sum of trace amplitudes is simply divided by N, so that stack amplitude is the
            average of ensemble amplitudes. Must be in [0, 1] range.

        Returns
        -------
        gather : Gather
            Stacked gather.
        """
        amplify_factor = np.clip(amplify_factor, 0, 1)
        self.headers = self.headers.iloc[[0]]  # Preserve headers of the first trace of the gather being stacked
        self.data = stacked_amplitude(self.data, amplify_factor, abs=False)[0].reshape(1, -1)
        return self

    def crop(self, origins, crop_shape, n_crops=1, stride=None, pad_mode='constant', **kwargs):
        """Crop gather data.

        Parameters
        ----------
        origins : list, tuple, np.ndarray or str
            Origins define top-left corners for each crop (the first trace and the first time sample respectively)
            or a rule used to calculate them. All array-like values are cast to an `np.ndarray` and treated as origins
            directly, except for a 2-element tuple of `int`, which will be treated as a single individual origin.
            If `str`, represents a mode to calculate origins. Two options are supported:
            - "random": calculate `n_crops` crops selected randomly using a uniform distribution over the gather data,
              so that no crop crosses gather boundaries,
            - "grid": calculate a deterministic uniform grid of origins, whose density is determined by `stride`.
        crop_shape : tuple with 2 elements
            Shape of the resulting crops.
        n_crops : int, optional, defaults to 1
            The number of generated crops if `origins` is "random".
        stride : tuple with 2 elements, optional, defaults to crop_shape
            Steps between two adjacent crops along both axes if `origins` is "grid". The lower the value is, the more
            dense the grid of crops will be. An extra origin will always be placed so that the corresponding crop will
            fit in the very end of an axis to guarantee complete data coverage with crops regardless of passed
            `crop_shape` and `stride`.
        pad_mode : str or callable, optional, defaults to 'constant'
            Padding mode used when a crop with given origin and shape crossed boundaries of gather data. Passed
            directly to `np.pad`, see https://numpy.org/doc/stable/reference/generated/numpy.pad.html for more
            details.
        kwargs : dict, optional
            Additional keyword arguments to `np.pad`.

        Returns
        -------
        crops : CroppedGather
            Calculated gather crops.
        """
        origins = make_origins(origins, self.shape, crop_shape, n_crops, stride)
        return CroppedGather(self, origins, crop_shape, pad_mode, **kwargs)

    @batch_method(target="threads")
    def bandpass_filter(self, low=None, high=None, filter_size=81, **kwargs):
        """Filter frequency spectrum of the gather.

        `low` and `high` define the range of the remaining signal frequencies. If both of them are given, acts as a
        bandpass filter. If only one of them is given, acts as a highpass or lowpass filter respectively.

        Examples
        --------
        Apply highpass filter: remove all the frequencies bellow 30 Hz.
        >>> gather.bandpass_filter(low=30)

        Apply bandpass filter: keep frequencies within [30, 100] Hz range.
        >>> gather.bandpass_filter(low=30, high=100)

        Apply lowpass filter, remove all the frequencies above 100 Hz.
        >>> gather.bandpass_filter(high=100)

        Notes
        -----
        Default `filter_size` is set to 81 to guarantee that transition bandwidth of the filter does not exceed 10% of
        the Nyquist frequency for the default Hamming window.

        Parameters
        ----------
        low : float, optional
            Lower bound for the remaining frequencies.
        high : float, optional
            Upper bound for the remaining frequencies.
        filter_size : int, optional, defaults to 81
            The length of the filter.
        kwargs : misc, optional
            Additional keyword arguments to the `scipy.firwin`.

        Returns
        -------
        self : Gather
            `self` with filtered frequency spectrum.
        """
        filter_size |= 1  # Guarantee that filter size is odd
        pass_zero = low is None
        cutoffs = [cutoff for cutoff in [low, high] if cutoff is not None]

        # Construct the filter and flip it since opencv computes crosscorrelation instead of convolution
        kernel = firwin(filter_size, cutoffs, pass_zero=pass_zero, fs=self.sample_rate, **kwargs)[::-1]
        cv2.filter2D(self.data, dst=self.data, ddepth=-1, kernel=kernel.reshape(1, -1))
        return self

    @batch_method(target="threads")
    def resample(self, new_sample_interval, kind=3, enable_anti_aliasing=True):
        """Change sample interval of traces in the gather.

        This method changes the number of samples in each trace if the new sample interval differs from the current
        one. If downsampling is performed, an anti-aliasing filter can optionally be applied to avoid frequency
        aliasing.

        Parameters
        ----------
        new_sample_interval : float
            New sample interval of seismic traces.
        kind : int or str, optional, defaults to 3
            The interpolation method to use.
            If `int`, use piecewise polynomial interpolation with degree `kind`.
            If `str`, delegate interpolation to scipy.interp1d with mode `kind`.
        enable_anti_aliasing : bool, optional, defaults to True
            Whether to apply anti-aliasing filter or not. Ignored in case of upsampling.

        Returns
        -------
        self : Gather
            Resampled gather.
        """
        current_sample_interval = self.sample_interval
        if current_sample_interval == new_sample_interval:
            return self

        # Anti-aliasing filter is optionally applied during downsampling to avoid frequency aliasing
        if enable_anti_aliasing and new_sample_interval > current_sample_interval:
            # Smoothly attenuate frequencies starting from 0.8 of the new Nyquist frequency so that all frequencies
            # above the new Nyquist frequency are zeroed out
            nyquist_frequency = 1000 / (2 * new_sample_interval)
            filter_size = int(40 * new_sample_interval / current_sample_interval)
            self.bandpass_filter(high=0.9*nyquist_frequency, filter_size=filter_size, window="hann")

        new_n_samples = math.floor((self.samples[-1] - self.samples[0]) / new_sample_interval) + 1
        new_samples = self.create_samples(new_n_samples, new_sample_interval, self.delay)

        if isinstance(kind, int):
            data_resampled = piecewise_polynomial(new_samples, self.samples, self.data, kind)
        elif isinstance(kind, str):
            data_resampled = scipy.interpolate.interp1d(self.samples, self.data, kind=kind)(new_samples)

        self.data = data_resampled
        self.sample_interval = new_sample_interval
        self.samples = new_samples
        return self

    def apply_agc(self, window_size=250, mode='rms', return_coefs=False):
        """Calculate instantaneous or RMS amplitude AGC coefficients and apply them to gather data.

        Parameters
        ----------
        window_size : int, optional, defaults to 250
            Window size to calculate AGC scaling coefficient in, measured in milliseconds.
        mode : str, optional, defaults to 'rms'
            Mode for AGC: if 'rms', root mean squared value of non-zero amplitudes in the given window
            is used as scaling coefficient (RMS amplitude AGC), if 'abs' - mean of absolute non-zero
            amplitudes (instantaneous AGC).
        return_coefs : bool, optional, defaults to False
            Whether to return a `Gather` with AGC coefficients in `data` attribute.

        Raises
        ------
        ValueError
            If window_size is less than 2 * `sample_interval` milliseconds or larger than trace length.
            If mode is neither 'rms' nor 'abs'.

        Returns
        -------
        self : Gather
            Gather with AGC applied to its data.
        coefs_gather : Gather, optional
            Gather with AGC coefficients in `data` attribute. Returned only if `return_coefs` was set to `True`.
        """
        # Cast window from ms to samples
        window_size_samples = int(window_size // self.sample_interval) + 1

        if mode not in ['abs', 'rms']:
            raise ValueError(f"mode should be either 'abs' or 'rms', but {mode} was given")
        if (window_size_samples < 3) or (window_size_samples > self.n_samples):
            raise ValueError(f'window_size should be at least {2*self.sample_interval} milliseconds and '
                             f'{(self.n_samples-1)*self.sample_interval} at most, but {window_size} was given')
        # Avoid using str in funciton decorated with njit for performance reasons
        use_rms_mode = mode == 'rms'
        data, coefs = gain.apply_agc(data=self.data, window_size=window_size_samples, use_rms_mode=use_rms_mode)
        self.data = data
        if return_coefs:
            coefs_gather = self.copy(ignore="data")
            coefs_gather.data = coefs
            return self, coefs_gather
        return self

    def undo_agc(self, coefs_gather):
        """Undo previously applied AGC correction using precomputed AGC coefficients.

        Parameters
        ----------
        coefs_gather : Gather
            Gather with AGC coefficients in `data` attribute.

        Returns
        -------
        self : Gather
            Gather without AGC.
        """
        self.data = gain.undo_agc(data=self.data, coefs=coefs_gather.data)
        return self

    @batch_method(target="threads")
    def apply_sdc(self, velocity=None, v_pow=2, t_pow=1):
        """Calculate spherical divergence correction coefficients and apply them to gather data.

        Parameters
        ----------
        velocities: StackingVelocity or None, optional, defaults to None.
            StackingVelocity that is used to obtain velocities at self.times, measured in meters / second.
            If None, default StackingVelocity object is used.
        v_pow : float, optional, defaults to 2
            Velocity power value.
        t_pow: float, optional, defaults to 1
            Time power value.

        Returns
        -------
        self : Gather
            Gather with applied SDC.
        """
        if velocity is None:
            velocity = DEFAULT_STACKING_VELOCITY
        if not isinstance(velocity, StackingVelocity):
            raise ValueError("Only StackingVelocity instance or None can be passed as velocity")
        self.data = gain.apply_sdc(self.data, v_pow, velocity(self.times), t_pow, self.times)
        return self

    @batch_method(target="threads")
    def undo_sdc(self, velocity=None, v_pow=2, t_pow=1):
        """Calculate spherical divergence correction coefficients and use them to undo previously applied SDC.

        Parameters
        ----------
        velocities: StackingVelocity or None, optional, defaults to None.
            StackingVelocity that is used to obtain velocities at self.times, measured in meters / second.
            If None, default StackingVelocity object is used.
        v_pow : float, optional, defaults to 2
            Velocity power value.
        t_pow: float, optional, defaults to 1
            Time power value.

        Returns
        -------
        self : Gather
            Gather without SDC.
        """
        if velocity is None:
            velocity = DEFAULT_STACKING_VELOCITY
        if not isinstance(velocity, StackingVelocity):
            raise ValueError("Only StackingVelocity instance or None can be passed as velocity")
        self.data = gain.undo_sdc(self.data, v_pow, velocity(self.times), t_pow, self.times)
        return self

    #------------------------------------------------------------------------#
    #                         Visualization methods                          #
    #------------------------------------------------------------------------#

    @plotter(figsize=(10, 7), args_to_unpack="masks")
    def plot(self, mode="seismogram", *, title=None, x_ticker=None, y_ticker=None, ax=None, **kwargs):
        """Plot gather traces.

        The traces can be displayed in a number of representations, depending on the `mode` provided. Currently, the
        following options are supported:
        - `seismogram`: a 2d grayscale image of seismic traces. This mode supports the following `kwargs`:
            * `colorbar`: whether to add a colorbar to the right of the gather plot (defaults to `False`). If `dict`,
              defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`,
            * `q_vmin`, `q_vmax`: quantile range of amplitude values covered by the colormap (defaults to 0.1 and 0.9),
            * Any additional arguments for `matplotlib.pyplot.imshow`. Note, that `vmin` and `vmax` arguments take
              priority over `q_vmin` and `q_vmax` respectively.
        - `wiggle`: an amplitude vs time plot for each trace of the gather as an oscillating line around its mean
          amplitude. This mode supports the following `kwargs`:
            * `norm_tracewise`: specifies whether to standardize each trace independently or use gather mean amplitude
              and standard deviation (defaults to `True`),
            * `std`: amplitude scaling factor. Higher values result in higher plot oscillations (defaults to 0.5),
            * `lw` and `alpha`: width of the lines and transparency of polygons, by default estimated
              based on the number of traces in the gather and figure size.
            * `color`: defines a color for traces,
            * Any additional arguments for `matplotlib.pyplot.plot`.
        - `hist`: a histogram of the trace data amplitudes or header values. This mode supports the following `kwargs`:
            * `bins`: if `int`, the number of equal-width bins; if sequence, bin edges that include the left edge of
              the first bin and the right edge of the last bin,
            * `grid`: whether to show the grid lines,
            * `log`: set y-axis to log scale. If `True`, formatting defined in `y_ticker` is discarded,
            * Any additional arguments for `matplotlib.pyplot.hist`.

        Some areas of a gather may be highlighted in color by passing optional `masks` argument. Trace headers, whose
        values are measured in milliseconds (e.g. first break times) may be displayed over a seismogram or wiggle plot
        if passed as `event_headers`. If `top_header` is passed, an auxiliary scatter plot of values of this header
        will be shown on top of the gather plot.

        While the source of label ticks for both `x` and `y` is defined by `x_tick_src` and `y_tick_src`, ticker
        appearance can be controlled via `x_ticker` and `y_ticker` parameters respectively. In the most general form,
        each of them is a `dict` with the following most commonly used keys:
        - `label`: axis label. Can be any string.
        - `round_to`: the number of decimal places to round tick labels to (defaults to 0).
        - `rotation`: the rotation angle of tick labels in degrees (defaults to 0).
        - One of the following keys, defining the way to place ticks:
            * `num`: place a given number of evenly-spaced ticks,
            * `step_ticks`: place ticks with a given step between two adjacent ones,
            * `step_labels`: place ticks with a given step between two adjacent ones in the units of the corresponding
              labels (e.g. place a tick every 200ms for `y` axis or every 300m offset for `x` axis). This option is
              valid only for "seismogram" and "wiggle" modes.
        A short argument form allows defining both tickers labels as a single `str`, which will be treated as the value
        for the `label` key. See :func:`~plot_utils.set_ticks` for more details on the ticker parameters.

        Parameters
        ----------
        mode : "seismogram", "wiggle" or "hist", optional, defaults to "seismogram"
            A type of the gather representation to display:
            - "seismogram": a 2d grayscale image of seismic traces;
            - "wiggle": an amplitude vs time plot for each trace of the gather;
            - "hist": histogram of the data amplitudes or some header values.
        title : str or dict, optional, defaults to None
            If `str`, a title of the plot.
            If `dict`, should contain keyword arguments to pass to `matplotlib.axes.Axes.set_title`. In this case, the
            title string is stored under the `label` key.
        x_ticker : str or dict, optional, defaults to None
            Parameters to control `x` axis label and ticker formatting and layout.
            If `str`, it will be displayed as axis label.
            If `dict`, the axis label is specified under the "label" key and the rest of keys define labels formatting
            and layout, see :func:`~plot_utils.set_ticks` for more details.
            If not given, axis label is defined by `x_tick_src`.
        y_ticker : str or dict, optional, defaults to None
            Parameters to control `y` axis label and ticker formatting and layout.
            If `str`, it will be displayed as axis label.
            If `dict`, the axis label is specified under the "label" key and the rest of keys define labels formatting
            and layout, see :func:`~plot_utils.set_ticks` for more details.
            If not given, axis label is defined by `y_tick_src`.
        ax : matplotlib.axes.Axes, optional, defaults to None
            An axis of the figure to plot on. If not given, it will be created automatically.
        x_tick_src : str, optional
            Source of the tick labels to be plotted on x axis. For "seismogram" and "wiggle" can be either "index"
            (default if gather is not sorted) or any header; for "hist" it also defines the data source and can be
            either "amplitude" (default) or any header.
            Also serves as a default for axis label.
        y_tick_src : str, optional
            Source of the tick labels to be plotted on y axis. For "seismogram" and "wiggle" can be either "time"
            (default) or "samples"; has no effect in "hist" mode. Also serves as a default for axis label.
        event_headers : str, array-like or dict, optional, defaults to None
            Valid only for "seismogram" and "wiggle" modes.
            Headers, whose values will be displayed over the gather plot. Must be measured in milliseconds.
            If `dict`, allows controlling scatter plot options and handling outliers (header values falling out the `y`
            axis range). The following keys are supported:
            - `headers`: header names, can be either `str` or an array-like.
            - `process_outliers`: an approach for outliers processing. Available options are:
                * `clip`: clip outliers to fit the range of `y` axis,
                * `discard`: do not display outliers,
                * `none`: plot all the header values (default behavior).
            - Any additional arguments for `matplotlib.axes.Axes.scatter`.
            If some dictionary value is array-like, each its element will be associated with the corresponding header.
            Otherwise, the single value will be used for all the scatter plots.
        top_header : str, array-like, optional, defaults to None
            Valid only for "seismogram" and "wiggle" modes.
            If str, the name of a header whose values will be plotted on top of the gather plot.
            If array-like, the value for each trace that will be plotted on top of the gather plot.
        masks : array-like, str, dict or Gather, optional, defaults to None
            Valid only for "seismogram" and "wiggle" modes.
            Mask or list of masks to plot on top of the gather plot.
            If `array-like` either mask or list of masks where each mask should be one of:
            - `2d array`, a mask with shape equals to self.shape to plot on top of the gather plot;
            - `1d array`, a vector containing self.n_traces elements that determines which traces to mask;
            - `Gather`, its `data` attribute will be treated as a mask, note that Gather shape should be the same as
            self.shape;
            - `str`, either a header name to take mask from or a batch component name.
            If `dict`, the mask (or list of masks) is specified under the "masks" key and the rest of keys define masks
            layout. The following keys are supported:
                - `masks`: mask or list of masks,
                - `threshold`: the value after which all values will be threated as mask,
                - `label`: the name of the mask that will be shown in legend,
                - `color`: mask color,
                - `alpha`: mask transparency.
            If some dictionary value is array-like, each its element will be associated with the corresponding mask.
            Otherwise, the single value will be used for all masks.
        figsize : tuple, optional, defaults to (10, 7)
            Size of the figure to create if `ax` is not given. Measured in inches.
        save_to : str or dict, optional, defaults to None
            If `str`, a path to save the figure to.
            If `dict`, should contain keyword arguments to pass to `matplotlib.pyplot.savefig`. In this case, the path
            is stored under the `fname` key.
            If `None`, the figure is not saved.
        kwargs : misc, optional
            Additional keyword arguments to the plotter depending on the `mode`.

        Returns
        -------
        self : Gather
            Gather unchanged.

        Raises
        ------
        ValueError
            If given `mode` is unknown.
            If `colorbar` is not `bool` or `dict`.
            If `event_headers` argument has the wrong format or given outlier processing mode is unknown.
            If `x_ticker` or `y_ticker` has the wrong format.
        """
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        # Plot the gather depending on the mode passed
        plotters_dict = {
            "seismogram": self._plot_seismogram,
            "wiggle": self._plot_wiggle,
            "hist": self._plot_histogram,
        }
        if mode not in plotters_dict:
            raise ValueError(f"Unknown mode {mode}")

        plotters_dict[mode](ax, title=title, x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)
        return self

    def _plot_histogram(self, ax, title, x_ticker, y_ticker, x_tick_src="amplitude", bins=None,
                        log=False, grid=True, **kwargs):
        """Plot histogram of the data specified by x_tick_src."""
        if x_tick_src.title() == 'Amplitude':
            x_tick_src = 'Amplitude'
            data = self.data.ravel()
        else:
            data = self[x_tick_src]

        _ = ax.hist(data, bins=bins, **kwargs)
        set_ticks(ax, "x", **{"label": x_tick_src, 'round_to': None, **x_ticker})
        set_ticks(ax, "y", **{"label": "Counts", **y_ticker})

        ax.grid(grid)
        if log:
            ax.set_yscale("log")
        ax.set_title(**{'label': None, **title})

    # pylint: disable=too-many-arguments
    def _plot_seismogram(self, ax, title, x_ticker, y_ticker, x_tick_src=None, y_tick_src='time', colorbar=False,
                         q_vmin=0.1, q_vmax=0.9, event_headers=None, top_header=None, masks=None, **kwargs):
        """Plot the gather as a 2d grayscale image of seismic traces."""
        # Make the axis divisible to further plot colorbar and header subplot
        divider = make_axes_locatable(ax)
        vmin, vmax = self.get_quantile([q_vmin, q_vmax])
        kwargs = {"cmap": "gray", "aspect": "auto", "vmin": vmin, "vmax": vmax, **kwargs}
        img = ax.imshow(self.data.T, **kwargs)
        if masks is not None:
            default_mask_kwargs = {"aspect": "auto", "alpha": 0.5, "interpolation": "none"}
            for mask_kwargs in self._process_masks(masks):
                mask_kwargs = {**default_mask_kwargs, **mask_kwargs}
                mask = mask_kwargs.pop("masks")
                cmap = ListedColormap(mask_kwargs.pop("color"))
                label = mask_kwargs.pop("label")
                # Add an invisible artist to display mask label on the legend since imshow does not support it
                ax.add_patch(Polygon([[0, 0]], color=cmap(1), label=label, alpha=mask_kwargs["alpha"]))
                ax.imshow(mask.T, cmap=cmap, **mask_kwargs)
        add_colorbar(ax, img, colorbar, divider, y_ticker)
        self._finalize_plot(ax, title, divider, event_headers, top_header, x_ticker, y_ticker, x_tick_src, y_tick_src)

    #pylint: disable=invalid-name
    def _plot_wiggle(self, ax, title, x_ticker, y_ticker, x_tick_src=None, y_tick_src="time", norm_tracewise=True,
                     std=0.5, event_headers=None, top_header=None, masks=None, lw=None, alpha=None, color="black",
                     **kwargs):
        """Plot the gather as an amplitude vs time plot for each trace."""
        def _get_start_end_ixs(ixs):
            """Return arrays with indices of beginnings and ends of polygons defined by continuous subsequences in
            `ixs`."""
            start_ix = np.argwhere((np.diff(ixs[:, 0], prepend=ixs[0, 0]) != 0) |
                                   (np.diff(ixs[:, 1], prepend=ixs[0, 1]) != 1)).ravel()
            end_ix = start_ix + np.diff(start_ix, append=len(ixs)) - 1
            return start_ix, end_ix

        # Make the axis divisible to further plot colorbar and header subplot
        divider = make_axes_locatable(ax)

        # The default parameters lw = 1 and alpha = 1 are fine for 150 traces gather being plotted on 7.75 inches width
        # axes(by default created by gather.plot()). Scale this parameters linearly for bigger gathers or smaller axes.
        axes_width = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted()).width

        MAX_TRACE_DENSITY = 150 / 7.75
        BOUNDS = [[0.25, 1], [0, 1.5]] # The clip limits for parameters after linear scale.

        alpha, lw = [np.clip(MAX_TRACE_DENSITY * (axes_width / self.n_traces), *val_bounds) if val is None else val
                     for val, val_bounds in zip([alpha, lw], BOUNDS)]

        std_axis = 1 if norm_tracewise else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            traces = std * ((self.data - np.nanmean(self.data, axis=1, keepdims=True)) /
                            (np.nanstd(self.data, axis=std_axis, keepdims=True) + 1e-10))

        # Shift trace amplitudes according to the trace index in the gather
        amps = traces + np.arange(traces.shape[0]).reshape(-1, 1)
        # Plot all the traces as one Line, then hide transitions between adjacent traces
        amps = np.concatenate([amps, np.full((len(amps), 1), np.nan)], axis=1)
        ax.plot(amps.ravel(), np.broadcast_to(np.arange(amps.shape[1]), amps.shape).ravel(),
                color=color, lw=lw, **kwargs)

        # Find polygons bodies: indices of target amplitudes, start and end
        poly_amp_ix = np.argwhere(traces > 0)
        start_ix, end_ix = _get_start_end_ixs(poly_amp_ix)
        shift = np.arange(len(start_ix)) * 3
        # For each polygon we need to:
        # 1. insert 0 amplitude at the start.
        # 2. append 0 amplitude to the end.
        # 3. append the start point to the end to close polygon.
        # Fill the array storing resulted polygons
        verts = np.empty((len(poly_amp_ix) + 3 * len(start_ix), 2))
        verts[start_ix + shift] = poly_amp_ix[start_ix]
        verts[end_ix + shift + 2] = poly_amp_ix[end_ix]
        verts[end_ix + shift + 3] = poly_amp_ix[start_ix]

        body_ix = np.setdiff1d(np.arange(len(verts)),
                               np.concatenate([start_ix + shift, end_ix + shift + 2, end_ix + shift + 3]),
                               assume_unique=True)
        verts[body_ix] = np.column_stack([amps[tuple(poly_amp_ix.T)], poly_amp_ix[:, 1]])

        # Fill the array representing the nodes codes: either start, intermediate or end code.
        codes = np.full(len(verts), Path.LINETO)
        codes[start_ix + shift] = Path.MOVETO
        codes[end_ix + shift + 3] = Path.CLOSEPOLY

        patch = PathPatch(Path(verts, codes), color=color, alpha=alpha)
        ax.add_artist(patch)

        if masks is not None:
            for mask_kwargs in self._process_masks(masks):
                mask = mask_kwargs.pop("masks")
                mask_ix = np.argwhere(mask > 0)
                start_ix, end_ix = _get_start_end_ixs(mask_ix)
                # Compute the polygon bodies, that represent mask coordinates with a small indent
                up_verts = mask_ix[start_ix].reshape(-1, 1, 2) + np.array([[0.5, -0.5], [-0.5, -0.5]])
                down_verts = mask_ix[end_ix].reshape(-1, 1, 2) + np.array([[-0.5, 0.5], [0.5, 0.5]])
                # Combine upper and lower vertices and add plaсeholders for Path.CLOSEPOLY code with coords [0, 0]
                # after each polygon.
                verts = np.hstack((up_verts, down_verts, np.zeros((len(up_verts), 1, 2)))).reshape(-1, 2)

                # Fill the array representing the nodes codes: either start, intermediate or end code.
                codes = np.full(len(verts), Path.LINETO)
                codes[::5] = Path.MOVETO
                codes[4::5] = Path.CLOSEPOLY

                default_mask_kwargs = {"alpha": alpha*0.7, "lw": 0}
                mask_patch = PathPatch(Path(verts, codes), **{**default_mask_kwargs, **mask_kwargs})
                ax.add_artist(mask_patch)

        ax.update_datalim([(0, 0), traces.shape])
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        self._finalize_plot(ax, title, divider, event_headers, top_header, x_ticker, y_ticker, x_tick_src, y_tick_src)

    def _finalize_plot(self, ax, title, divider, event_headers, top_header,
                       x_ticker, y_ticker, x_tick_src, y_tick_src):
        """Plot optional artists and set ticks on the `ax`. Utility method for 'seismogram' and 'wiggle' modes."""
        # Add headers scatter plot if needed
        if event_headers is not None:
            self._plot_headers(ax, event_headers)

        # Add a top subplot for given header if needed and set plot title
        top_ax = ax
        if top_header is not None:
            if isinstance(top_header, str):
                header_values = self[top_header]
            elif isinstance(top_header, (np.ndarray, list, tuple)) and len(top_header) == self.n_traces:
                header_values = top_header
            else:
                msg = f"`top_header` should be `str`, `np.ndarray`, `list` or `tuple` not `{type(top_header)}`"
                warnings.warn(msg)
                header_values = None

            if header_values is not None:
                top_ax = self._plot_top_subplot(ax=ax, divider=divider, header_values=header_values, y_ticker=y_ticker)

        # Set axis ticks.
        self._set_x_ticks(ax, tick_src=x_tick_src, ticker=x_ticker)
        self._set_y_ticks(ax, tick_src=y_tick_src, ticker=y_ticker)

        top_ax.set_title(**{'label': None, **title})

        if len(ax.get_legend_handles_labels()[0]):
            # Define legend position to speed up plotting for huge gathers
            ax.legend(loc='upper right')

    def _process_masks(self, masks):
        colors_iterator = cycle(['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:pink',
                                 'tab:olive', 'tab:cyan'])
        masks_list = self._parse_headers_kwargs(masks, "masks")
        for ix, (mask_dict, default_color) in enumerate(zip(masks_list, colors_iterator)):
            mask = mask_dict["masks"]
            if isinstance(mask, Gather):
                mask = mask.data
            elif isinstance(mask, str):
                mask_dict["label"] = mask_dict.get("label", mask)
                mask = self[mask]

            mask = np.array(mask)
            if mask.ndim == 1:
                mask = mask.reshape(-1, 1)
            threshold = mask_dict.pop("threshold", 0.5)
            mask_dict["masks"] = np.broadcast_to(np.where(mask < threshold, np.nan, 1), self.shape)
            mask_dict["label"] = mask_dict.get("label", f"Mask {ix+1}")
            mask_dict["color"] = mask_dict.get("color", default_color)
        return [mask_dict for mask_dict in masks_list if not np.isnan(mask_dict["masks"]).all()]

    @staticmethod
    def _parse_headers_kwargs(headers_kwargs, headers_key):
        """Construct a `dict` of kwargs for each header defined in `headers_kwargs` under `headers_key` key so that it
        contains all other keys from `headers_kwargs` with the values defined as follows:
        1. If the value in `headers_kwargs` is a list or tuple, it is indexed with the index of the currently processed
           header,
        2. Otherwise, it is kept unchanged.

        Examples
        --------
        >>> headers_kwargs = {
        ...     "headers": ["FirstBreakTrue", "FirstBreakPred"],
        ...     "s": 5,
        ...     "c": ["blue", "red"]
        ... }
        >>> Gather._parse_headers_kwargs(headers_kwargs, headers_key="headers")
        [{'headers': 'FirstBreakTrue', 's': 5, 'c': 'blue'},
         {'headers': 'FirstBreakPred', 's': 5, 'c': 'red'}]
        """
        if not isinstance(headers_kwargs, dict):
            headers_kwargs = headers_kwargs if isinstance(headers_kwargs, (list, tuple)) else [headers_kwargs]
            return [{headers_key: header} for header in headers_kwargs]

        if headers_key not in headers_kwargs:
            raise KeyError(f'Missing {headers_key} key in passed kwargs')

        headers_kwargs = {key: value if isinstance(value, (list, tuple)) else [value]
                          for key, value in headers_kwargs.items()}
        n_headers = len(headers_kwargs[headers_key])

        kwargs_list = [{} for _ in range(n_headers)]
        for key, values in headers_kwargs.items():
            if len(values) == 1:
                values = values * n_headers
            elif len(values) != n_headers:
                raise ValueError(f"Incompatible length of {key} array: {n_headers} expected but {len(values)} given.")
            for ix, value in enumerate(values):
                kwargs_list[ix][key] = value
        return kwargs_list

    def _plot_headers(self, ax, headers_kwargs):
        """Add scatter plots of values of one or more headers over the main gather plot."""
        x_coords = np.arange(self.n_traces)
        kwargs_list = self._parse_headers_kwargs(headers_kwargs, "headers")
        for kwargs in kwargs_list:
            kwargs = {"zorder": 10, **kwargs}  # Increase zorder to plot headers on top of gather
            header = kwargs.pop("headers")
            label = kwargs.pop("label", header)
            process_outliers = kwargs.pop("process_outliers", "none")
            y_coords = self.times_to_indices(self[header])
            if process_outliers == "clip":
                y_coords = np.clip(y_coords, 0, self.n_samples - 1)
            elif process_outliers == "discard":
                y_coords = np.where((y_coords >= 0) & (y_coords <= self.n_samples - 1), y_coords, np.nan)
            elif process_outliers != "none":
                raise ValueError(f"Unknown outlier processing mode {process_outliers}")
            ax.scatter(x_coords, y_coords, label=label, **kwargs)

    def _plot_top_subplot(self, ax, divider, header_values, y_ticker, **kwargs):
        """Add a scatter plot of given header values on top of the main gather plot."""
        top_ax = divider.append_axes("top", sharex=ax, size="12%", pad=0.05)
        top_ax.scatter(np.arange(self.n_traces), header_values, **{"s": 5, "color": "black", **kwargs})
        top_ax.xaxis.set_visible(False)
        top_ax.yaxis.tick_right()
        top_ax.invert_yaxis()
        format_subplot_yticklabels(top_ax, **y_ticker)
        return top_ax

    def _set_x_ticks(self, ax, tick_src, ticker):
        """Infer and set ticks for x axis. """
        tick_src = to_list(tick_src or self.sort_by or 'index')[:2]
        if tick_src[0].title() == "Index":
            tick_src[0] = "Index"
            major_labels, minor_labels = np.arange(self.n_traces), None
        elif len(tick_src) == 1:
            major_labels, minor_labels =  self[tick_src[0]], None
        else:
            major_labels, minor_labels = self[tick_src[0]], self[tick_src[1]]

        # Format axis label
        UNITS = {  # pylint: disable=invalid-name
            "offset": ", m",
        }

        tick_src = [ix_tick_src + UNITS.get(ix_tick_src, '') for ix_tick_src in tick_src]
        axis_label = '\n'.join(tick_src)

        set_ticks(ax, 'x', major_labels=major_labels, minor_labels=minor_labels, **{"label": axis_label, **ticker})

    def _set_y_ticks(self, ax, tick_src, ticker):
        """Infer and set ticks for y axis. """
        tick_src = tick_src.title()
        if tick_src == "Time":
            tick_src = "Time, ms"
            major_labels =  self.samples
        if tick_src == "Samples":
            major_labels = np.arange(self.n_samples)

        set_ticks(ax, 'y', major_labels=major_labels, **{"label": tick_src, **ticker})

    def plot_nmo_correction(self, min_vel=1500, max_vel=6000, figsize=(6, 4.5), show_grid=True, **kwargs):
        """Perform interactive NMO correction of the gather with selected constant velocity.

        The plot provides 2 views:
        * Corrected gather (default). NMO correction is performed on the fly with the velocity controlled by a slider
          on top of the plot.
        * Source gather. This view disables the velocity slider.

        Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        min_vel : float, optional, defaults to 1500
            Minimum seismic velocity value for NMO correction. Measured in meters/seconds.
        max_vel : float, optional, defaults to 6000
            Maximum seismic velocity value for NMO correction. Measured in meters/seconds.
        figsize : tuple with 2 elements, optional, defaults to (6, 4.5)
            Size of the created figure. Measured in inches.
        show_grid : bool, defaults to True
            If `True` shows the horizontal grid with a step based on `y_ticker`.
        kwargs : misc, optional
            Additional keyword arguments to `Gather.plot`.
        """
        NMOCorrectionPlot(self, min_vel=min_vel, max_vel=max_vel, figsize=figsize, show_grid=show_grid,
                          **kwargs).plot()

    def plot_lmo_correction(self, min_vel=500, max_vel=3000, figsize=(6, 4.5), show_grid=True, **kwargs):
        """Perform interactive LMO correction of the gather with the selected velocity.

        The plot provides 2 views:
        * Corrected gather (default). LMO correction is performed on the fly with the velocity controlled by a slider
        on top of the plot.
        * Source gather. This view disables the velocity slider.

        Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        min_vel : float, optional, defaults to 500
            Minimum velocity value for LMO correction. Measured in meters/seconds.
        max_vel : float, optional, defaults to 3000
            Maximum velocity value for LMO correction. Measured in meters/seconds.
        figsize : tuple with 2 elements, optional, defaults to (6, 4.5)
            Size of the created figure. Measured in inches.
        show_grid : bool, defaults to True
            If `True` shows the horizontal grid with a step based on `y_ticker`.
        kwargs : misc, optional
            Additional keyword arguments to `Gather.plot`.
        """
        LMOCorrectionPlot(self, min_vel=min_vel, max_vel=max_vel, figsize=figsize, show_grid=show_grid,
                          **kwargs).plot()
