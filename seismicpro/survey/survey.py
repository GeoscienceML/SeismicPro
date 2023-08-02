"""Implements Survey class describing a single SEG-Y file"""

import os
import warnings
from textwrap import dedent
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import cv2
import segyio
import numpy as np
import scipy as sp
import pandas as pd
import polars as pl
from segfast import Loader
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from .headers_checks import validate_headers, validate_source_headers, validate_receiver_headers
from .metrics import (SurveyAttribute, TracewiseMetric, BaseWindowRMSMetric, MetricsRatio, DeadTrace,
                      DEFAULT_TRACEWISE_METRICS)
from .plot_geometry import SurveyGeometryPlot
from .utils import calculate_trace_stats
from ..config import config
from ..gather import Gather
from ..containers import GatherContainer, SamplesContainer
from ..metrics import initialize_metrics
from ..utils import to_list, maybe_copy, get_cols, get_first_defined, ForPoolExecutor
from ..const import HDR_TRACE_POS


class Survey(GatherContainer, SamplesContainer):  # pylint: disable=too-many-instance-attributes
    """A class representing a single SEG-Y file.

    In order to reduce memory footprint, `Survey` instance does not store trace data, but only a requested subset of
    trace headers and general file meta such as `samples` and `sample_rate`. Trace data can be obtained by generating
    an instance of `Gather` class by calling either :func:`~Survey.get_gather` or :func:`~Survey.sample_gather`
    method.

    The resulting gather type depends on `header_index` argument passed during `Survey` creation: traces are grouped
    into gathers by the common value of headers, defined by `header_index`. Some frequently used values of
    `header_index` are:
    - 'TRACE_SEQUENCE_FILE' - to get individual traces,
    - 'FieldRecord' - to get common source gathers,
    - ['GroupX', 'GroupY'] - to get common receiver gathers,
    - ['INLINE_3D', 'CROSSLINE_3D'] - to get common midpoint gathers.

    `header_cols` argument specifies all other trace headers to load to further be available in gather processing
    pipelines. All loaded headers are stored in the `headers` attribute as a `pd.DataFrame` with `header_index` columns
    set as its index.

    Values of both `header_index` and `header_cols` must be any of those specified in
    https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
    `UnassignedInt2` since they are treated differently from all other headers by `segyio`. Also, `TRACE_SEQUENCE_FILE`
    header is not loaded from the file but always automatically reconstructed.

    Sample interval of the survey is calculated by two values stored in:
    - bytes 3217-3218 of the binary header, called `Interval` in `segyio`,
    - bytes 117-118 of the trace header of the first trace in the file, called `TRACE_SAMPLE_INTERVAL` in `segyio`.
    If both of them are present and equal or only one of them is well-defined (non-zero), it is used as a sample rate.
    Otherwise, an error is raised.

    If `INLINE_3D` and `CROSSLINE_3D` trace headers are loaded, properties of survey binning are automatically inferred
    on survey construction which allows accessing:
    - Some bin-related attributes of the survey, such as `n_bins`,
    - `dist_to_bin_contours` method which calculates distances from points to a contour of the survey in bin
      coordinates.

    If `CDP_X` and `CDP_Y` headers are loaded together with `INLINE_3D` and `CROSSLINE_3D`, field geometry is also
    inferred which allows accessing:
    - Some geometry-related attributes of the survey, such as `area`, `perimeter` and `bin_size`,
    - `coords_to_bins` and `bins_to_coords` methods which convert geographic coordinates to bins and back respectively,
    - `dist_to_geographic_contours` method which calculates distances from points to a contour of the survey in
      geographic coordinates.

    Examples
    --------
    Create a survey of common source gathers and get a randomly selected gather from it:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"])
    >>> gather = survey.sample_gather()

    Parameters
    ----------
    path : str
        A path to the source SEG-Y file.
    header_index : str or list of str
        Trace headers to be used to group traces into gathers. Must be any of those specified in
        https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
        `UnassignedInt2`.
    header_cols : str or list of str or "all", optional
        Extra trace headers to load. Must be any of those specified in
        https://segyio.readthedocs.io/en/latest/segyio.html#trace-header-keys except for `UnassignedInt1` and
        `UnassignedInt2`. `TRACE_SEQUENCE_FILE` header is automatically reconstructed and always present in `headers`.
        If not given, only headers from `header_index`, `source_id_cols` and `receiver_id_cols` are loaded.
        If "all", all available headers are loaded.
    source_id_cols : str or list of str, optional
        Trace headers that uniquely identify a seismic source. If not given, set in the following way (in order of
        priority):
        - `FieldRecord` if it is loaded,
        - [`SourceX`, `SourceY`] if they are loaded,
        - `None` otherwise.
    receiver_id_cols : str or list of str, optional
        Trace headers that uniquely identify a receiver. If not given, set to [`GroupX`, `GroupY`] if they are loaded.
    name : str, optional
        Survey name. If not given, source file name is used. This name is mainly used to identify the survey when it is
        added to an index, see :class:`~index.SeismicIndex` docs for more info.
    sample_interval : float, optional
        Sample interval of seismic traces in the source SEG-Y file. Inferred from binary and trace headers if not
        given. Measured in milliseconds.
    delay : float, optional, defaults to 0
        Global delay recording time of seismic traces in the source SEG-Y file. Measured in milliseconds.
    limits : int or tuple or slice, optional
        Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
        used as arguments to init a `slice` object. If not given, whole traces are used. Measured in samples.
    validate : bool, optional, defaults to True
        Whether to perform validation of trace headers consistency.
    engine : {"segyio", "memmap"}, optional, defaults to "memmap"
        SEG-Y file loading engine. Two options are supported:
        - "segyio" - directly uses `segyio` library interface,
        - "memmap" - optimizes data fetching using `numpy` memory mapping. Generally provides up to 10x speedup
          compared to "segyio" and thus set as default.
    endian : {"big", "msb", "little", "lsb"}, optional, defaults to "big"
        SEG-Y file endianness.
    chunk_size : int, optional, defaults to 25000
        The number of traces to load by each of spawned processes.
    n_workers : int, optional
        The maximum number of simultaneously spawned processes to load trace headers. Defaults to the number of cpu
        cores.
    bar : bool, optional, defaults to True
        Whether to show trace headers loading progress bar.

    Attributes
    ----------
    path : str
        An absolute path to the source SEG-Y file.
    name : str
        Survey name.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    sample_interval : float
        Sample interval of seismic traces. Measured in milliseconds.
    delay : float
        Delay recording time of seismic traces. Measured in milliseconds.
    limits : slice
        Default time limits to be used during trace loading and survey statistics calculation. Measured in samples.
    source_id_cols : str or list of str or None
        Trace headers that uniquely identify a seismic source.
    receiver_id_cols : str or list of str or None
        Trace headers that uniquely identify a receiver.
    n_sources : int or None
        The number of sources in the survey. `None` if `source_id_cols` are undefined.
    n_receivers : int or None
        The number of receivers in the survey. `None` if `receiver_id_cols` are undefined.
    loader : segfast.SegyioLoader or segfast.MemmapLoader
        SEG-Y file loader. Its type depends on the `engine` passed during survey instantiation.
    has_stats : bool
        Whether the survey has trace statistics calculated. `False` until `collect_stats` method is called.
    min : np.float32 or None
        Minimum trace value. `None` until trace statistics are calculated.
    max : np.float32 or None
        Maximum trace value. `None` until trace statistics are calculated.
    mean : np.float32 or None
        Mean trace value. `None` until trace statistics are calculated.
    std : np.float32 or None
        Standard deviation of trace values. `None` until trace statistics are calculated.
    quantile_interpolator : scipy.interpolate.interp1d or None
        Interpolator of trace values quantiles. `None` until trace statistics are calculated.
    qc_metrics : dict
        A mapping from a name of a calculated tracewise QC metric to the corresponding metric instance.
        Empty until `qc` method is called.
    has_inferred_binning : bool
        Whether properties of survey binning have been inferred. `True` if `INLINE_3D` and `CROSSLINE_3D` trace headers
        are loaded on survey instantiation or `infer_binning` method is explicitly called.
    n_bins : int or None
        The number of bins in the survey. `None` until properties of survey binning are inferred.
    is_stacked : bool or None
        Whether the survey is stacked. `None` until properties of survey binning are inferred.
    field_mask : 2d np.ndarray or None
        A binary mask of the field with ones set for bins with at least one trace and zeros otherwise. The origin of
        the mask is stored in the `field_mask_origin` attribute. `None` until properties of survey binning are
        inferred.
    field_mask_origin : np.ndarray with 2 elements
        Minimum values of inline and crossline over the field. `None` until properties of survey binning are inferred.
    bin_contours : tuple of np.ndarray or None
        Contours of all connected components of the field in bin coordinates. `None` until properties of survey binning
        are inferred.
    has_inferred_geometry : bool
        Whether the survey has inferred geometry. `True` if `INLINE_3D`, `CROSSLINE_3D`, `CDP_X` and `CDP_Y` trace
        headers are loaded on survey instantiation or `infer_geometry` method is explicitly called.
    is_2d : bool or None
        Whether the survey is 2D. `None` until survey geometry is inferred.
    area : float or None
        Field area in squared meters. `None` until survey geometry is inferred.
    perimeter : float or None
        Field perimeter in meters. `None` until survey geometry is inferred.
    bin_size : np.ndarray with 2 elements or None
        Bin sizes in meters along inline and crossline directions. `None` until survey geometry is inferred.
    inline_length : float or None
        Maximum field length along inline direction in meters. `None` until survey geometry is inferred.
    crossline_length : float or None
        Maximum field length along crossline direction in meters. `None` until survey geometry is inferred.
    geographic_contours : tuple of np.ndarray or None
        Contours of all connected components of the field in geographic coordinates. `None` until survey geometry is
        inferred.
    """

    # pylint: disable-next=too-many-arguments, too-many-statements
    def __init__(self, path, header_index, header_cols=None, source_id_cols=None, receiver_id_cols=None, name=None,
                 sample_interval=None, delay=0, limits=None, validate=True, engine="memmap", endian="big",
                 chunk_size=25000, n_workers=None, bar=True):
        self.path = os.path.abspath(path)
        self.name = os.path.splitext(os.path.basename(self.path))[0] if name is None else name

        # Forbid loading UnassignedInt1 and UnassignedInt2 headers since they are treated differently from all other
        # headers by `segyio`
        allowed_headers = set(segyio.tracefield.keys.keys()) - {"UnassignedInt1", "UnassignedInt2"}
        header_index = to_list(header_index)
        if header_cols is None:
            header_cols = set()
        elif header_cols == "all":
            header_cols = allowed_headers
        else:
            header_cols = set(to_list(header_cols))
        headers_to_load = set(header_index) | header_cols

        # Parse source and receiver id cols and set defaults if needed
        if source_id_cols is None:
            if "FieldRecord" in headers_to_load:
                source_id_cols = "FieldRecord"
            elif {"SourceX", "SourceY"} <= headers_to_load:
                source_id_cols = ["SourceX", "SourceY"]
        else:
            headers_to_load |= set(to_list(source_id_cols))
        self.source_id_cols = source_id_cols

        if receiver_id_cols is None:
            if {"GroupX", "GroupY"} <= headers_to_load:
                receiver_id_cols = ["GroupX", "GroupY"]
        else:
            headers_to_load |= set(to_list(receiver_id_cols))
        self.receiver_id_cols = receiver_id_cols

        # Validate that only valid headers are being loaded
        unknown_headers = headers_to_load - allowed_headers
        if unknown_headers:
            raise ValueError(f"Unknown headers {', '.join(unknown_headers)}")

        # Open the SEG-Y file and set samples-related attributes of the file
        self.loader = Loader(self.path, engine=engine, endian=endian, ignore_geometry=True)
        sample_interval = get_first_defined(sample_interval, self.loader.sample_interval / 1000)
        if sample_interval <= 0:
            raise ValueError("Sample interval must be positive, please provide a valid sample_interval")
        self.file_samples = self.create_samples(self.loader.n_samples, sample_interval, delay)
        self.file_sample_interval = sample_interval
        self.file_delay = delay

        # Set samples and sample_rate according to passed `limits`
        self.limits = None
        self.samples = None
        self.sample_interval = None
        self.delay = None
        self.set_limits(limits)

        # Load trace headers and check them for consistency
        pbar = partial(tqdm, desc="Trace headers loaded") if bar else False
        headers = self.loader.load_headers(headers_to_load, reconstruct_tsf=True, sort_columns=True,
                                           chunk_size=chunk_size, max_workers=n_workers, pbar=pbar)
        if validate:
            validate_headers(pl.from_pandas(headers, rechunk=False), source_id_cols, receiver_id_cols)

        # Sort headers by the required index in order to optimize further subsampling and merging.
        # Sorting preserves trace order from the file within each gather.
        headers.set_index(header_index, inplace=True)
        headers.sort_index(kind="stable", inplace=True)
        self._headers = None
        self._indexer = None
        self.n_sources = None
        self.n_receivers = None
        self.headers = headers

        # Define all stats-related attributes
        self.has_stats = False
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.quantile_interpolator = None

        # calculated QC metrics
        self.qc_metrics = {}

        # Define all bin-related attributes and automatically infer them if both INLINE_3D and CROSSLINE_3D are loaded
        self.has_inferred_binning = False
        self.n_bins = None
        self.is_stacked = None
        self.field_mask = None
        self.field_mask_origin = None
        self.bin_contours = None
        if {"INLINE_3D", "CROSSLINE_3D"} <= headers_to_load:
            self.infer_binning()

        # Define all geometry-related attributes and automatically infer field geometry if required headers are loaded
        self.has_inferred_geometry = False
        self._bins_to_coords_reg = None
        self._coords_to_bins_reg = None
        self.is_2d = None
        self.area = None  # m^2
        self.perimeter = None  # m
        self.bin_size = None  # (m, m)
        self.inline_length = None  # m
        self.crossline_length = None  # m
        self.geographic_contours = None
        if {"INLINE_3D", "CROSSLINE_3D", "CDP_X", "CDP_Y"} <= headers_to_load:
            self.infer_geometry()

    @property
    def file_sample_rate(self):
        """float: Sample rate of seismic traces in the source SEG-Y file. Measured in Hz."""
        return 1000 / self.file_sample_interval

    @property
    def n_file_samples(self):
        """int: Trace length in samples in the source SEG-Y file."""
        return len(self.file_samples)

    @property
    def is_uphole(self):
        """bool or None: Whether the survey is uphole. `None` if uphole-related headers are not loaded."""
        has_uphole_times = "SourceUpholeTime" in self.available_headers
        has_uphole_depths = "SourceDepth" in self.available_headers
        has_positive_uphole_times = has_uphole_times and (self["SourceUpholeTime"] > 0).any()
        has_positive_uphole_depths = has_uphole_depths and (self["SourceDepth"] > 0).any()
        if not has_uphole_times and not has_uphole_depths:
            return None
        return has_positive_uphole_times or has_positive_uphole_depths

    @property
    def stats_summary(self):
        """str: Descriptive statistics of survey traces."""
        if not self.has_stats:
            raise ValueError("Global statistics were not calculated, call `Survey.collect_stats` first.")
        msg = f"""
        Survey statistics:
        mean | std:                {self.mean:>10.2f} | {self.std:<10.2f}
         min | max:                {self.min:>10.2f} | {self.max:<10.2f}
         q01 | q99:                {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        """
        return dedent(msg).strip()

    @property
    def qc_summary(self):
        """str: Brief report about calculated metrics."""
        if not self.qc_metrics:
            raise ValueError("Metrics were not calculated, call `Survey.qc` first.")
        summary = [metric.describe(self[metric.header_cols]) for metric in self.qc_metrics.values()]
        msg = "Tracewise QC summary:\n" + "\n".join(summary)
        return msg

    @GatherContainer.headers.setter
    def headers(self, headers):
        """Reconstruct trace positions on each headers assignment."""
        GatherContainer.headers.fset(self, headers)
        htp_dtype = np.int32 if len(headers) < np.iinfo(np.int32).max else np.int64
        self.headers[HDR_TRACE_POS] = np.arange(self.n_traces, dtype=htp_dtype)

        # Update the number of sources and receivers
        if self.source_id_cols is not None or self.receiver_id_cols is not None:
            polars_headers = self.get_polars_headers()
            if self.source_id_cols is not None:
                self.n_sources = len(polars_headers.select(self.source_id_cols).unique())
            if self.receiver_id_cols is not None:
                self.n_receivers = len(polars_headers.select(self.receiver_id_cols).unique())

    def __getstate__(self):
        """Create pickling state of a survey from its `__dict__`. Don't pickle `headers` and `indexer` if
        `enable_fast_pickling` config option is set."""
        state = self.__dict__.copy()
        if config["enable_fast_pickling"]:
            state["_headers"] = None
            state["_indexer"] = None
        return state

    def __str__(self):
        """Print survey metadata including information about the source file, field geometry if it was inferred and
        trace statistics if they were calculated."""
        offsets = self.headers.get('offset')
        offset_range = f"[{np.min(offsets)} m, {np.max(offsets)} m]" if offsets is not None else "Unknown"

        msg = f"""
        Survey path:               {self.path}
        Survey name:               {self.name}
        Survey size:               {os.path.getsize(self.path) / (1024**3):4.3f} GB

        Number of traces:          {self.n_traces}
        Trace length:              {self.n_samples} samples
        Sample interval:           {self.sample_interval} ms
        Sample rate:               {self.sample_rate} Hz
        Times range:               [{min(self.samples)} ms, {max(self.samples)} ms]
        Offsets range:             {offset_range}
        Is uphole:                 {get_first_defined(self.is_uphole, "Unknown")}

        Indexed by:                {", ".join(to_list(self.indexed_by))}
        Number of gathers:         {self.n_gathers}
        Mean gather fold:          {int(self.n_traces / self.n_gathers)}
        """

        if self.has_inferred_binning:
            msg += f"""
        Is stacked:                {self.is_stacked}
        Number of bins:            {self.n_bins}
        Mean bin fold:             {int(self.n_traces / self.n_bins)}
        """

        if self.source_id_cols is not None:
            msg += f"""
        Source ID headers:         {", ".join(to_list(self.source_id_cols))}
        Number of sources:         {self.n_sources}
        Mean source fold:          {int(self.n_traces / self.n_sources)}
        """

        if self.receiver_id_cols is not None:
            msg += f"""
        Receiver ID headers:       {", ".join(to_list(self.receiver_id_cols))}
        Number of receivers:       {self.n_receivers}
        Mean receiver fold:        {int(self.n_traces / self.n_receivers)}
        """

        if self.has_inferred_geometry:
            msg += f"""
        Field geometry:
        Dimensionality:            {"2D" if self.is_2d else "3D"}
        Area:                      {(self.area / 1000**2):.2f} km^2
        Perimeter:                 {(self.perimeter / 1000):.2f} km
        Inline bin size:           {self.bin_size[0]:.1f} m
        Crossline bin size:        {self.bin_size[1]:.1f} m
        Inline length:             {(self.inline_length / 1000):.2f} km
        Crossline length:          {(self.crossline_length / 1000):.2f} km
        """
        msg = dedent(msg).strip()
        if self.has_stats:
            msg += "\n\n" + self.stats_summary

        if self.qc_metrics:
            msg += "\n\n" + self.qc_summary
        return msg

    def info(self):
        """Print survey metadata including information about the source file, field geometry if it was inferred and
        trace statistics if they were calculated."""
        print(self)

    def get_polars_headers(self):
        """Return survey trace headers as a `polars.DataFrame`. The index is transformed into individual columns."""
        return pl.from_pandas(self.headers, rechunk=False, include_index=True)

    def set_source_id_cols(self, cols, validate=True):
        """Set new trace headers that uniquely identify a seismic source and optionally validate consistency of
        source-related trace headers by checking that each source has unique coordinates, surface elevation, uphole
        time and depth."""
        if set(to_list(cols)) - self.available_headers:
            raise ValueError("Required headers were not loaded")
        polars_headers = self.get_polars_headers()
        if validate:
            validate_source_headers(polars_headers, cols)
        self.source_id_cols = cols
        self.n_sources = len(polars_headers.select(cols).unique())

    def set_receiver_id_cols(self, cols, validate=True):
        """Set new trace headers that uniquely identify a receiver and optionally validate consistency of
        receiver-related trace headers by checking that each receiver has unique coordinates and surface elevation."""
        if set(to_list(cols)) - self.available_headers:
            raise ValueError("Required headers were not loaded")
        polars_headers = self.get_polars_headers()
        if validate:
            validate_receiver_headers(polars_headers, cols)
        self.receiver_id_cols = cols
        self.n_receivers = len(polars_headers.select(cols).unique())

    def validate_headers(self, offset_atol=10, cdp_atol=10, elevation_atol=5, elevation_radius=50):
        """Check trace headers for consistency.

        1. Validate trace headers by checking that:
           - All headers are not empty,
           - Trace identifier (FieldRecord, TraceNumber) has no duplicates,
           - Source uphole times and depths are non-negative,
           - Source uphole time is zero if and only if source depth is also zero,
           - Traces do not have signed offsets,
           - Offsets in trace headers coincide with distances between sources (SourceX, SourceY) and receivers (GroupX,
             GroupY),
           - Coordinates of a midpoint (CDP_X, CDP_Y) matches those of the corresponding source (SourceX, SourceY) and
             receiver (GroupX, GroupY),
           - Surface elevation is unique for a given spatial location,
           - Elevation-related headers (ReceiverGroupElevation, SourceSurfaceElevation) have consistent ranges,
           - Mapping from geographic (CDP_X, CDP_Y) to line-based (INLINE_3D, CROSSLINE_3D) coordinates and back is
             unique.

        2. Validate consistency of source-related trace headers by checking that each source has unique coordinates,
           surface elevation, uphole time and depth.

        3. Validate consistency of receiver-related trace headers by checking that each receiver has unique coordinates
           and surface elevation.

        If any of the checks fail, a warning is displayed.

        Parameters
        ----------
        offset_atol : int, optional, defaults to 10
            Maximum allowed difference between a trace offset and the distance between its source and receiver.
        cdp_atol : int, optional, defaults to 10
            Maximum allowed difference between coordinates of a trace CDP and the midpoint between its source and
            receiver.
        elevation_atol : int, optional, defaults to 5
            Maximum allowed difference between surface elevation at a given source/receiver location and mean elevation
            of all sources and receivers within a radius defined by `elevation_radius`.
        elevation_radius : int, optional, defaults to 50
            Radius of the neighborhood to estimate mean surface elevation.
        """
        validate_headers(self.get_polars_headers(), self.source_id_cols, self.receiver_id_cols,
                         offset_atol=offset_atol, cdp_atol=cdp_atol, elevation_atol=elevation_atol,
                         elevation_radius=elevation_radius)

    #------------------------------------------------------------------------#
    #                        Geometry-related methods                        #
    #------------------------------------------------------------------------#

    def infer_binning(self):
        """Infer properties of survey binning by estimating the following entities:
        1. Number of bins,
        2. Pre- or post-stack flag,
        3. Binary mask of the field and its origin,
        4. Field contours in bin coordinate system.

        After the method is executed `has_inferred_binning` flag is set to `True` and all the calculated values can be
        obtained via corresponding attributes.
        """
        # Find unique pairs of inlines and crosslines
        lines = pl.from_pandas(self.get_headers(["INLINE_3D", "CROSSLINE_3D"]), rechunk=False).unique().to_numpy()

        # Construct a binary mask of a field where True value is set for bins containing at least one trace
        # and False otherwise
        origin = lines.min(axis=0)
        normed_lines = lines - origin
        field_mask = np.zeros(normed_lines.max(axis=0) + 1, dtype=np.uint8)
        field_mask[normed_lines[:, 0], normed_lines[:, 1]] = 1
        bin_contours = cv2.findContours(field_mask.T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=origin)[0]

        # Set all bin-related attributes
        self.has_inferred_binning = True
        self.n_bins = len(lines)
        self.is_stacked = self.n_traces == self.n_bins
        self.field_mask = field_mask
        self.field_mask_origin = origin
        self.bin_contours = bin_contours

    def infer_geometry(self):
        """Infer survey geometry by estimating the following entities:
        1. Survey dimensionality (2D/3D),
        2. Bin sizes along inline and crossline directions,
        3. Survey lengths along inline and crossline directions,
        4. Survey area and perimeter,
        5. Field contours in geographic coordinate system,
        6. Mappings from geographic coordinates to bins and back.

        After the method is executed `has_inferred_geometry` flag is set to `True` and all the calculated values can be
        obtained via corresponding attributes.
        """
        coords_cols = ["CDP_X", "CDP_Y"]
        bins_cols = ["INLINE_3D", "CROSSLINE_3D"]

        # Construct a mapping from bins to their coordinates and back
        bins_to_coords = pl.from_pandas(self.get_headers(coords_cols + bins_cols), rechunk=False)
        bins_to_coords = bins_to_coords.groupby(bins_cols).agg(pl.col(coords_cols).mean()).to_pandas()
        bins = bins_to_coords[bins_cols].to_numpy()
        coords = bins_to_coords[coords_cols].to_numpy()
        bins_to_coords_reg = LinearRegression(copy_X=False, n_jobs=-1).fit(bins, coords)
        coords_to_bins_reg = LinearRegression(copy_X=False, n_jobs=-1).fit(coords, bins)

        # Compute geographic field contour
        geographic_contours = tuple(bins_to_coords_reg.predict(contour[:, 0])[:, None].astype(np.float32)
                                    for contour in self.bin_contours)
        perimeter = sum(cv2.arcLength(contour, closed=True) for contour in geographic_contours)

        # Set all geometry-related attributes
        self.has_inferred_geometry = True
        self._bins_to_coords_reg = bins_to_coords_reg
        self._coords_to_bins_reg = coords_to_bins_reg
        self.bin_size = np.diag(sp.linalg.polar(bins_to_coords_reg.coef_)[1])
        self.inline_length = (np.ptp(bins_to_coords["INLINE_3D"]) + 1) * self.bin_size[0]
        self.crossline_length = (np.ptp(bins_to_coords["CROSSLINE_3D"]) + 1) * self.bin_size[1]
        self.area = self.n_bins * np.prod(self.bin_size)
        self.perimeter = perimeter
        self.geographic_contours = geographic_contours
        self.is_2d = np.isclose(self.area, 0)

    @staticmethod
    def _cast_coords(coords, transformer):
        """Linearly convert `coords` from one coordinate system to another according to a passed `transformer`."""
        if transformer is None:
            raise ValueError("Survey geometry was not inferred, call `infer_geometry` method first.")
        coords = np.array(coords)
        is_coords_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        transformed_coords = transformer.predict(coords)
        if is_coords_1d:
            return transformed_coords[0]
        return transformed_coords

    def coords_to_bins(self, coords):
        """Convert `coords` from geographic coordinate system to floating-valued bins.

        Notes
        -----
        Before calling this method, survey geometry must be inferred using :func:`~Survey.infer_geometry`.

        Parameters
        ----------
        coords : array-like with 2 elements or 2d array-like with shape (n_coords, 2)
            Geographic coordinates to be converted to bins.

        Returns
        -------
        bins : np.ndarray with 2 elements or 2d np.ndarray with shape (n_coords, 2)
            Floating-valued bin for each coordinate from `coords`. Has the same shape as `coords`.

        Raises
        ------
        ValueError
            If survey geometry was not inferred.
        """
        return self._cast_coords(coords, self._coords_to_bins_reg)

    def bins_to_coords(self, bins):
        """Convert `bins` to coordinates in geographic coordinate system.

        Notes
        -----
        Before calling this method, survey geometry must be inferred using :func:`~Survey.infer_geometry`.

        Parameters
        ----------
        bins : array-like with 2 elements or 2d array-like with shape (n_bins, 2)
            Bins to be converted to geographic coordinates.

        Returns
        -------
        coords : np.ndarray with 2 elements or 2d np.ndarray with shape (n_bins, 2)
            Floating-valued geographic coordinates for each bin from `bins`. Has the same shape as `bins`.

        Raises
        ------
        ValueError
            If survey geometry was not inferred.
        """
        return self._cast_coords(bins, self._bins_to_coords_reg)

    @staticmethod
    def _dist_to_contours(coords, contours):
        """Calculate minimum signed distance from points in `coords` to each contour in `contours`."""
        coords = np.array(coords, dtype=np.float32)
        is_coords_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        dist = np.empty(len(coords), dtype=np.float32)
        for i, coord in enumerate(coords):
            dists = [cv2.pointPolygonTest(contour, coord, measureDist=True) for contour in contours]
            dist[i] = dists[np.abs(dists).argmin()]
        if is_coords_1d:
            return dist[0]
        return dist

    def dist_to_geographic_contours(self, coords):
        """Calculate signed distances from each of `coords` to the field contour in geographic coordinate system.

        Returned values may by positive (inside the contour), negative (outside the contour) or zero (on an edge).

        Notes
        -----
        Before calling this method, survey geometry must be inferred using :func:`~Survey.infer_geometry`.

        Parameters
        ----------
        coords : array-like with 2 elements or 2d array-like with shape (n_coords, 2)
            Geographic coordinates to estimate distance to field contour for.

        Returns
        -------
        dist : np.float32 or np.ndarray with shape (n_coords,)
            Signed distances from each of `coords` to the field contour in geographic coordinate system. Matches the
            length of `coords`.

        Raises
        ------
        ValueError
            If survey geometry was not inferred.
        """
        if not self.has_inferred_geometry:
            raise ValueError("Survey geometry was not inferred, call `infer_geometry` method first.")
        return self._dist_to_contours(coords, self.geographic_contours)

    def dist_to_bin_contours(self, bins):
        """Calculate signed distances from each of `bins` to the field contour in bin coordinate system.

        Returned values may by positive (inside the contour), negative (outside the contour) or zero (on an edge).

        Notes
        -----
        Before calling this method, properties of survey binning must be inferred using :func:`~Survey.infer_binning`.

        Parameters
        ----------
        bins : array-like with 2 elements or 2d array-like with shape (n_bins, 2)
            Bin coordinates to estimate distance to field contour for.

        Returns
        -------
        dist : np.float32 or np.ndarray with shape (n_bins,)
            Signed distances from each of `bins` to the field contour in bin coordinate system. Matches the length of
            `coords`.

        Raises
        ------
        ValueError
            If properties of survey binning were not inferred.
        """
        if not self.has_inferred_binning:
            raise ValueError("Properties of survey binning were not inferred, call `infer_binning` method first.")
        return self._dist_to_contours(bins, self.bin_contours)

    #------------------------------------------------------------------------#
    #                     Statistics computation methods                     #
    #------------------------------------------------------------------------#

    # pylint: disable-next=too-many-statements
    def collect_stats(self, indices=None, n_quantile_traces=100000, quantile_precision=2, limits=None,
                      chunk_size=10000, n_workers=None, bar=True, verbose=False):
        """Collect the following statistics by iterating over survey traces:
        1. Min and max amplitude,
        2. Mean amplitude and trace standard deviation,
        3. Approximation of trace data quantiles with given precision.

        Since fair quantile calculation requires simultaneous loading of all traces from the file we avoid such memory
        overhead by calculating approximate quantiles for a small subset of `n_quantile_traces` traces selected
        randomly. Only a set of quantiles defined by `quantile_precision` is calculated, the rest of them are linearly
        interpolated by the collected ones.

        After the method is executed `has_stats` flag is set to `True` and all the calculated values can be obtained
        via corresponding attributes.

        Parameters
        ----------
        indices : pd.Index, optional
            A subset of survey headers indices to collect stats for. If not given, statistics are calculated for the
            whole survey.
        n_quantile_traces : positive int, optional, defaults to 100000
            The number of traces to use for quantiles estimation.
        quantile_precision : positive int, optional, defaults to 2
            Calculate an approximate quantile for each q with `quantile_precision` decimal places. All other quantiles
            will be linearly interpolated on request.
        limits : int or tuple or slice, optional
            Time limits to be used for statistics calculation. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, `limits` passed to `__init__` are used. Measured in samples.
        chunk_size : int, optional, defaults to 1000
            The number of traces to be processed at once.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.
        verbose : bool, optional, defaults to False
            Whether to print collected statistics.

        Returns
        -------
        survey : Survey
            The survey with collected stats. Sets `has_stats` flag to `True` and updates statistics attributes inplace.
        """

        headers = self.headers
        if indices is not None:
            headers = self.get_headers_by_indices(indices)
        n_traces = len(headers)
        limits = get_first_defined(limits, self.limits)

        if n_quantile_traces < 0:
            raise ValueError("n_quantile_traces must be non-negative")
        # Clip n_quantile_traces if it's greater than the total number of traces
        n_quantile_traces = min(n_traces, n_quantile_traces)

        # Sort traces by TRACE_SEQUENCE_FILE: sequential access to trace amplitudes is much faster than random
        traces_pos = np.sort(get_cols(headers, "TRACE_SEQUENCE_FILE").to_numpy() - 1)
        quantile_traces_mask = np.zeros(n_traces, dtype=np.bool_)
        quantile_traces_mask[np.random.choice(n_traces, size=n_quantile_traces, replace=False)] = True

        # Split traces by chunks
        n_chunks, last_chunk_size = divmod(n_traces, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            n_chunks += 1
            chunk_sizes += [last_chunk_size]
        chunk_borders = np.cumsum(chunk_sizes[:-1])
        chunk_traces_pos = np.split(traces_pos, chunk_borders)
        chunk_quantile_traces_mask = np.split(quantile_traces_mask, chunk_borders)

        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)

        # Define buffers. chunk_mean, chunk_var and chunk_weights have float64 dtype to be numerically stable
        quantile_traces_buffer = [[] for _ in range(n_chunks)]
        min_buffer = np.empty(n_chunks, dtype=np.float32)
        max_buffer = np.empty(n_chunks, dtype=np.float32)
        mean_buffer = np.empty(n_chunks, dtype=np.float64)
        var_buffer = np.empty(n_chunks, dtype=np.float64)
        chunk_weights = np.array(chunk_sizes, dtype=np.float64) / n_traces

        def collect_chunk_stats(i):
            chunk = self.load_traces(chunk_traces_pos[i], limits=limits)
            chunk_quantile_mask = chunk_quantile_traces_mask[i]
            if chunk_quantile_mask.any():
                quantile_traces_buffer[i] = chunk[chunk_quantile_mask].ravel()
            min_buffer[i], max_buffer[i], mean_buffer[i], var_buffer[i] = calculate_trace_stats(chunk.ravel())
            return len(chunk)

        # Precompile njitted function to correctly initialize TBB from the main thread
        _ = calculate_trace_stats(self.load_traces([0], limits=limits).ravel())

        # Accumulate min, max, mean and var values of traces chunks
        bar_desc = f"Calculating statistics for traces in survey {self.name}"
        with tqdm(total=n_traces, desc=bar_desc, disable=not bar) as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    future = pool.submit(collect_chunk_stats, i)
                    future.add_done_callback(lambda fut: pbar.update(fut.result()))

        # Calculate global survey statistics by individual chunks
        global_min = np.min(min_buffer)
        global_max = np.max(max_buffer)
        global_mean = np.average(mean_buffer, weights=chunk_weights)
        global_var = np.average(var_buffer + (mean_buffer - global_mean)**2, weights=chunk_weights)

        # Cast all calculated statistics to float32
        self.min = np.float32(global_min)
        self.max = np.float32(global_max)
        self.mean = np.float32(global_mean)
        self.std = np.float32(np.sqrt(global_var))

        if n_quantile_traces == 0:
            q = [0, 1]
            quantiles = [self.min, self.max]
        else:
            # Calculate all q-quantiles from 0 to 1 with step 1 / 10**quantile_precision
            q = np.round(np.linspace(0, 1, num=10**quantile_precision), decimals=quantile_precision)
            quantiles = np.nanquantile(np.concatenate(quantile_traces_buffer), q=q)
            # 0 and 1 quantiles are replaced with actual min and max values respectively
            quantiles[0], quantiles[-1] = self.min, self.max
        self.quantile_interpolator = interp1d(q, quantiles)

        self.has_stats = True

        if verbose:
            print(self.stats_summary)
        return self

    def get_quantile(self, q):
        """Calculate an approximation of the `q`-th quantile of the survey data.

        Notes
        -----
        Before calling this method, survey statistics must be calculated using :func:`~Survey.collect_stats`.

        Parameters
        ----------
        q : float or array-like of floats
            Quantile or a sequence of quantiles to compute, which must be between 0 and 1 inclusive.

        Returns
        -------
        quantile : float or array-like of floats
            Approximate `q`-th quantile values. Has the same shape as `q`.

        Raises
        ------
        ValueError
            If survey statistics were not calculated.
        """
        if not self.has_stats:
            raise ValueError('Global statistics were not calculated, call `Survey.collect_stats` first.')
        quantiles = self.quantile_interpolator(q).astype(np.float32)
        # return the same type as q: either single float or array-like
        return quantiles.item() if quantiles.ndim == 0 else quantiles

    #------------------------------------------------------------------------#
    #                            Loading methods                             #
    #------------------------------------------------------------------------#

    def load_traces(self, indices, limits=None, buffer=None, chunk_size=None, n_workers=None,
                    return_samples_info=False):
        """Load seismic traces by their indices.

        Parameters
        ----------
        indices : 1d array-like
            Indices of the traces to read.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        buffer : 2d np.ndarray, optional
            Buffer to read the data into. Created automatically if not given.
        chunk_size : int, optional
            The number of traces to load by each of spawned threads. Loads all traces in the main thread by default.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.
        return_samples_info : bool
            Whether to also return sample interval and delay recording time of loaded traces.

        Returns
        -------
        traces : 2d np.ndarray
            Loaded seismic traces.
        sample_interval : float
            Sample interval of loaded seismic traces. Returned only if `return_samples_info` is `True`.
        delay : float
            Delay recording time of loaded seismic traces. Returned only if `return_samples_info` is `True`.
        """
        if chunk_size is None:
            chunk_size = len(indices)
        n_chunks, last_chunk_size = divmod(len(indices), chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            n_chunks += 1
            chunk_sizes += [last_chunk_size]
        chunk_borders = np.cumsum([0] + chunk_sizes)

        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)
        executor_class = ForPoolExecutor if n_workers == 1 else ThreadPoolExecutor

        limits, n_samples, sample_interval, delay = self._get_limits_info(get_first_defined(limits, self.limits))
        if buffer is None:
            buffer = np.empty((len(indices), n_samples), dtype=self.loader.dtype)

        with executor_class(max_workers=n_workers) as pool:
            for start, end in zip(chunk_borders[:-1], chunk_borders[1:]):
                pool.submit(self.loader.load_traces, indices[start:end], limits=limits, buffer=buffer[start:end])

        if return_samples_info:
            return buffer, sample_interval, delay
        return buffer

    def load_gather(self, headers, limits=None, copy_headers=False, chunk_size=None, n_workers=None):
        """Load a gather with given `headers`.

        Parameters
        ----------
        headers : pd.DataFrame
            Headers of traces to load. Must be a subset of `self.headers`.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the passed `headers` when instantiating the gather.
        chunk_size : int, optional
            The number of traces to load by each of spawned threads. Loads all traces in the main thread by default.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        if copy_headers:
            headers = headers.copy()
        indices = get_cols(headers, "TRACE_SEQUENCE_FILE").to_numpy() - 1
        data, sample_interval, delay = self.load_traces(indices, limits=limits, chunk_size=chunk_size,
                                                        n_workers=n_workers, return_samples_info=True)
        return Gather(headers=headers, data=data, sample_interval=sample_interval, delay=delay, survey=self)

    def get_gather(self, index, limits=None, copy_headers=False, chunk_size=None, n_workers=None):
        """Load a gather with given `index`.

        Parameters
        ----------
        index : int or 1d array-like
            An index of the gather to load. Must be one of `self.indices`.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of survey `headers` describing the gather.
        chunk_size : int, optional
            The number of traces to load by each of spawned threads. Loads all traces in the main thread by default.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        return self.load_gather(self.get_headers_by_indices((index,)), limits=limits, copy_headers=copy_headers,
                                chunk_size=chunk_size, n_workers=n_workers)

    def sample_gather(self, limits=None, copy_headers=False, chunk_size=None, n_workers=None):
        """Load a gather with random index.

        Parameters
        ----------
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of survey `headers` describing the sampled gather.
        chunk_size : int, optional
            The number of traces to load by each of spawned threads. Loads all traces in the main thread by default.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.

        Returns
        -------
        gather : Gather
            Loaded gather instance.
        """
        return self.get_gather(index=np.random.choice(self.indices), limits=limits, copy_headers=copy_headers,
                               chunk_size=chunk_size, n_workers=n_workers)

    #------------------------------------------------------------------------#
    #                       Survey processing methods                        #
    #------------------------------------------------------------------------#

    def _get_limits_info(self, limits):
        """Convert given `limits` to a `slice` and return it together with the number of samples, sample interval and
        delay recording time these limits imply."""
        limits = self.loader.process_limits(limits)
        samples = self.file_samples[limits]
        return limits, len(samples), self.file_sample_interval * limits.step, samples[0]

    def set_limits(self, limits):
        """Update default survey time limits that are used during trace loading and statistics calculation.

        Parameters
        ----------
        limits : int or tuple or slice
            Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
            used as arguments to init a `slice`. The resulting object is stored in `self.limits` attribute and used to
            recalculate `self.samples`, `self.sample_interval` and `self.delay`. Measured in samples.

        Raises
        ------
        ValueError
            If negative step of limits was passed.
            If the resulting samples length is zero.
        """
        self.limits, _, self.sample_interval, self.delay = self._get_limits_info(limits)
        self.samples = self.file_samples[self.limits]

    def filter_by_metric(self, metric_name, threshold=None, inplace=False, bad_only=False):
        """Filter traces using metric with name `metric_name` and passed `threshold`.

        Parameters
        ----------
        metric_name : str
            Name of the metric that is stored in `self.qc_metrics`.
        threshold : int, float, array-like with 2 elements, optional, defaults to None
            Threshold to use during filtration, see :func:`~metric.TracewiseMetric.binarize` docs for more info.
            If None, threshold defined in metric will be used.
        inplace : bool, optional, defaults to False
            Whether to transform the survey inplace or process its copy.
        bad_only : bool, optional, defaults to False
            If True, keep only traces that marked as `bad` by the metric.
            Otherwise, keep traces approved by the metric.

        Returns
        -------
        Survey
            Filtered survey.
        """
        if metric_name not in self.qc_metrics:
            raise ValueError(f"Metric with name {metric_name} has not been calculated yet.")

        metric = self.qc_metrics[metric_name]
        def binarize(metric_value):
            bin_mask = metric.binarize(metric_value, threshold)
            return bin_mask if bad_only else ~bin_mask

        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        self.filter(binarize, cols=metric_name, inplace=True)
        return self

    def remove_dead_traces(self, header_name=None, chunk_size=1000, n_workers=None, inplace=False, bar=True):
        """ Remove dead (constant) traces from the survey.
        Calculates :class:`~survey.metrics.DeadTrace` if it was not calculated.

        Parameters
        ----------
        header_name : str, optional, defaults to None
            Name of the header column with marked dead traces.
        chunk_size : int, optional, defaults to 1000
            Number of traces loaded on each iteration.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to find and remove dead traces. Defaults to the
            number of cpu cores.
        inplace : bool, optional, defaults to False
            Whether to transform the survey inplace or process its copy.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        Survey
            Survey with no dead traces.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        if header_name is None:
            header_name = DeadTrace.__name__
            if header_name not in self.headers:
                self.qc(DeadTrace, chunk_size=chunk_size, n_workers=n_workers, bar=bar)

        self.filter_by_metric(header_name, inplace=True)
        return self

    #------------------------------------------------------------------------#
    #                         Task specific methods                          #
    #------------------------------------------------------------------------#

    @staticmethod
    def _get_optimal_origin(arr, step):
        """Find a position in an array `arr` that maximizes sum of each `step`-th element from it to the end of the
        array. In case of multiple such positions, return the one closer to `step // 2`."""
        mod = len(arr) % step
        if mod:
            arr = np.pad(arr, (0, step - mod))
        step_sums = arr.reshape(-1, step).sum(axis=0)
        max_indices = np.nonzero(step_sums == step_sums.max())[0]
        return max_indices[np.abs(max_indices - step // 2).argmin()]

    def generate_supergathers(self, centers=None, origin=None, size=3, step=20, border_indent=0, strict=True,
                              reindex=True, inplace=False):
        """Combine several adjacent CDP gathers into ensembles called supergathers.

        Supergather generation is usually performed as a first step of velocity analysis. A substantially larger number
        of traces processed at once leads to increased signal-to-noise ratio: seismic wave reflections are much more
        clearly visible than on single CDP gathers and the velocity spectra calculated using
        :func:`~Gather.calculate_vertical_velocity_spectrum` are more coherent
        which allows for more accurate stacking velocity picking.

        The method creates two new `headers` columns called `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D`
        equal to `INLINE_3D` and `CROSSLINE_3D` of the central CDP gather. Note, that some gathers may be assigned to
        several supergathers at once and their traces will become duplicated in `headers`.

        Parameters
        ----------
        centers : 2d array-like with shape (n_supergathers, 2), optional
            Centers of supergathers being generated. If not given, calculated by the `origin` of a supergather grid.
            Measured in lines.
        origin : int or tuple of 2 ints, optional
            Origin of the supergather grid, used only if `centers` are not given. If `None`, generated automatically to
            maximize the number of supergathers. Measured in lines.
        size : int or tuple of 2 ints, optional, defaults to 3
            Supergather size along inline and crossline axes. Single int defines sizes for both axes. Measured in
            lines.
        step : int or tuple of 2 ints, optional, defaults to 20
            Supergather step along inline and crossline axes. Single int defines steps for both axes. Used to define a
            grid of supergathers if `centers` are not given. Measured in lines.
        border_indent : int, optional, defaults to 0
            Avoid placing supergather centers closer than this distance to the field contour. Used only if `centers`
            are not given. Measured in lines.
        strict : bool, optional, defaults to True
            If `True`, guarantees that each gather in a generated supergather will have at least one trace or, in other
            words, that the supergather entirely lies within the field. Used only if `centers` are not given.
        reindex : bool, optional, defaults to True
            Whether to reindex a survey with the created `SUPERGATHER_INLINE_3D` and `SUPERGATHER_CROSSLINE_3D` headers
            columns.
        inplace : bool, optional, defaults to False
            Whether to transform the survey inplace or process its copy.

        Returns
        -------
        survey : Survey
            A survey with generated supergathers.

        Raises
        ------
        KeyError
            If `INLINE_3D` and `CROSSLINE_3D` headers were not loaded.
        ValueError
            If supergathers have already been generated.
        """
        line_cols = ["INLINE_3D", "CROSSLINE_3D"]
        super_line_cols = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        if set(super_line_cols) <= self.available_headers:
            raise ValueError("Supergathers have already been generated")
        if not self.has_inferred_binning:
            if set(line_cols) <= self.available_headers:
                self.infer_binning()
            else:
                raise KeyError("INLINE_3D and CROSSLINE_3D headers must be loaded")

        new_index = super_line_cols if reindex else self.indexed_by
        self = maybe_copy(self, inplace, ignore="headers")  # pylint: disable=self-cls-assignment
        size = np.broadcast_to(size, 2)
        step = np.broadcast_to(step, 2)

        if centers is None:
            # Erode the field mask according to border_indent and strict flags
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, np.broadcast_to(border_indent, 2) * 2 + 1).T
            field_mask = cv2.erode(self.field_mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if strict:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size).T
                field_mask = cv2.erode(field_mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            step = np.minimum(step, field_mask.shape)

            # Calculate origins of the supergather grid along inline and crossline directions
            if origin is not None:
                origin_i, origin_x = (np.broadcast_to(origin, 2) - self.field_mask_origin) % step
            else:
                origin_i = self._get_optimal_origin(field_mask.sum(axis=1), step[0])
                origin_x = self._get_optimal_origin(field_mask.sum(axis=0), step[1])

            # Calculate supergather centers by their grid
            grid_i = np.arange(origin_i, field_mask.shape[0], step[0])
            grid_x = np.arange(origin_x, field_mask.shape[1], step[1])
            centers = np.stack(np.meshgrid(grid_i, grid_x), -1).reshape(-1, 2)
            is_valid = field_mask[centers[:, 0], centers[:, 1]].astype(bool)
            centers = centers[is_valid] + self.field_mask_origin

        centers = np.array(centers)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("Passed centers must have shape (n_supergathers, 2)")

        polars_headers = self.get_polars_headers()

        # Construct a bridge table with mapping from supergather centers to their bins
        shifts_grid = np.meshgrid(np.arange(size[0]) - size[0] // 2, np.arange(size[1]) - size[1] // 2)
        shifts = np.stack(shifts_grid, axis=-1).reshape(-1, 2)
        bridge = np.column_stack([centers.repeat(size.prod(), axis=0), (centers[:, None] + shifts).reshape(-1, 2)])
        bridge_schema = [
            ("SUPERGATHER_INLINE_3D", polars_headers.schema["INLINE_3D"]),
            ("SUPERGATHER_CROSSLINE_3D", polars_headers.schema["CROSSLINE_3D"]),
            ("INLINE_3D", polars_headers.schema["INLINE_3D"]),
            ("CROSSLINE_3D", polars_headers.schema["CROSSLINE_3D"]),
        ]
        bridge = pl.from_numpy(bridge, schema=bridge_schema, orient="row")

        headers = polars_headers.join(bridge, on=line_cols, how="inner").sort(new_index).to_pandas()
        headers.set_index(new_index, inplace=True)
        self.headers = headers
        return self

    # pylint: disable-next=invalid-name
    def qc(self, metrics=None, chunk_size=1000, n_workers=None, bar=True, overwrite=True, verbose=False):
        """Perform quality control of the traces in the survey.

        The following metrics are calculated for each trace by default:
        * A boolean indicator of a dead trace,
        * Absolute value of the trace's mean scaled by trace's std,
        * Maximum absolute amplitude value scaled by trace's std,
        * The maximum number of consecutive values clipped with either minimum or maximum trace amplitude,
        * The maximum number of consecutive identical amplitudes.

        The metrics are calculated in parallel threads each processing no more than `chunk_size` traces. The resulting
        values are stored in `self.headers` under the name defined by `metric.header_cols` which usually matches the
        name of the metric class.

        Some metrics, however, store not their values but some intermediate results which will be aggregated into the
        actual metric value upon `construct_qc_map` call. For example, metrics that compute window-based RMS amplitudes
        create two columns in `headers`: one with the sum of squared amplitudes and another with the number of
        amplitudes in the window for each trace. This allows constructing RMS maps aggregated by different types of
        gathers (e.g. sources or receivers) without metric recalculation.

        Examples
        --------
        Define metrics to calculate:
            - Indicator of a dead (constant) trace:
            >>> dead_trace = DeadTrace()
            - Spike indicator with required `muter` parameter:
            >>> spikes = Spikes(muter=muter)
            - Maximum absolute amplitude value divided by trace's std, note that a metric can be defined directly by
              the metric class, not an instance:
            >>> max_abs = TraceMaxAbs
            - RMS in a window located in a signal zone with required window parameters and metric name:
            >>> signal_rms = WindowRMS(offsets=[650, 2000], times=[1000, 1400], name="SignalWindowRMS")
            - RMS in a window located in a noise zone with required parameters:
            >>> noise_rms = WindowRMS(offsets=[650, 2000], times=[100, 500], name="NoiseWindowRMS")

        Compute provided metrics:
        >>> survey.qc([dead_trace, spikes, max_abs, signal_rms, noise_rms])

        It is not necessary to define instance of tracewise metric. For metrics that do not have required arguments one
        may use only class name:
        >>> survey.qc(TraceMaxAbs)

        Parameters
        ----------
        metrics : TracewiseMetric or array-like of TracewiseMetric, optional
            Metrics to calculate. If `None`, metrics listed in `DEFAULT_TRACEWISE_METRICS` are calculated.
        chunk_size : int, optional, defaults to 1000
            Number of traces processed in one thread.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to compute metrics. Defaults to the number of cpu
            cores.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.
        overwrite : bool, optional, defaults to True
            Whether to rewrite metrics that were previously calculated or skip them.
        verbose : bool, optional, defaults to False
            Whether to print QC results.

        Returns
        -------
        Survey
            Survey with metrics stored in `self.qc_metrics` and metrics values in `self.headers`.

        Raises
        ------
        ValueError
            If `overwrite` is `False` and any of the given metrics were previously calculated.
        """
        if metrics is None:
            metrics = DEFAULT_TRACEWISE_METRICS
        metrics, _ = initialize_metrics(metrics, metric_class=TracewiseMetric)

        overwrite_metric = {metric.name for metric in metrics} & self.qc_metrics.keys()
        if overwrite_metric:
            msg = ', '.join(overwrite_metric)
            if not overwrite:
                metrics = [metric for metric in metrics if metric.name not in self.qc_metrics]
                if not len(metrics):
                    warnings.warn("All metrics already calculated. Use `overwrite=True` to recalculate them.")
                    return self
            else:
                warnings.warn(f"{msg} already calculated and will be rewritten.")

        n_chunks = self.n_traces // chunk_size + (1 if self.n_traces % chunk_size else 0)
        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)

        _, idx_sort, idx_orig = np.unique(self['TRACE_SEQUENCE_FILE'], return_index=True, return_inverse=True)

        def calc_metrics(ixs):
            gather = self.load_gather(self.headers.iloc[ixs])
            results = {}
            for metric in metrics:
                # Save header_cols since the metric may become partial and the attribute will be unreachable
                header_cols = metric.header_cols
                if isinstance(metric, BaseWindowRMSMetric):
                    metric = partial(metric, return_rms=False)
                results.update(zip(to_list(header_cols), np.atleast_2d(metric(gather))))
            return pd.DataFrame(results)

        # Precompile all njit decorated metrics to avoid hanging of the ThreadPoolExecutor upon the first call
        _ = calc_metrics([0])

        futures = []
        with tqdm(total=self.n_traces, desc="Traces processed", disable=not bar) as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    future = pool.submit(calc_metrics, idx_sort[i * chunk_size: (i + 1) * chunk_size])
                    future.add_done_callback(lambda fut: pbar.update(len(fut.result())))
                    futures.append(future)

        results = pd.concat([future.result() for future in futures], ignore_index=True, copy=False).iloc[idx_orig]
        results.index = self.headers.index
        self.headers[results.columns] = results
        self.qc_metrics.update({metric.name: metric for metric in metrics})

        if verbose:
            print(self.qc_summary)
        return self

    #------------------------------------------------------------------------#
    #                         Visualization methods                          #
    #------------------------------------------------------------------------#

    def plot_geometry(self, **kwargs):
        """Plot shot and receiver locations on a field map.

        This plot is interactive and provides 2 views:
        * Shot view: displays shot locations. Highlights all activated receivers on click and displays the
          corresponding common shot gather.
        * Receiver view: displays receiver locations. Highlights all shots that activated the receiver on click and
          displays the corresponding common receiver gather.

        Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        show_contour : bool, optional, defaults to True
            Whether to display a field contour if survey geometry was inferred.
        keep_aspect : bool, optional, defaults to False
            Whether to keep aspect ratio of the map plot.
        source_id_cols : str or list of str, optional
            Trace headers that uniquely identify a seismic source. If not given, `self.source_id_cols` is used.
        source_sort_by : str or list of str, optional
            Header names to sort the displayed common source gathers by. If not given, passed `sort_by` value is used.
        receiver_id_cols : str or list of str, optional
            Trace headers that uniquely identify a receiver. If not given, `self.receiver_id_cols` is used.
        receiver_sort_by : str or list of str, optional
            Header names to sort the displayed common receiver gathers by. If not given, passed `sort_by` value is
            used.
        sort_by : str or list of str, optional
            Default header names to sort the displayed gather by. If not given, no sorting is performed.
        gather_plot_kwargs : dict, optional
            Additional arguments to pass to `Gather.plot`.
        x_ticker : str or dict, optional
            Parameters to control `x` axis tick formatting and layout of the map plot. See `.utils.set_ticks` for more
            details.
        y_ticker : dict, optional
            Parameters to control `y` axis tick formatting and layout of the map plot. See `.utils.set_ticks` for more
            details.
        figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
            Size of created map and gather figures. Measured in inches.
        orientation : {"horizontal", "vertical"}, optional, defaults to "horizontal"
            Defines whether to stack the main and auxiliary plots horizontally or vertically.
        kwargs : misc, optional
            Additional keyword arguments to pass to `matplotlib.axes.Axes.scatter` when plotting the map.
        """
        SurveyGeometryPlot(self, **kwargs).plot()

    def _construct_map(self, values, metric, by, id_cols=None, drop_duplicates=False, agg=None, bin_size=None,
                       **kwargs):
        """Construct a metric map of `values` aggregated by gather, whose type is defined by `by`."""
        by_to_cols = {
            "source": (self.source_id_cols, ["SourceX", "SourceY"]),
            "shot": (self.source_id_cols, ["SourceX", "SourceY"]),
            "receiver": (self.receiver_id_cols, ["GroupX", "GroupY"]),
            "rec": (self.receiver_id_cols, ["GroupX", "GroupY"]),
            "cdp": (None, ["CDP_X", "CDP_Y"]),
            "cmp": (None, ["CDP_X", "CDP_Y"]),
            "midpoint": (None, ["CDP_X", "CDP_Y"]),
            "bin": (None, ["INLINE_3D", "CROSSLINE_3D"]),
            "supergather": (None, ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]),
        }
        index_cols, coords_cols = by_to_cols.get(by.lower())
        if coords_cols is None:
            raise ValueError(f"by must be one of {', '.join(by_to_cols.keys())} but {by} given.")
        index_cols = get_first_defined(id_cols, index_cols)

        metric = SurveyAttribute(name=metric) if isinstance(metric, str) else metric
        metric_data = self.get_headers(coords_cols)
        if index_cols is not None:
            index_cols = to_list(index_cols)
            metric_data[index_cols] = self[index_cols]
        metric_data[metric.header_cols] = values
        if drop_duplicates:
            metric_data.drop_duplicates(inplace=True)
        index = metric_data[index_cols] if index_cols is not None else None
        coords = metric_data[coords_cols]
        values = metric_data[metric.header_cols]

        metric = metric.provide_context(survey=self)
        return metric.construct_map(coords, values, index=index, agg=agg, bin_size=bin_size, **kwargs)

    def construct_header_map(self, col, by, id_cols=None, drop_duplicates=False, agg=None, bin_size=None):
        """Construct a metric map of trace header values aggregated by gather.

        Examples
        --------
        Construct a map of maximum offset by shots:
        >>> max_offset_map = survey.construct_header_map("offset", by="shot", agg="max")
        >>> max_offset_map.plot()

        The map allows for interactive plotting: a gather type defined by `by` will be displayed on click on the map.
        The gather may optionally be sorted if `sort_by` argument is passed to the `plot` method:
        >>> max_offset_map.plot(interactive=True, sort_by="offset")

        Parameters
        ----------
        col : str
            Headers column to extract values from.
        by : {"source", "shot", "receiver", "rec", "cdp", "cmp", "midpoint", "bin", "supergather"}
            Gather type to aggregate header values over.
        id_cols : str or list of str, optional
            Trace headers that uniquely identify a gather of the chosen type. Acts as an index of the resulting map. If
            not given and `by` represents either a common source or common receiver gather, `self.source_id_cols` or
            `self.receiver_id_cols` are used respectively.
        drop_duplicates : bool, optional, defaults to False
            Whether to drop duplicated entries of (index, coordinates, metric value). Useful when dealing with a header
            defined for a shot or receiver, not a trace (e.g. constructing a map of elevations by shots).
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        header_map : BaseMetricMap
            Constructed header map.
        """
        return self._construct_map(self[col], metric=col, by=by, id_cols=id_cols, drop_duplicates=drop_duplicates,
                                   agg=agg, bin_size=bin_size)

    def construct_fold_map(self, by, id_cols=None, agg=None, bin_size=None):
        """Construct a metric map which stores the number of traces for each gather (fold).

        Examples
        --------
        Generate supergathers and calculate their fold:
        >>> supergather_columns = ["SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"]
        >>> supergather_survey = survey.generate_supergathers(size=7, step=7)
        >>> fold_map = supergather_survey.construct_fold_map(by="supergather")
        >>> fold_map.plot()

        Parameters
        ----------
        by : {"source", "shot", "receiver", "rec", "cdp", "cmp", "midpoint", "bin", "supergather"}
            Gather type to aggregate header values over.
        id_cols : str or list of str, optional
            Trace headers that uniquely identify a gather of the chosen type. Acts as an index of the resulting map. If
            not given and `by` represents either a common source or common receiver gather, `self.source_id_cols` or
            `self.receiver_id_cols` are used respectively.
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        fold_map : BaseMetricMap
            Constructed fold map.
        """
        tmp_map = self._construct_map(np.ones(self.n_traces), metric="fold", by=by, id_cols=id_cols, agg="sum")
        index = tmp_map.index_data[tmp_map.index_cols]
        coords = tmp_map.index_data[tmp_map.coords_cols]
        values = tmp_map.index_data[tmp_map.metric_name]
        return tmp_map.metric.construct_map(coords, values, index=index, agg=agg, bin_size=bin_size)

    def construct_qc_maps(self, metric_names=None, by=None, id_cols=None, agg=None, bin_size=None):
        """Construct a metric map of tracewise metric values aggregated by gathers.

        A ratio of any two RMS metrics may be calculated by passing their names separated by `/` in `metric_names`,
        which allows displaying more complex metrics such as signal-to-noise ratio by gather. By default, RMS
        amplitudes are first calculated for each gather defined by `by` and then used to calculate the ratio. If
        `tracewise` flag is set to `True` (see examples below), RMS amplitudes or ratios are first independently
        calculated for each trace and then aggregated by gathers.

        Examples
        --------
        Define metrics to calculate:
            - Maximum absolute amplitude value divided by trace's std:
            >>> max_abs = TraceMaxAbs
            - RMS in a window located in a signal zone:
            >>> signal_rms = WindowRMS(offsets=[650, 2000], times=[1000, 1400], name="SignalWindowRMS")
            - RMS in a window located in a noise zone:
            >>> noise_rms = WindowRMS(offsets=[650, 2000], times=[100, 500], name="NoiseWindowRMS")

        Compute provided metrics:
        >>> survey.qc([max_abs, signal_rms, noise_rms])

        Construct a metric map of:
            - metric with name `TraceMaxAbs` by shots with `max` aggregation:
            >>> qc_map = survey.construct_qc_maps(metric_names="TraceMaxAbs", by="shot", agg="max")

            - tracewise signal RMS aggregated by shots:
            >>> tracewise_qc_map = survey.construct_qc_maps(metric_names={"metric": "SignalWindowRMS",
                                                                          "tracewise":True}, by="shot")
            - signal-to-noise ratio map by shots:
            >>> ratio_qc_map = survey.construct_qc_maps(metric_names="SignalWindowRMS/NoiseWindowRMS", by="shot")

        Plot the map of the first metric:
        >>> qc_map.plot()

        The map allows for interactive plotting: a gather type defined by `by` with a tracewise metric value on top of
        the gather plot will be displayed on click on the map. Depending on the metric, other arguments may be passed
        to the metric plot. See `metric.plot()` for available arguments. In this example, the gather will be sorted by
        `offset` and the default threshold for the metric will be changed to 20:
        >>> qc_map.plot(interactive=True, threshold=20, sort_by="offset")

        The map in interactive mode also allows to pass arguments for a gather plot. In this example, the gather will
        be plotted in `seismogram` mode:
        >>> qc_map.plot(interactive=True, plot_on_click_kwargs={"mode": "seismogram"})

        Parameters
        ----------
        metric_names : str, dict or array-like, optional, defaults to None
            Metrics names to construct metrics maps for.
            If `dict`, allows passing any kwargs to `construct_map` method of the specified metric.
            The following keys are supported:
            - `metric`: metric name,
            - Any additional arguments for `metric.construct_map`.
            If array-like, each element should be `str` or `dict`.
            If None, maps for all metrics that were calculated for this survey are built.
        by : {"source", "shot", "receiver", "rec", "cdp", "cmp", "midpoint", "bin", "supergather"}
            Gather type to aggregate metric values over. Note that this argument cannot be None and should be defined.
        id_cols : str or list of str, optional
            Trace headers that uniquely identify a gather of the chosen type. Acts as an index of the resulting map. If
            not given and `by` represents either a common source or common receiver gather, `self.source_id_cols` or
            `self.receiver_id_cols` are used respectively.
        agg : str or callable, optional, defaults to "mean"
            An aggregation function. Passed directly to `pandas.core.groupby.DataFrameGroupBy.agg`.
        bin_size : int, float or array-like with length 2, optional
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.

        Returns
        -------
        metrics_maps : BaseMetricMap or list of BaseMetricMap
            Calculated metrics maps. Has the same length as `metric_names`.
        """
        is_single_metric = isinstance(metric_names, (str, dict))
        if metric_names is None:
            metric_names = list(self.qc_metrics)
        metrics_list = to_list(metric_names)
        for metric in metrics_list:
            if isinstance(metric, dict) and "metric" not in metric:
                raise ValueError("Missing key `metric` for one of the passed metrics in `metric_names`")
        # Copy dicts to prevent deleting keys from passed objects during metric map constructing
        metrics_list = [{"metric": metric} if isinstance(metric, str) else metric.copy() for metric in metrics_list]

        metric_maps = []
        for metric_dict in metrics_list:
            metric_name = metric_dict.pop("metric")
            if "/" in metric_name:
                operand_metrics = list(map(lambda name: name.strip(), metric_name.split("/")))
                if len(operand_metrics) != 2:
                    raise ValueError(f"Exactly two metrics should be used for division, not {len(operand_metrics)}")
                for metric in operand_metrics:
                    if metric not in self.qc_metrics:
                        raise ValueError(f'Metric with name "{metric}" is not calculated yet')
                metric = MetricsRatio(*[self.qc_metrics[metric] for metric in operand_metrics])
            elif metric_name in self.qc_metrics:
                metric = self.qc_metrics[metric_name]
            else:
                raise ValueError(f'Metric with name "{metric_name}" is not calculated yet')
            metric_map = self._construct_map(self.get_headers(metric.header_cols), metric=metric, by=by,
                                             id_cols=id_cols, agg=agg, bin_size=bin_size, **metric_dict)
            metric_maps.append(metric_map)
        return metric_maps[0] if is_single_metric else metric_maps
