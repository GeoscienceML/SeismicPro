"""Defines base containers - mixin classes that implement properties and basic processing logic for objects that store
particular types of data:
* `SamplesContainer` - implements extra properties and methods for concrete subclasses that store information about
  recording times of some data attribute,
* `TraceContainer` - implements properties and processing methods for subclasses with defined `headers` attribute,
* `GatherContainer` - a subclass of `TraceContainer` that also implements fast selection of gather headers by index.
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl

from .decorators import batch_method
from .utils import to_list, get_cols, create_indexer, maybe_copy, load_dataframe, dump_dataframe
from .const import HDR_FIRST_BREAK


class SamplesContainer:
    """A mixin class that implements extra properties and methods for concrete subclasses that store information about
    recording times of some data attribute (usually seismic traces).

    The following attributes are assumed to be defined:
    - `sample_interval` - sample interval of seismic traces,
    - `delay` - global delay recording time of the container,
    - `samples` - recording times for each trace value as a 1d `np.ndarray`. Should be generally created using
      `create_samples` method.
    """

    @property
    def times(self):
        """1d np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.samples

    @property
    def n_samples(self):
        """int: Trace length in samples."""
        return len(self.samples)

    @property
    def n_times(self):
        """int: Trace length in samples."""
        return len(self.times)

    @property
    def sample_rate(self):
        """float: Sample rate of seismic traces. Measured in Hz."""
        return 1000 / self.sample_interval

    @staticmethod
    def create_samples(n_samples, sample_interval, delay=0):
        """Create an array of samples by their number, sampling interval and delay recording time."""
        return (delay + sample_interval * np.arange(n_samples)).astype(np.float32)

    def times_to_indices(self, times, round=False):
        """Convert `times` to their indices in the `samples` array of the container. If some value of `times` is not
        present in `samples`, its index is linearly interpolated or extrapolated. The resulting indices are returned as
        a floating-valued array if `round` is `False` or rounded to the nearest integer otherwise."""
        indices = (times - self.delay) / self.sample_interval
        if round:
            indices = np.rint(indices).astype(np.int32)
        return indices


class TraceContainer:
    """A mixin class that implements extra properties and processing methods for concrete subclasses with defined
    `headers` attribute that stores loaded trace headers as a `pd.DataFrame`."""

    @property
    def indexed_by(self):
        """str or list of str: Names of header indices."""
        index_names = list(self.headers.index.names)
        if len(index_names) == 1:
            return index_names[0]
        return index_names

    @property
    def available_headers(self):
        """set of str: Names of available trace headers: both loaded and created manually."""
        return set(self.headers.columns) | set(self.headers.index.names)

    @property
    def n_traces(self):
        """int: The number of traces."""
        return len(self.headers)

    @property
    def is_empty(self):
        """bool: Whether no traces are stored in the container."""
        return self.n_traces == 0

    def __getitem__(self, key):
        """Select values of trace headers by their names and return them as a `np.ndarray`. Unlike `pandas` indexing
        allows for selection of headers the container is indexed by. The returned array will be 1d if a single header
        is selected and 2d otherwise.

        Parameters
        ----------
        key : str or list of str
            Names of headers to get values for.

        Returns
        -------
        result : np.ndarray
            Headers values.
        """
        return get_cols(self.headers, key).to_numpy()

    def __setitem__(self, key, value):
        """Set given values to selected headers.

        Parameters
        ----------
        key : str or list of str
            Headers to set values for.
        value : np.ndarray
            Headers values to set.
        """
        self.headers[key] = value

    def get_headers(self, cols):
        """Select values of trace headers by their names and return them as a `pandas.DataFrame`. Unlike `pandas`
        indexing allows for selection of headers the container is indexed by.

        Parameters
        ----------
        cols : str or list of str
            Names of headers to get values for.

        Returns
        -------
        headers : pandas.Series or pandas.DataFrame
            Headers values.
        """
        return get_cols(self.headers, cols)

    def copy(self, ignore=None):
        """Perform a deepcopy of all attributes of `self` except for those specified in `ignore`, which are kept
        unchanged.

        Parameters
        ----------
        ignore : str or array-like of str, defaults to None
            Attributes that won't be copied.

        Returns
        -------
        copy : same type as self
            Copy of `self`.
        """
        ignore = set() if ignore is None else set(to_list(ignore))
        ignore_attrs = [getattr(self, attr) for attr in ignore]

        # Construct a memo dict with attributes, that should not be copied
        memo = {id(attr): attr for attr in ignore_attrs}
        return deepcopy(self, memo)

    @staticmethod
    def _apply(func, df, axis, unpack_args, **kwargs):
        """Apply a function to a `pd.DataFrame` along the specified axis.

        Parameters
        ----------
        func : callable
            A function to be applied to `df`.
        df : pd.DataFrame
            A `DataFrame` to which the function will be applied.
        axis : {0 or "index", 1 or "columns", None}
            An axis along which the function is applied:
            - 0 or "index": apply a function to each column,
            - 1 or "columns": apply a function to each row,
            - `None`: apply a function to the `DataFrame` as a whole.
        unpack_args : bool
            If `True`, row or column values are passed to `func` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, columns of the `df` are passed to
            the `func` as individual arguments.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        result : np.ndarray
            The result of applying `func` to `df`.
        """
        if axis is None:
            args = (col_val for _, col_val in df.items()) if unpack_args else (df,)
            res = func(*args, **kwargs)
        else:
            # FIXME: Workaround for a pandas bug https://github.com/pandas-dev/pandas/issues/34822
            # raw=True causes incorrect apply behavior when axis=1 and multiple values are returned from `func`
            raw = axis != 1

            apply_func = (lambda args, **kwargs: func(*args, **kwargs)) if unpack_args else func
            res = df.apply(apply_func, axis=axis, raw=raw, result_type="expand", **kwargs)

        # Convert np.ndarray/pd.Series/pd.DataFrame outputs from `func` to a 2d array
        return pd.DataFrame(res).to_numpy()

    def _post_index(self, key):
        """Implement extra indexing logic of concrete subclass attributes if some of them should also be indexed
        besides `headers`."""
        _ = key
        return

    @batch_method(target="for")
    def filter(self, cond, cols, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Keep only those rows of `headers` where `cond` is `True`.

        Examples
        --------
        Keep only traces whose offset is less than 1500 meters:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> survey.filter(lambda offset: offset < 1500, cols="offset", inplace=True)

        Parameters
        ----------
        cond : callable
            A function to be applied to `self.headers` to get a filtering mask. Must return a boolean array whose
            length equals to the length of `headers` and `True` values correspond to traces to keep.
        cols : str or list of str
            `self.headers` columns for which condition is checked.
        axis : {0 or "index", 1 or "columns", None}, optional, defaults to None
            An axis along which `cond` is applied:
            - 0 or "index": apply `cond` to each column,
            - 1 or "columns": apply `cond` to each row,
            - `None`: apply `cond` to the `DataFrame` as a whole.
        unpack_args : bool, optional, defaults to False
            If `True`, row or column values are passed to `cond` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, each column from `cols` is passed
            to the `cond` as an individual argument.
        inplace : bool, optional, defaults to False
            Whether to perform filtering inplace or process a copy.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `cond` or `pd.DataFrame.apply`.

        Returns
        -------
        result : same type as self
            Filtered `self`.

        Raises
        ------
        ValueError
            If `cond` returns more than one bool value for each row of `headers`.
        """
        self = maybe_copy(self, inplace, ignore="headers")  # pylint: disable=self-cls-assignment
        cols = to_list(cols)
        headers = self.get_headers(cols)
        mask = self._apply(cond, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        if (mask.ndim != 2) or (mask.shape[1] != 1):
            raise ValueError("cond must return a single value for each header row")
        if mask.dtype != np.bool_:
            raise ValueError("cond must return a bool value for each header row")
        mask = mask[:, 0]
        # Guarantee that a copy is set
        self.headers = self.headers.loc[mask].copy()  # pylint: disable=attribute-defined-outside-init
        if self.is_empty:
            warnings.warn("Empty headers after filtering", RuntimeWarning)
        self._post_index(mask)
        return self

    @batch_method(target="for")
    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Apply a function to `self.headers` along the specified axis.

        Examples
        --------
        Convert signed offsets to their absolute values:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> survey.apply(lambda offset: np.abs(offset), cols="offset", inplace=True)

        Parameters
        ----------
        func : callable
            A function to be applied to `self.headers`. Must return a 2d object with shape (`len(self.headers)`,
            `len(res_cols)`).
        cols : str or list of str
            `self.headers` columns for which the function is applied.
        res_cols : str or list of str, optional, defaults to None
            `self.headers` columns in which the result is saved. If not given, equals to `cols`.
        axis : {0 or "index", 1 or "columns", None}, optional, defaults to None
            An axis along which the function is applied:
            - 0 or "index": apply a function to each column,
            - 1 or "columns": apply a function to each row,
            - `None`: apply a function to the `DataFrame` as a whole.
        unpack_args : bool, optional, defaults to False
            If `True`, row or column values are passed to `func` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, each column from `cols` is passed
            to the `func` as an individual argument.
        inplace : bool, optional, defaults to False
            Whether to apply the function inplace or to a copy.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        result : same type as self
            `self` with the function applied.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        cols = to_list(cols)
        headers = self.get_headers(cols)
        res_cols = cols if res_cols is None else to_list(res_cols)
        res = self._apply(func, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        self.headers[res_cols] = res
        return self

    @batch_method(target="for")
    # pylint: disable-next=too-many-arguments
    def load_headers(self, path, has_header=False, headers_names=None, usecols=None, join_on=None, how="inner",
                     skiprows=0, format="fwf", sep=',', decimal=None, encoding="UTF-8", inplace=False, **kwargs):
        """Load headers from a file and join them to `self.headers`.

        Parameters:
        -----------
        path : str
            A path to the file with headers.
        has_header : bool, optional, defaults to False
            Indicate if the first row of the file contains header names or not.
        headers_names : array-like of str, optional, defaults to None
            An array with column names to use as trace header names. If `has_header` is `True`, then `headers_names`
            specifies which columns will be loaded from the file.
        usecols : array-like of int or None, optional, defaults to None
            Columns indices to be selected from the file. Unlike `pandas` loaders, it is allowed to use negative
            indices. Should be always passed in ascending order and have the same length as `headers_names` if both
            passed.
        join_on : str, array-like of str or None, optional, defaults to None
            Column(s) based on which loaded headers will be joined to `self.headers`. If `None`, intersection of
            headers from `headers_names` and `self.headers.columns` will be used.
        how : "inner" or "left", optional, defaults to "inner"
            If "inner", intersection of traces from `self.headers` and trace headers from the loaded file will be used
            as new `self.headers`. If "left", all traces will be kept in `self.headers`. For traces that were missing
            in the loaded file, headers will be filled with `np.nan`.
        skiprows : int, optional, defaults to 0
            Number of rows to skip from the beginning of the file.
        format : "fwf" or "csv", optional, defaults to "fwf"
            Format of the file with headers. Currently, the following options are supported:
            * "fwf" - fixed-width format,
            * "csv" - comma-separated values format.
        sep : str, defaults to ','
            Separator used in the file. Used only for "csv" `format`.
        decimal : str, optional, defaults to None
            Decimal point character. If not provided, it will be inferred from the file. Used only for "fwf" `format`.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        inplace : bool, optional, defaults to False
            Whether to load headers inplace or to a copy.
        **kwargs : misc, optional
            Additional arguments for loading function. If `format="fwf"`, passed to `pandas.read_csv`.
            If `format="csv"`, passed to `polars.read_csv`.

        Returns
        -------
        result : same type as self
            `self` with the loaded headers.

        Raises:
        -------
        ValueError
            If the `format` argument is not one of the supported formats ('fwf', 'csv').
            If the `how` argument is not one of the supported formats ('inner', 'left').
        """
        self = maybe_copy(self, inplace, ignore="headers") # pylint: disable=self-cls-assignment

        loaded_headers = load_dataframe(path=path, has_header=has_header, columns=headers_names, usecols=usecols,
                                        skiprows=skiprows, format=format, sep=sep, decimal=decimal,
                                        encoding=encoding, **kwargs)
        loaded_headers = pl.from_pandas(loaded_headers, nan_to_null=False)

        index_cols = self.headers.index.names  # pylint: disable=access-member-before-definition
        headers = self.headers.copy(deep=False)  # pylint: disable=access-member-before-definition
        headers.reset_index(inplace=True)
        # Save traces positions to remove traces that were deleted after join from other trace related attributes like
        # `data` in Gather.
        headers = pl.from_pandas(headers).with_row_count("row_index")
        # Use intersection of columns from file and self.headers as join columns by default
        if join_on is None:
            join_on = set(headers.columns) & set(loaded_headers.columns)
        if how not in ["inner", "left"]:
            raise ValueError(f"Argument `how` supports only 'inner' and 'left', but given `{how}`.")
        casts = [pl.col(column).cast(headers[column].dtype) for column in to_list(join_on)]
        joined_headers = headers.join(loaded_headers.with_columns(*casts), on=join_on, how=how, suffix="_loaded")
        self.headers = joined_headers.drop(columns="row_index").to_pandas().set_index(index_cols)  # pylint: disable=attribute-defined-outside-init

        if self.is_empty:
            warnings.warn("Empty headers after headers loading", RuntimeWarning)
        # Perform additional filter for traces that were deleted after file loading.
        self._post_index(joined_headers["row_index"].to_numpy())
        return self

    @batch_method(target="for", use_lock=True)
    def dump_headers(self, path, headers_names, dump_headers_names=False, format="fwf", append=False, sep=",",
                     decimal='.', float_precision=2, min_width=None, **kwargs):
        """Save the selected headers to a file.

        Parameters
        ----------
        path : str
            A path to the output file.
        headers_names : str or array-like of str
            `self.headers` columns to be included in the output file.
        dump_headers_names : bool, optional, defaults to False
            Whether to include the headers names in the output file.
        format : "fwf" or "csv", optional, defaults to "fwf"
            Output file format. If "fwf", use fixed-width format. If "csv", use comma-separated format.
        append : bool, optional, defaults to False
            Whether to append dumped headers to the file or write them to an empty file.
        sep : str, defaults to ','
            Separator used in the file. Used only for "csv" `format`.
        decimal : str, optional, defaults to '.'
            Decimal point character. Used only for "fwf" `format`.
        float_precision : int, optional, defaults to 2
            Number of decimal places to write.
        min_width : int or None, optional, defaults to None
            Minimal column width in the output file. Used only for "fwf" `format`.
        kwargs : misc, optional
            Additional arguments for dumping function `polars.write_csv`. Used only for "csv" `format`.

        Returns
        -------
        result : same type as self
            `self` unchanged.

        Raises
        ------
        ValueError
            If the `format` argument is not one of the supported formats ('fwf', 'csv').
        """
        df = self.get_headers(headers_names)
        mode = 'ab' if append else 'wb'
        with open(path, mode) as f:
            dump_dataframe(df=df, path=f, has_header=dump_headers_names, format=format, sep=sep, decimal=decimal,
                           float_precision=float_precision, min_width=min_width, **kwargs)
        return self

    #------------------------------------------------------------------------#
    #                         Task specific methods                          #
    #------------------------------------------------------------------------#

    @batch_method(target="for")
    def load_first_breaks(self, path, trace_id_headers=('FieldRecord', 'TraceNumber'),
                          first_breaks_header=HDR_FIRST_BREAK, inplace=False, **kwargs):
        """Load times of first breaks from a file and save them to a new column in headers.

        Each line of the file stores the first break time for a trace in the last column. The combination of all but
        the last columns should act as a unique trace identifier and is used to match the trace from the file with the
        corresponding trace in `self.headers`.
        The file can have any format that can be read by :func:`TraceContainer.load_headers`.

        Parameters
        ----------
        path : str
            A path to the file with first break times in milliseconds.
        trace_id_headers : str or tuple of str, defaults to ('FieldRecord', 'TraceNumber')
            Columns names from `self.headers`, whose values are stored in all but the last columns of the file.
        first_breaks_header : str, optional, defaults to 'FirstBreak'
            Column name in `self.headers` where loaded first break times will be stored.
        inplace : bool, optional, defaults to False
            Whether to load first break times inplace or to a survey copy.
        kwargs : misc, optional
            Additional keyword arguments to pass to :func:`TraceContainer.load_headers`.

        Returns
        -------
        self : Survey
            A survey with loaded times of first breaks.
        """
        headers_names = to_list(trace_id_headers) + [first_breaks_header]
        return self.load_headers(path=path, headers_names=headers_names, join_on=trace_id_headers, inplace=inplace,
                                 **kwargs)

    @batch_method(target="for", use_lock=True)
    def dump_first_breaks(self, path, trace_id_headers=('FieldRecord', 'TraceNumber'),
                          first_breaks_header=HDR_FIRST_BREAK, **kwargs):
        """Save first break picking times to a file.

        Each line in the resulting file corresponds to one trace, where all columns but the last one store values from
        `trace_id_headers` headers and identify the trace while the last column stores first break time from
        `first_breaks_header` header.

        Parameters
        ----------
        path : str
            A path to the output file.
        trace_id_headers : tuple of str, defaults to ('FieldRecord', 'TraceNumber')
            Columns names from `self.headers` that act as trace id. These would be present in the file.
        first_breaks_header : str, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `self.headers` where first break times are stored.
        kwargs : misc, optional
            Additional keyword arguments to pass to :func:`TraceContainer.dump_headers`.

        Returns
        -------
        self : Survey
            A Survey unchanged
        """
        headers_names = to_list(trace_id_headers) + to_list(first_breaks_header)
        return self.dump_headers(path=path, headers_names=headers_names, **kwargs)


class GatherContainer(TraceContainer):
    """A mixin class that implements extra properties and processing methods for concrete subclasses with defined
    `headers` attribute that stores loaded trace headers for several gathers as a `pd.DataFrame` and means for fast
    selection of gather headers by index."""

    def __len__(self):
        """The number of gathers."""
        return self.n_gathers

    def __contains__(self, index):
        """Returns whether a gather with given `index` is presented in `headers`."""
        return index in self.indices

    @property
    def headers(self):
        """pd.DataFrame: loaded trace headers."""
        return self._headers

    @headers.setter
    def headers(self, headers):
        """Reconstruct an indexer on each headers assignment."""
        if not (headers.index.is_monotonic_increasing or headers.index.is_monotonic_decreasing):
            headers = headers.sort_index(kind="stable")
        self._indexer = create_indexer(headers.index)
        self._headers = headers

    @property
    def indices(self):
        """pd.Index: indices of gathers."""
        return self._indexer.unique_indices

    @property
    def n_gathers(self):
        """int: The number of gathers."""
        return len(self.indices)

    def get_traces_locs(self, indices):
        """Get positions of traces in `headers` by `indices` of their gathers.

        Parameters
        ----------
        indices : array-like
            Indices of gathers to get trace locations for.

        Returns
        -------
        locations : array-like
            Locations of traces of the requested gathers.
        """
        return self._indexer.get_locs_in_indices(indices)

    def get_gathers_locs(self, indices):
        """Get ordinal positions of gathers in the container by their `indices`.

        Parameters
        ----------
        indices : array-like
            Indices of gathers to get ordinal positions for.

        Returns
        -------
        locations : np.ndarray
            Locations of the requested gathers.
        """
        return self._indexer.get_locs_in_unique_indices(indices)

    def get_headers_by_indices(self, indices):
        """Return headers for gathers with given `indices`.

        Parameters
        ----------
        indices : array-like
            Indices of gathers to get headers for.

        Returns
        -------
        headers : pd.DataFrame
            Selected headers values.
        """
        return self.headers.iloc[self.get_traces_locs(indices)]

    def copy(self, ignore=None):
        """Perform a deepcopy of all attributes of `self` except for indexer and those specified in `ignore`, which are
        kept unchanged.

        Parameters
        ----------
        ignore : str or array-like of str, defaults to None
            Attributes that won't be copied.

        Returns
        -------
        copy : same type as self
            Copy of `self`.
        """
        ignore = set() if ignore is None else set(to_list(ignore))
        return super().copy(ignore | {"_indexer"})

    def reindex(self, new_index, inplace=False):
        """Change the index of `self.headers` to `new_index`.

        Parameters
        ----------
        new_index : str or list of str
            Headers columns to become a new index.
        inplace : bool, optional, defaults to False
            Whether to perform reindexation inplace or return a new instance.

        Returns
        -------
        self : same type as self
            Reindexed self.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        headers = self.headers
        headers.reset_index(inplace=True)
        headers.set_index(new_index, inplace=True)
        headers.sort_index(kind="stable", inplace=True)
        self.headers = headers
        return self
