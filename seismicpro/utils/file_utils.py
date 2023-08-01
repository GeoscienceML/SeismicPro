"""Implements functions to load and dump data in various formats"""

import os
import glob

import segyio
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from pandas.api.types import is_float_dtype

from .general_utils import to_list


def aggregate_segys(in_paths, out_path, recursive=False, mmap=False, keep_exts=("sgy", "segy"), delete_in_files=False,
                    bar=True):
    """Merge several SEG-Y files into a single one.

    Parameters
    ----------
    in_paths : str or list of str
        Glob mask or masks to search for source files to merge.
    out_path : str
        A path to the resulting merged file.
    recursive : bool, optional, defaults to False
        Whether to treat '**' pattern in `in_paths` as zero or more directories to perform a recursive file search.
    mmap : bool, optional, defaults to False
        Whether to perform memory mapping of input files. Setting this flag to `True` may result in faster reads.
    keep_exts : None, array-like, optional, defaults to ("sgy", "segy")
        Extensions of files to use for merging. If `None`, no filtering is performed.
    delete_in_files : bool, optional, defaults to False
        Whether to delete source files, defined by `in_paths`.
    bar : bool, optional, defaults to True
        Whether to show the progress bar.

    Raises
    ------
    ValueError
        If no files match the given pattern.
        If source files contain inconsistent samples.
    """
    in_paths = sum([glob.glob(path, recursive=recursive) for path in to_list(in_paths)], [])
    if keep_exts is not None:
        in_paths = [path for path in in_paths if os.path.splitext(path)[1][1:] in keep_exts]
    if not in_paths:
        raise ValueError("No files match the given pattern")

    # Calculate total tracecount and check whether all files have the same trace length, time delay and sample rate
    tracecount = 0
    samples_params = set()
    for path in in_paths:
        with segyio.open(path, ignore_geometry=True) as handler:
            tracecount += handler.tracecount
            bin_sample_interval = handler.bin[segyio.BinField.Interval]
            trace_sample_interval = handler.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
            n_samples = handler.trace.shape
            samples_params.add((bin_sample_interval, trace_sample_interval, n_samples))

    if len(samples_params) != 1:
        raise ValueError("Source files contain inconsistent samples")

    # Create segyio spec for the new file, which inherits most of its attributes from the first input file
    spec = segyio.spec()
    spec.tracecount = tracecount
    with segyio.open(in_paths[0], ignore_geometry=True) as handler:
        spec.samples = handler.samples
        spec.ext_headers = handler.ext_headers
        spec.format = handler.format

    # Write traces and their headers from source files into the new one
    os.makedirs(os.path.abspath(os.path.dirname(out_path)), exist_ok=True)
    with segyio.create(out_path, spec) as out_handler:
        trace_pos = 0
        out_handler.bin[segyio.BinField.Interval] = bin_sample_interval
        out_handler.bin[segyio.BinField.Samples] = n_samples
        out_handler.bin[segyio.BinField.ExtSamples] = n_samples
        for path in tqdm(in_paths, desc="Aggregating files", disable=not bar):
            with segyio.open(path, ignore_geometry=True) as in_handler:
                if mmap:
                    in_handler.mmap()
                in_tracecount = in_handler.tracecount
                out_handler.trace[trace_pos : trace_pos + in_tracecount] = in_handler.trace
                out_handler.header[trace_pos : trace_pos + in_tracecount] = in_handler.header
                for i in range(trace_pos, trace_pos + in_tracecount):
                    out_handler.header[i].update({segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1})
                trace_pos += in_tracecount

    # Delete input files if needed
    if delete_in_files:
        for path in in_paths:
            os.remove(path)


def load_dataframe(path, has_header=False, columns=None, usecols=None, skiprows=0, format="fwf", sep=',', decimal=None,
                   encoding="UTF-8", **kwargs):
    """Read a file into a `pd.DataFrame`. See :func:`TraceContainer.load_headers` for arguments description."""
    if usecols is not None or decimal is None:
        with open(path, 'r', encoding=encoding) as f:
            n_skip = 1 + has_header + skiprows
            row = [next(f) for _ in range(n_skip)][-1]
        # If decimal is not provided, try inferring it from the file
        if decimal is None:
            decimal = '.' if '.' in row or format == 'csv' else ','
        if usecols is not None:
            usecols_sep = sep if format == "csv" else None
            # Avoid negative values in `usecols`
            usecols = np.arange(len(row.split(usecols_sep)))[usecols]
            if np.any(usecols[:-1] > usecols[1:]):
                raise ValueError("`usecols` should be sorted in ascending order.")
            usecols = usecols.tolist()

    if format == "fwf":
        header = 0 if has_header else None
        names, usecols = (None, columns) if has_header and usecols is None else (columns, usecols)
        return pd.read_csv(path, sep=r'\s+', header=header, names=names, usecols=usecols, decimal=decimal,
                           skiprows=skiprows, encoding=encoding, **kwargs)
    if format == "csv":
        if decimal != ".":
            # FIXME: Add decimal support when the issue (https://github.com/pola-rs/polars/issues/6698) is solved
            raise ValueError("`decimal` different from '.' is not supported for 'csv' format")
        new_columns = None if has_header else columns
        columns = usecols if not has_header or columns is None else columns
        return pl.read_csv(path, has_header=has_header, columns=columns, new_columns=new_columns, separator=sep,
                           skip_rows=skiprows, encoding=encoding, **kwargs).to_pandas()
    raise ValueError(f"Unknown format `{format}`, available formats are ('fwf', 'csv')")


def dump_dataframe(df, path, has_header=False, format="fwf", sep=",", decimal=".", float_precision=2, min_width=None,
                   **kwargs):
    """Save a provided `pd.DataFrame` to a file. See :func:`TraceContainer.dump_headers` for arguments description."""
    if format == "fwf":
        _dump_to_fwf(df=df, path=path, has_header=has_header, decimal=decimal, float_precision=float_precision,
                     min_width=min_width)
    elif format == "csv":
        if decimal != ".":
            raise ValueError("`decimal` different from '.' is not supported for 'csv' format")
        df = pl.from_pandas(df)
        df.write_csv(path, has_header=has_header, float_precision=float_precision, separator=sep, **kwargs)
    else:
        raise ValueError(f"Unknown format `{format}`, available formats are ('fwf', 'csv')")


def _dump_to_fwf(df, path, has_header, decimal, float_precision, min_width=None):
    def format_float(col, n):
        """Clip all floats to the same amount of fractional numbers and pad with zeros where needed."""
        round_col = pl.col(col).round(n).abs()
        int_part = round_col.floor().cast(int)
        # Converting fractional numbers to int by raising it to a power of `n` and padding with zeros if needed
        frac_part = ((round_col - int_part) * pl.lit(10)**n).round(0).cast(int).cast(str).str.rjust(n, "0")
        str_num = pl.concat_str([int_part.cast(str), frac_part], separator=decimal)
        # Restoring the sign of numbers
        return pl.when(pl.col(col) < 0).then("-" + str_num).otherwise(str_num).alias(col)

    columns = df.columns
    is_float_cols = [is_float_dtype(t) for t in df.dtypes]
    df = pl.from_pandas(df)
    str_df = df.select([format_float(col, float_precision) if is_float else pl.col(col).cast(str)
                        for col, is_float in zip(columns, is_float_cols)])
    col_lens = str_df.select(pl.all().str.lengths().max()).row(0)
    if has_header:
        col_lens = np.maximum(col_lens, [len(column) for column in columns])
    if min_width is not None:
        col_lens = np.maximum(col_lens, min_width)
    col_names = " ".join(col.rjust(n, " ") for col, n in zip(str_df.columns, col_lens))
    str_col = str_df.select(pl.concat_str([pl.col(col).str.rjust(n, " ") for col, n in zip(columns, col_lens)],
                                          separator=" ").alias(col_names))
    # Avoid matching decimal and separator to achieve an unambiguous reading
    separator = "," if decimal == "." else "."
    str_col.write_csv(path, has_header=has_header, separator=separator)


# pylint: disable=too-many-arguments, invalid-name
def make_prestack_segy(path, fmt=5, survey_size=(1000, 1000), origin=(0, 0), sources_step=(50, 300),
                       receivers_step=(100, 25), bin_size=(50, 50), activation_dist=(500, 500), n_samples=1500,
                       sample_interval=2000, delay=0, bar=True, trace_gen=None, **kwargs):
    """Make a prestack SEG-Y file with rectangular geometry. Its headers are filled with values inferred from survey
    geometry parameters, traces are filled with data generated by `trace_gen`.

    All tuples in method arguments indicate either coordinate in (`x`, `y`) or distance in (`x_dist`, `y_dist`) format.

    Parameters
    ----------
    path : str
        Path to store the generated SEG-Y file.
    fmt : 1 or 5, optional, defaults to 5
        Data type used to store trace amplitudes. 1 stands for IBM 4-byte float, 5 - for IEEE float32. Saved in bytes
        3225–3226 of the binary header.
    survey_size : tuple of ints, defaults to (1000, 1000)
        Survey dimensions measured in meters.
    origin : tuple of ints, defaults to (0, 0)
        Coordinates of bottom left corner of the survey.
    sources_step : tuple of ints, defaults to (50, 300)
        Distances between sources. (50, 300) are standard values indicating that source lines are positioned along `y`
        axis with 300 meters step, while sources in each line are located every 50 meters along `x` axis.
    receivers_step : tuple of ints, defaults to (100, 25)
        Distances between receivers. It is supposed that receiver lines span along `x` axis. By default distance
        between receiver lines is 100 meters along `x` axis, and distance between receivers in lines is 25 meters
        along `y` axis.
    bin_size : tuple of ints, defaults to (50, 50)
        Size of a CDP bin in meters.
    activation_dist : tuple of ints, defaults to (500, 500)
        Maximum distance from source to active receiver along each axis. Each source activates a rectangular field of
        receivers with source at its center and shape (2 * activation_dist[0], 2 * activation_dist[1])
    n_samples : int, defaults to 1500
        Number of samples in traces.
    sample_interval : int, defaults to 2000
        Sampling interval in microseconds.
    delay : int, defaults to 0
        Delay time of the seismic trace in milliseconds.
    bar : bool, optional, defaults to True
        Whether to show a progress bar.
    trace_gen : callable, defaults to None.
        Callable to generate trace data. It receives a dict of trace headers along with everything passed in `kwargs`.
        If `None`, traces are filled with gaussian noise.
        Passed headers: TRACE_SEQUENCE_FILE, FieldRecord, TraceNumber, SourceX, SourceY, Group_X, Group_Y, offset, CDP,
                        CDP_X, CDP_Y, INLINE_3D, CROSSLINE_3D, TRACE_SAMPLE_COUNT, TRACE_SAMPLE_INTERVAL,
                        DelayRecordingTime
    kwargs : misc, optional
        Additional keyword arguments to `trace_gen`.
    """
    # By default traces are filled with random noise
    if trace_gen is None:
        def trace_gen(TRACE_SAMPLE_COUNT, **kwargs):
            _ = kwargs
            return np.random.normal(size=TRACE_SAMPLE_COUNT).astype(np.float32)

    def generate_coordinates(origin, sur_size, coords_step):
        """ Support function to create coordinates of sources / receivers """
        x, y = np.mgrid[[slice(start, start+size, step) for start, size, step in zip(origin, sur_size, coords_step)]]
        return np.vstack([x.ravel(), y.ravel()]).T

    # Create coordinate points for sources and receivers
    source_coords = generate_coordinates(origin, survey_size, sources_step)
    receiver_coords = generate_coordinates(origin, survey_size, receivers_step)

    # Create and fill up a SEG-Y spec
    spec = segyio.spec()
    spec.format = fmt
    spec.samples = np.arange(n_samples) * sample_interval / 1000

    # Calculate matrix of active receivers for each source and get overall number of traces
    activation_dist = np.array(activation_dist)
    active_receivers_mask = np.all(np.abs(source_coords[:, None, :] - receiver_coords) <= activation_dist, axis=-1)
    spec.tracecount = np.sum(active_receivers_mask)

    with segyio.create(path, spec) as dst_file:
        # Loop over the survey and put all the data into the new SEG-Y file
        TRACE_SEQUENCE_FILE = 0

        for FieldRecord, source_location in enumerate(tqdm(source_coords, disable=not bar,
                                                           desc='Common shot gathers generated')):
            active_receivers_coords = receiver_coords[active_receivers_mask[FieldRecord]]

            # TODO: maybe add trace with zero offset
            for TraceNumber, receiver_location in enumerate(active_receivers_coords):
                TRACE_SEQUENCE_FILE += 1
                # Create header
                header = dst_file.header[TRACE_SEQUENCE_FILE-1]
                # Fill headers dict
                trace_header_dict = {}
                trace_header_dict['FieldRecord'] = FieldRecord
                trace_header_dict['TraceNumber'] = TraceNumber
                trace_header_dict['SourceX'], trace_header_dict['SourceY'] = source_location
                trace_header_dict['GroupX'], trace_header_dict['GroupY'] = receiver_location
                trace_header_dict['offset'] = int(np.sum((source_location - receiver_location)**2)**0.5)

                # Fill bin-related headers
                midpoint = (source_location + receiver_location) / 2
                trace_header_dict['INLINE_3D'], trace_header_dict['CROSSLINE_3D'] = (midpoint // bin_size).astype(int)
                trace_header_dict['CDP_X'] = trace_header_dict['INLINE_3D'] * bin_size[0] + bin_size[0] // 2
                trace_header_dict['CDP_Y'] = trace_header_dict['CROSSLINE_3D'] * bin_size[1] + bin_size[1] // 2
                trace_header_dict['CDP'] = (trace_header_dict['INLINE_3D'] * survey_size[1] +
                                            trace_header_dict['CROSSLINE_3D'])

                # Fill depth-related fields in header
                trace_header_dict['TRACE_SAMPLE_COUNT'] = n_samples
                trace_header_dict['TRACE_SAMPLE_INTERVAL'] = sample_interval
                trace_header_dict['DelayRecordingTime'] = delay

                # Generate trace and write it to file
                trace = trace_gen(TRACE_SEQUENCE_FILE=TRACE_SEQUENCE_FILE, **trace_header_dict, **kwargs)
                dst_file.trace[TRACE_SEQUENCE_FILE-1] = trace

                # Rename keys in trace_header_dict and update SEG-Y files' header
                trace_header_dict = {segyio.tracefield.keys[k]: v for k, v in trace_header_dict.items()}
                header.update(trace_header_dict)

        dst_file.bin = {segyio.BinField.Traces: TRACE_SEQUENCE_FILE,
                        segyio.BinField.Samples: n_samples,
                        segyio.BinField.Interval: sample_interval}
