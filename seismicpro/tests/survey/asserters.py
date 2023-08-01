"""General Survey assertions"""

# pylint: disable=protected-access
import pathlib

import segyio
import segfast
import numpy as np

from seismicpro import Survey
from seismicpro.utils.indexer import create_indexer
from seismicpro.const import HDR_TRACE_POS

from ..utils import assert_indexers_equal


# Define default tolerances to check if two float values are close
RTOL = 1e-5
ATOL = 1e-7


def assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers, rtol=RTOL, atol=ATOL):
    """Check if a SEG-Y file was properly loaded into a `Survey` instance."""
    with segyio.open(segy_path, ignore_geometry=True) as f:
        file_samples = f.samples
        n_traces = f.tracecount
    n_samples = len(file_samples)
    file_sample_interval = np.unique(np.diff(file_samples))[0]

    # Check all path-related attributes
    assert survey.name == expected_name
    assert pathlib.Path(survey.path) == segy_path
    assert pathlib.Path(survey.loader.path) == segy_path
    assert pathlib.Path(survey.loader.file_handler._filename) == segy_path

    # Check whether samples data matches that of the source file
    assert survey.n_file_samples == n_samples
    assert np.allclose(survey.file_samples, file_samples, rtol=rtol, atol=atol)
    assert np.isclose(survey.file_sample_interval, file_sample_interval, rtol=rtol, atol=atol)
    assert np.isclose(survey.file_sample_rate, 1000 / file_sample_interval, rtol=rtol, atol=atol)

    # Check names of loaded trace headers and the resulting headers shape
    assert survey.n_traces == n_traces
    assert len(survey.headers) == n_traces
    assert set(survey.headers.index.names) == expected_index
    assert set(survey.headers.index.names) | set(survey.headers.columns) == expected_headers
    assert survey.headers.index.is_monotonic_increasing
    # Check that HDR_TRACE_POS reflects traces positions in the headers. We are checking `survey.headers` instead of
    # `loaded_headers`, because `loaded_headers` are sorted differently, and HEAT_TRACE_POS has lost all meaning there.
    assert np.array_equal(survey.headers[HDR_TRACE_POS].values, np.arange(n_traces))
    # Restore the order of the traces from the source file
    loaded_headers = survey.headers.reset_index().sort_values(by="TRACE_SEQUENCE_FILE")

    # Check loaded trace headers values
    assert np.array_equal(loaded_headers["TRACE_SEQUENCE_FILE"].values, np.arange(1, n_traces + 1))
    for header in set(loaded_headers.columns) - {"TRACE_SEQUENCE_FILE", HDR_TRACE_POS}:
        assert np.array_equal(loaded_headers[header].values,
                              survey.loader.file_handler.attributes(segyio.tracefield.keys[header])[:])

    # Check whether indexer was constructed correctly
    assert_indexers_equal(survey._indexer, create_indexer(survey.headers.index))


def assert_both_none_or_close(left, right, rtol=RTOL, atol=ATOL):
    """Check whether both `left` and `right` are `None` or they are close."""
    left_none = left is None
    right_none = right is None
    assert not(left_none ^ right_none)  # pylint: disable=superfluous-parens
    assert left_none and right_none or np.allclose(left, right, rtol=rtol, atol=atol)


def assert_loaders_equal(left, right):
    """Check whether both `left` and `right` SEG-Y loaders are equal."""
    assert isinstance(left, (segfast.SegyioLoader, segfast.MemmapLoader))
    assert type(left) is type(right)
    assert pathlib.Path(left.path) == pathlib.Path(right.path)
    assert pathlib.Path(left.file_handler._filename) == pathlib.Path(right.file_handler._filename)
    assert left.endian == right.endian
    assert left.strict == right.strict
    assert left.ignore_geometry == right.ignore_geometry


def assert_surveys_equal(left, right, ignore_column_order=False, ignore_dtypes=False, rtol=RTOL, atol=ATOL):
    """Check if two surveys are equal. Optionally allow for changes in headers order or dtypes."""
    # Check whether all path-related attributes are equal
    assert isinstance(left, Survey)
    assert isinstance(right, Survey)
    assert left.name == right.name
    assert pathlib.Path(left.path) == pathlib.Path(right.path)
    assert_loaders_equal(left.loader, right.loader)

    # Check whether source and receiver id cols are equal
    assert left.source_id_cols == right.source_id_cols
    assert left.receiver_id_cols == right.receiver_id_cols

    # Check whether source file samples are equal
    assert left.n_file_samples == right.n_file_samples
    assert np.allclose(left.file_samples, right.file_samples, rtol=rtol, atol=atol)
    assert np.isclose(left.file_sample_rate, right.file_sample_rate, rtol=rtol, atol=atol)
    assert np.isclose(left.file_sample_interval, right.file_sample_interval, rtol=rtol, atol=atol)

    # Check whether loaded headers are equal
    left_headers = left.headers
    right_headers = right.headers
    assert set(left_headers.columns) == set(right_headers.columns)
    if ignore_column_order:
        right_headers = right_headers[left_headers.columns]
    if ignore_dtypes:
        right_headers = right_headers.astype(left_headers.dtypes)
    assert left_headers.equals(right_headers)
    assert left.n_traces == right.n_traces

    # Check whether survey indexers are equal
    assert_indexers_equal(left._indexer, right._indexer)

    # Check whether same default limits are applied
    assert left.limits == right.limits
    assert left.n_samples == right.n_samples
    assert np.allclose(left.times, right.times, rtol=rtol, atol=atol)
    assert np.allclose(left.samples, right.samples, rtol=rtol, atol=atol)
    assert np.isclose(left.sample_rate, right.sample_rate, rtol=rtol, atol=atol)
    assert np.isclose(left.sample_interval, right.sample_interval, rtol=rtol, atol=atol)

    # Assert that either stats were not calculated for both surveys or they are equal
    assert left.has_stats == right.has_stats
    assert_both_none_or_close(left.min, right.min, rtol=rtol, atol=atol)
    assert_both_none_or_close(left.max, right.max, rtol=rtol, atol=atol)
    assert_both_none_or_close(left.mean, right.mean, rtol=rtol, atol=atol)
    assert_both_none_or_close(left.std, right.std, rtol=rtol, atol=atol)

    # Assert that qc metrics were not calculated for both surveys or metrics are the same
    assert len(left.qc_metrics) == len(right.qc_metrics)
    assert all(name in right.qc_metrics and right.qc_metrics[name] is obj for name, obj in left.qc_metrics.items())

    q = np.linspace(0, 1, 11)
    left_quantiles = left.quantile_interpolator(q) if left.quantile_interpolator is not None else None
    right_quantiles = right.quantile_interpolator(q) if right.quantile_interpolator is not None else None
    assert_both_none_or_close(left_quantiles, right_quantiles, rtol=rtol, atol=atol)


def assert_surveys_not_linked(base, altered):
    """Check whether attributes of both `base` and `altered` surveys have links to the same data by changing `altered`
    data inplace. Each data attribute is copied via its own deepcopy method in order not to use `Survey.copy`, which
    calls this function in its own tests."""
    # Modify headers
    unchanged_headers = base.headers.copy()
    altered.headers.iloc[:, :] = 0
    assert unchanged_headers.equals(base.headers)

    # Modify samples
    unchanged_samples = np.copy(base.samples)
    altered.samples += 1
    assert np.allclose(unchanged_samples, base.samples)


def assert_survey_processed_inplace(before, after, inplace):
    """Assert whether survey processing was performed inplace depending on the `inplace` flag. Changes `after` data
    inplace if `inplace` flag is set to `True`."""
    assert (id(before) == id(after)) is inplace
    if not inplace:
        assert_surveys_not_linked(before, after)


def assert_survey_limits_set(survey, limits, rtol=RTOL, atol=ATOL):
    """Check if `survey` limits were set correctly. `limits` must be a `slice` object with `start`, `stop` and `step`
    arguments set to `int`s."""
    limited_samples = survey.file_samples[limits]
    limited_sample_interval = survey.file_sample_interval * limits.step

    assert survey.limits == limits
    assert survey.n_samples == len(limited_samples)
    assert np.allclose(survey.times, limited_samples, rtol=rtol, atol=atol)
    assert np.allclose(survey.samples, limited_samples, rtol=rtol, atol=atol)
    assert np.isclose(survey.sample_interval, limited_sample_interval, rtol=rtol, atol=atol)
    assert np.isclose(survey.sample_rate, 1000 / limited_sample_interval, rtol=rtol, atol=atol)
