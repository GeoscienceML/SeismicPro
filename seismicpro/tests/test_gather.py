"""Implementation of tests for Gather"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from seismicpro import Survey, Gather, Muter, StackingVelocity
from seismicpro.utils import to_list
from seismicpro.const import HDR_FIRST_BREAK

from .conftest import SAMPLE_INTERVAL, DELAY


@pytest.fixture(scope='module')
def survey(segy_path):
    """Create a survey."""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'],
                    header_cols=['offset', 'FieldRecord'], validate=False)
    survey.remove_dead_traces(bar=False)
    survey.collect_stats(bar=False)
    survey.headers[HDR_FIRST_BREAK] = np.random.randint(0, 1000, len(survey.headers))
    return survey


@pytest.fixture(scope='function')
def gather(survey):
    """Load a gather."""
    return survey.get_gather((0, 0))


def assert_gathers_equal(first, second, drop_cols=None, same_survey=True):
    """Check if two gathers are equal."""
    first_headers = first.headers.reset_index()
    second_headers = second.headers.reset_index()
    if drop_cols is not None:
        first_headers.drop(columns=drop_cols, errors="ignore", inplace=True)
        second_headers.drop(columns=drop_cols, errors="ignore", inplace=True)
    assert len(first_headers) == len(second_headers)
    if len(first_headers) > 0:
        assert first_headers.equals(second_headers)
        assert np.all(first_headers.dtypes == second_headers.dtypes)

    assert np.allclose(first.data, second.data)
    assert first.data.dtype == second.data.dtype
    assert np.allclose(first.samples, second.samples)
    assert first.samples.dtype == second.samples.dtype
    assert first.sample_interval == second.sample_interval
    assert first.delay == second.delay
    assert first.sort_by == second.sort_by
    if same_survey:
        assert id(first.survey) == id(second.survey)


def test_gather_attrs(gather):
    """Check whether the gather stores only known attributes."""
    expected_attrs = {"data", "headers", "samples", "sample_interval", "delay", "sort_by", "survey"}
    unknown_attrs = gather.__dict__.keys() - expected_attrs
    assert len(unknown_attrs) == 0, f"The gather contains unknown attributes {', '.join(unknown_attrs)}"


@pytest.mark.parametrize("key", [
    "UnknownHeader",  # Unknown header
    ["offset", "AnotherUnknownHeader"],  # Selection of known and unknown headers
    (slice(None), "HeaderName"),  # str in key
    None,  # Adds dims
    ([0, 1], None),  # Adds dims
    (np.array([1, 2, 3]), None),  # Adds dims, array case
    (slice(5), slice(100), [1, 2, 3]),  # Too many indexers
    (slice(5), [1, 2, 3]),  # Advanced indexing of samples axis
    (slice(None), slice(200, 0, -2)),  # Negative step for samples axis
    slice(5, 5),  # Empty gather after indexing
    (slice(None), slice(0, 0)),  # Empty gather after indexing
])
@pytest.mark.parametrize("method", ["__getitem__", "get_item"])
def test_gather_getitem_fails(gather, method, key):
    """Check if gather indexing properly fails."""
    get_item_method = getattr(gather, method)
    with pytest.raises((KeyError, ValueError)):
        _ = get_item_method(key)


@pytest.mark.parametrize("key", [
    "offset",
    ["offset", "FieldRecord"]
])
def test_gather_getitem_headers(gather, key):
    """Test selection of headers values."""
    result_getitem = gather[key]
    result_get_item = gather.get_item(key)
    expected = gather.headers.reset_index()[key].to_numpy()

    assert np.allclose(result_getitem, expected)
    assert result_getitem.dtype == expected.dtype
    assert np.allclose(result_get_item, expected)
    assert result_get_item.dtype == expected.dtype


@pytest.mark.parametrize("key, trace_indexer, samples_indexer, sample_interval, delay, preserve_sort_by", [
    # Selecting only traces
    [0, [0], slice(None), SAMPLE_INTERVAL, DELAY, True],
    [-1, [-1], slice(None), SAMPLE_INTERVAL, DELAY, True],
    [[1, 2, 5], [1, 2, 5], slice(None), SAMPLE_INTERVAL, DELAY, True],
    [np.array([1, 2, 3]), [1, 2, 3], slice(None), SAMPLE_INTERVAL, DELAY, True],
    [np.array([1, 1, 1]), [1, 1, 1], slice(None), SAMPLE_INTERVAL, DELAY, True],
    [[5, 3, 1], [5, 3, 1], slice(None), SAMPLE_INTERVAL, DELAY, False],
    [slice(None), slice(None), slice(None), SAMPLE_INTERVAL, DELAY, True],
    [slice(None, None, 3), slice(None, None, 3), slice(None), SAMPLE_INTERVAL, DELAY, True],
    [slice(100, 0, -2), slice(100, 0, -2), slice(None), SAMPLE_INTERVAL, DELAY, False],

    # Selecting both traces and samples
    [(slice(None), slice(None)), slice(None), slice(None), SAMPLE_INTERVAL, DELAY, True],
    [(0, slice(None)), [0], slice(None), SAMPLE_INTERVAL, DELAY, True],
    [(0, 5), [0], slice(5, 6), SAMPLE_INTERVAL, DELAY + SAMPLE_INTERVAL * 5, True],
    [([3, 2, 1], 100), [3, 2, 1], slice(100, 101), SAMPLE_INTERVAL, DELAY + SAMPLE_INTERVAL * 100, False],
    [(np.array([2, 5]), slice(20, 40)), [2, 5], slice(20, 40), SAMPLE_INTERVAL, DELAY + SAMPLE_INTERVAL * 20, True],
    [(slice(3, 8), slice(100)), slice(3, 8), slice(100), SAMPLE_INTERVAL, DELAY, True],
    [(slice(5, 1, -2), slice(100, None)), slice(5, 1, -2), slice(100, None), SAMPLE_INTERVAL,
     DELAY + 100 * SAMPLE_INTERVAL, False],
    [(slice(4, 5), slice(50, 100, 4)), slice(4, 5), slice(50, 100, 4), SAMPLE_INTERVAL * 4,
     DELAY + 50 * SAMPLE_INTERVAL, True],
    [(6, slice(10, None, 2)), [6], slice(10, None, 2), SAMPLE_INTERVAL * 2, DELAY + 10 * SAMPLE_INTERVAL, True],
])
@pytest.mark.parametrize("sort_by", [None, "offset"])
def test_gather_getitem_gather(gather, key, trace_indexer, samples_indexer, sample_interval, delay, preserve_sort_by,
                               sort_by):
    """Test selection of gather traces and samples."""
    if sort_by is not None:
        gather = gather.sort(by=sort_by)
    result_getitem = gather[key]
    result_get_item = gather.get_item(key)
    target = Gather(headers=gather.headers.iloc[trace_indexer], data=gather.data[trace_indexer, samples_indexer],
                    sample_interval=sample_interval, survey=gather.survey, delay=delay)
    if preserve_sort_by:
        target.sort_by = gather.sort_by
    assert_gathers_equal(result_getitem, target)
    assert_gathers_equal(result_get_item, target)


@pytest.mark.parametrize("ignore, ignore_set", [
    [None, {"survey"}],
    ["survey", {"survey"}],
    ["data", {"survey", "data"}],
    [("data", "headers"), {"survey", "data", "headers"}],
    [{"headers", "samples"}, {"survey", "headers", "samples"}],
    [["data", "headers", "samples"], {"survey", "data", "headers", "samples"}],
])
def test_gather_copy(gather, ignore, ignore_set):
    """Test whether gather copy equals to the gather itself and avoids copying its survey and attributes listed in
    `ignore`."""
    gather_copy = gather.copy(ignore=ignore)
    assert_gathers_equal(gather, gather_copy)

    for attr in ["data", "headers", "samples", "survey"]:
        orig_id = id(getattr(gather, attr))
        copy_id = id(getattr(gather_copy, attr))
        if attr in ignore_set:
            assert copy_id == orig_id
        else:
            assert copy_id != orig_id


@pytest.mark.parametrize('columns', ['offset', 'FieldRecord', 'col_1', ['col_1'], ['col_1', 'col_2']])
def test_gather_store_headers_to_survey(segy_path, columns):
    """test_gather_store_headers_to_survey"""
    # Creating survey every time since this method affects survey and we cannot use global gather fixture here
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'],
                    header_cols=['offset', 'FieldRecord'], validate=False)
    copy_survey = survey.copy()

    gather = survey.get_gather((0, 0))
    gather.headers["col_1"] = np.arange(gather.n_traces, dtype=np.int32)
    gather.headers["col_2"] = 100 * np.random.random(gather.n_traces)
    gather.headers["offset"] = gather["offset"] * 0.1

    gather.store_headers_to_survey(columns)
    headers_from_survey = survey.headers.loc[gather.index].sort_values(by="offset")
    headers_from_gather = gather.headers.sort_values(by="offset")
    assert np.allclose(headers_from_survey[columns], headers_from_gather[columns])

    other_indices = list(set(survey.indices) ^ {gather.index})
    expected_survey = copy_survey.headers.loc[other_indices]
    changed_survey = survey.headers.loc[other_indices]
    for column in to_list(columns):
        if column in ["col_1", "col_2"]:
            assert changed_survey[column].isnull().sum() == len(changed_survey)
        else:
            assert np.allclose(expected_survey[column], changed_survey[column])

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
@pytest.mark.parametrize('q', [0.1, [0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2])])
def test_gather_get_quantile(gather, tracewise, use_global, q):
    """Test gather's methods"""
    # # check that quantile has the same type as q
    gather.get_quantile(q=q, tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_standard(gather, tracewise, use_global):
    """test_gather_scale_standard"""
    gather.scale_standard(tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_minmax(gather, tracewise, use_global):
    """test_gather_scale_minmax"""
    gather.scale_minmax(tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_maxabs(gather, tracewise, use_global):
    """test_gather_scale_minmax"""
    gather.scale_maxabs(tracewise=tracewise, use_global=use_global)

def test_gather_mask_to_pick_and_pick_to_mask(gather):
    """test_gather_mask_to_pick"""
    mask = gather.pick_to_mask(first_breaks_header=HDR_FIRST_BREAK)
    mask.mask_to_pick(first_breaks_header=HDR_FIRST_BREAK, save_to=gather)

@pytest.mark.parametrize('by', ('offset', ['FieldRecord', 'offset']))
def test_gather_sort(gather, by):
    """test_gather_sort"""
    gather.sort(by=by)

def test_gather_muting(gather):
    """test_gather_muting"""
    muter = Muter(offsets=[1000, 2000, 3000], times=[100, 300, 600])
    gather.mute(muter)

@pytest.mark.parametrize('mode', ('S', 'NS', 'NE', 'CC', 'ENCC'))
def test_gather_velocity_spectrum(gather, mode):
    """test_gather_velocity_spectrum"""
    gather.calculate_vertical_velocity_spectrum(mode=mode)

def test_gather_res_velocity_spectrum(gather):
    """test_gather_res_velocity_spectrum"""
    stacking_velocity = StackingVelocity(times=[0, 3000], velocities=[1600, 3500])
    gather.calculate_residual_velocity_spectrum(stacking_velocity=stacking_velocity)

def test_gather_stacking_velocity(gather):
    """test_gather_stacking_velocity"""
    gather.sort(by='offset')
    stacking_velocity = StackingVelocity(times=[0, 3000], velocities=[1600, 3500])
    gather.apply_nmo(stacking_velocity=stacking_velocity)

def test_gather_get_central_gather(segy_path):
    """test_gather_get_central_gather"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'], header_cols=['offset', 'FieldRecord'],
                    validate=False)
    survey = survey.generate_supergathers()
    gather = survey.sample_gather()
    gather.get_central_gather()

def test_gather_stack(gather):
    """test_gather_stack"""
    gather.stack()

def test_apply_undo_agc(gather):
    """test_apply_undo_agc"""
    gather_copy = gather.copy()
    gather, coefs = gather.apply_agc(return_coefs=True)
    gather = gather.undo_agc(coefs)
    assert_gathers_equal(gather_copy, gather)
