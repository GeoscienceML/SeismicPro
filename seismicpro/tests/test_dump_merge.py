"""Implementation of tests functions for dump and aggregate segy files."""

# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
import os

import glob
import pytest
import numpy as np

from seismicpro import Survey, aggregate_segys
from seismicpro.const import HDR_TRACE_POS
from .test_gather import assert_gathers_equal


@pytest.mark.parametrize('name', ['some_name', None])
@pytest.mark.parametrize('retain_parent_segy_headers', [False, True])
@pytest.mark.parametrize('header_index', ['FieldRecord', 'TRACE_SEQUENCE_FILE'])
@pytest.mark.parametrize('header_cols', [None, 'TraceNumber', 'all'])
@pytest.mark.parametrize('dump_index', [1, 3])
def test_dump_single_gather(segy_path, tmp_path, name, retain_parent_segy_headers, header_index, header_cols,
                            dump_index):
    survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, validate=False)
    expected_gather = survey.get_gather(dump_index)
    expected_gather.dump(path=tmp_path, name=name, retain_parent_segy_headers=retain_parent_segy_headers)

    files = glob.glob(os.path.join(tmp_path, '*'))
    assert len(files) == 1, "Dump creates more than one file"

    dumped_survey = Survey(files[0], header_index=header_index, header_cols=header_cols, validate=False)
    ix = 1 if header_index == 'TRACE_SEQUENCE_FILE' else dump_index
    dumped_gather = dumped_survey.get_gather(index=ix)
    drop_columns = ["TRACE_SEQUENCE_FILE", HDR_TRACE_POS]
    drop_columns += ["TRACE_SAMPLE_INTERVAL"] if "TRACE_SAMPLE_INTERVAL" in expected_gather.headers.columns else []

    assert_gathers_equal(expected_gather, dumped_gather, drop_cols=drop_columns, same_survey=False)

    if retain_parent_segy_headers:
        expected_survey = Survey(segy_path, header_index=header_index, header_cols='all', validate=False)
        full_exp_headers = expected_survey.headers
        full_exp_headers = full_exp_headers.loc[dump_index:dump_index].reset_index()
        full_dump_headers = Survey(files[0], header_index=header_index, header_cols='all', validate=False).headers
        full_dump_headers = full_dump_headers.reset_index()
        sample_intervals = full_dump_headers['TRACE_SAMPLE_INTERVAL']
        assert np.unique(sample_intervals) > 1
        assert np.allclose(sample_intervals[0] / 1000, expected_survey.sample_interval)
        drop_cols = ["TRACE_SEQUENCE_FILE", "TRACE_SAMPLE_INTERVAL", HDR_TRACE_POS]
        full_exp_headers.drop(columns=drop_cols, inplace=True)
        full_dump_headers.drop(columns=drop_cols, inplace=True)
        assert full_exp_headers.equals(full_dump_headers)


@pytest.mark.parametrize("dump_kwargs, error", [
    ({"path": ""}, FileNotFoundError),
    ({"path": "some_path", "name": ""}, ValueError)
])
def test_dump_single_gather_with_invalid_kwargs(segy_path, dump_kwargs, error):
    survey = Survey(segy_path, header_index='FieldRecord', validate=False)
    gather = survey.get_gather(1)
    with pytest.raises(error):
        gather.dump(**dump_kwargs)


@pytest.mark.parametrize('mode', ['one_folder', 'split'])
@pytest.mark.parametrize('indices', [[1], [1, 3], 'all'])
def test_aggregate_segys(segy_path, tmp_path, mode, indices):
    expected_survey = Survey(segy_path, header_index='FieldRecord', header_cols='all', name='raw', validate=False)
    indices = expected_survey.headers.index.drop_duplicates() if indices == 'all' else indices

    if mode == 'split':
        paths = [f'folder/folder_{i}' for i in range(len(indices))]
    else:
        paths = [''] * len(indices)
    for num, (ix, path) in enumerate(zip(indices, paths)):
        g = expected_survey.get_gather(ix)
        g.dump(os.path.join(tmp_path, path), name=f'{num}_{ix}', retain_parent_segy_headers=True)

    aggregate_segys(os.path.join(tmp_path, './**/*.sgy'), os.path.join(tmp_path, 'aggr.sgy'), recursive=True)

    dumped_survey = Survey(os.path.join(tmp_path, 'aggr.sgy'), header_index='FieldRecord', header_cols='all',
                           validate=False)
    assert np.allclose(expected_survey.samples, dumped_survey.samples),"Samples don't match"
    assert np.allclose(expected_survey.sample_rate, dumped_survey.sample_rate), "Sample rate doesn't match"
    assert np.allclose(expected_survey.n_samples, dumped_survey.n_samples), "length of samples doesn't match"

    #TODO: optimize
    drop_columns = ["TRACE_SEQUENCE_FILE", HDR_TRACE_POS]
    drop_columns += ["TRACE_SAMPLE_INTERVAL"] if "TRACE_SAMPLE_INTERVAL" in expected_survey.headers.columns else []
    expected_survey_headers = (expected_survey.headers.loc[indices].reset_index()
                                                                   .sort_values(['FieldRecord', 'TraceNumber'])
                                                                   .drop(columns=drop_columns)
                                                                   .reset_index(drop=True))
    dumped_survey_headers = (dumped_survey.headers.reset_index()
                                                  .sort_values(['FieldRecord', 'TraceNumber'])
                                                  .drop(columns=drop_columns)
                                                  .reset_index(drop=True))

    assert len(expected_survey_headers) == len(dumped_survey_headers), "Length of surveys' headers don't match"
    assert expected_survey_headers.equals(dumped_survey_headers), "The headers don't match"

    for ix in indices:
        expected_gather = expected_survey.get_gather(ix)
        expected_gather.sort(by='TraceNumber')
        dumped_gather = dumped_survey.get_gather(ix)
        dumped_gather.sort(by='TraceNumber')
        assert_gathers_equal(expected_gather, dumped_gather, drop_cols=drop_columns, same_survey=False)
