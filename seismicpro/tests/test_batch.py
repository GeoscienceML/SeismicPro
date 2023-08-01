"""Implementation of tests for seismic index"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np
from batchflow import Pipeline, L

from seismicpro import Survey, SeismicDataset


@pytest.fixture
def dataset(segy_path):
    """dataset"""
    survey = Survey(segy_path, header_index='FieldRecord', name='raw', validate=False)
    return SeismicDataset(survey)

def test_batch_load(dataset):
    """test_batch_load"""
    batch = dataset.next_batch(1)
    batch.load(src='raw')

def test_batch_load_combined(segy_path):
    """test_batch_load_combined"""
    survey = Survey(segy_path, header_index='TRACE_SEQUENCE_FILE', name='raw', validate=False)
    dataset = SeismicDataset(survey)
    batch = dataset.next_batch(200)
    batch = batch.load(src='raw', combined=True)
    assert len(batch.raw) == 1
    assert len(batch.raw[0].data) == 200


@pytest.mark.parametrize("batch_size", [1, 2])
def test_make_model_inputs(dataset, batch_size):
    """test_batch_make_model_inputs"""
    ppl = (Pipeline()
        .load(src='raw')
        .make_model_inputs(src=L("raw").data, dst="inputs", mode="c", axis=0)
    )
    batch = (dataset >> ppl).next_batch(batch_size, shuffle=False)
    assert np.allclose(batch.inputs, np.concatenate([gather.data for gather in batch.raw]))

@pytest.mark.parametrize("batch_size", [1, 2])
def test_split_model_outputs(dataset, batch_size):
    """test_batch_make_model_inputs"""
    ppl = (Pipeline()
        .load(src='raw')
        .make_model_inputs(src=L("raw").data, dst="inputs", mode="c", axis=0)
        .split_model_outputs(src="inputs", dst="outputs", shapes=L("raw").shape[0])
    )
    batch = (dataset >> ppl).next_batch(batch_size, shuffle=False)
    assert all(np.allclose(gather.data, output) for gather, output in zip(batch.raw, batch.outputs))
