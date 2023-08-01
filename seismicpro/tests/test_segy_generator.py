""" Tests for segy generator function """

# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
import os
import shutil

import pytest
from seismicpro import Survey, make_prestack_segy


@pytest.fixture(scope="module", params=[
    {"survey_area": (500, 500), "activation_dist": (500, 500), "bin_size": (50, 50), "samples": 1500,
     "sources_step": (300, 50), "receivers_step": (100, 25)},
    {"survey_area": (1200, 1200), "activation_dist": (300, 300), "bin_size": (30, 30), "samples": 100,
     "sources_step": (200, 50), "receivers_step": (100, 25)},
    {"survey_area": (100, 100), "activation_dist": (100, 100), "bin_size": (100, 100), "samples": 1000,
     "sources_step": (300, 50), "receivers_step": (200, 50)},
])
def segy_path(request):
    """ Fixture that creates segy file """
    folder = 'test_tmp'

    os.mkdir(folder)
    path = os.path.join(folder, 'test_prestack.sgy')
    make_prestack_segy(path, **request.param)

    def fin():
        shutil.rmtree(folder)

    request.addfinalizer(fin)
    return path

@pytest.mark.parametrize('header_index',
                         ('FieldRecord', ['INLINE_3D', 'CROSSLINE_3D'], ['GroupX', 'GroupY'], ['SourceX', 'SourceY']))
def test_generated_segy_loading(segy_path, header_index):
    s = Survey(segy_path, header_index=header_index, header_cols=['FieldRecord', 'TraceNumber', 'SourceX', 'SourceY',
                                                                  'GroupX', 'GroupY', 'offset', 'CDP_X', 'CDP_Y',
                                                                  'INLINE_3D', 'CROSSLINE_3D'], validate=False)
    assert s.sample_gather()
