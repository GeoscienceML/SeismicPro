"""Test Survey class instantiation"""

import pytest
import segyio
import segfast

from seismicpro import Survey
from seismicpro.const import HDR_TRACE_POS

from . import assert_survey_loaded, assert_surveys_equal, assert_survey_limits_set
from ..conftest import FILE_NAME, N_SAMPLES


ALL_HEADERS = set(segyio.tracefield.keys.keys()) - {"UnassignedInt1", "UnassignedInt2"}


HEADER_INDEX = [
    # Single header index passed as a string, list or tuple:
    ["TRACE_SEQUENCE_FILE", {"TRACE_SEQUENCE_FILE"}],
    [("FieldRecord",), {"FieldRecord"}],
    [["CDP"], {"CDP"}],

    # Multiple header indices passed as a list or tuple:
    [["FieldRecord", "TraceNumber"], {"FieldRecord", "TraceNumber"}],
    [("INLINE_3D", "CROSSLINE_3D"), {"INLINE_3D", "CROSSLINE_3D"}],
]


HEADER_COLS = [
    # Don't load extra headers
    [None, set()],

    # Load all SEG-Y headers
    ["all", ALL_HEADERS],

    # Load a single extra header passed as a string, list or tuple:
    ["offset", {"offset"}],
    [["offset"], {"offset"}],
    [("offset",), {"offset"}],

    # Load several extra headers passed as a list or a tuple:
    [["offset", "SourceDepth"], {"offset", "SourceDepth"}],
    [("offset", "SourceDepth"), {"offset", "SourceDepth"}],

    # Load several extra headers with possible intersection with index
    [["offset", "INLINE_3D", "CROSSLINE_3D"], {"offset", "INLINE_3D", "CROSSLINE_3D"}],
]


NAME = [  # passed survey name and expected name
    [None, FILE_NAME],  # Use file name if survey name is not passed
    ["raw", "raw"],  # Use passed name otherwise
]


LIMITS = [  # passed samples limits and expected limits with positive start and stop
    (None, slice(0, N_SAMPLES, 1)),
    (10, slice(0, 10, 1)),
    (slice(100, -100), slice(100, N_SAMPLES - 100, 1)),
]


WORKERS = [  # headers chunk size, number of workers and progress bar display flag
    [1, 1, True],  # Tracewise loading, single worker, bar enabled
    [10, 2, True],  # Small chunk size, 2 workers, bar enabled
    [10, None, False],  # Small chunk size, os.cpu_count() workers, bar disabled
    [10000000, None, False],  # Chunk size larger than the number of traces, os.cpu_count() workers, bar disabled
]


class TestInit:
    """Test `Survey` instantiation."""

    @pytest.mark.parametrize("engine, loader_type", [
        ["segyio", segfast.SegyioLoader],
        ["memmap", segfast.MemmapLoader],
    ])
    @pytest.mark.parametrize("chunk_size, n_workers, bar", WORKERS)
    @pytest.mark.parametrize("validate", [True, False])
    def test_headers_loading(self, segy_path, engine, loader_type, chunk_size, n_workers, bar, validate):
        """Test sequential and parallel loading of survey trace headers."""
        survey = Survey(segy_path, header_index="FieldRecord", header_cols="all", name="raw", engine=engine,
                        chunk_size=chunk_size, n_workers=n_workers, bar=bar, validate=validate)
        assert isinstance(survey.loader, loader_type)
        assert_survey_loaded(survey, segy_path, "raw", {"FieldRecord"}, ALL_HEADERS | {HDR_TRACE_POS})

    @pytest.mark.parametrize("header_index, expected_index", HEADER_INDEX)
    @pytest.mark.parametrize("header_cols, expected_cols", HEADER_COLS)
    @pytest.mark.parametrize("name, expected_name", NAME)
    def test_no_limits(self, segy_path, header_index, expected_index, header_cols, expected_cols, name, expected_name):
        """Test survey loading when limits are not passed."""
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, n_workers=1,
                        bar=False, validate=False)

        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE", HDR_TRACE_POS}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that whole traces are loaded
        limits = slice(0, survey.n_file_samples, 1)
        assert_survey_limits_set(survey, limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False
        assert isinstance(survey.qc_metrics, dict) and not survey.qc_metrics

    @pytest.mark.parametrize("header_index, expected_index", HEADER_INDEX)
    @pytest.mark.parametrize("header_cols, expected_cols", HEADER_COLS)
    @pytest.mark.parametrize("name, expected_name", NAME)
    @pytest.mark.parametrize("limits, slice_limits", LIMITS)
    def test_limits(self, segy_path, header_index, expected_index, header_cols, expected_cols, name, expected_name,
                    limits, slice_limits):
        """Test survey loading with limits set."""
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, limits=limits,
                        n_workers=1, bar=False, validate=False)

        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE", HDR_TRACE_POS}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that correct limits were set
        assert_survey_limits_set(survey, slice_limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False
        assert isinstance(survey.qc_metrics, dict) and not survey.qc_metrics

        # Check that passing limits to init is identical to running set_limits method
        other = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, validate=False)
        other.set_limits(limits)
        assert_surveys_equal(survey, other)
