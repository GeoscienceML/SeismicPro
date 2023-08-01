"""Test Survey reindexation"""

import pytest
import pandas as pd

from seismicpro.const import HDR_TRACE_POS
from . import assert_surveys_equal, assert_survey_processed_inplace


class TestReindex:
    """Test Survey reindexation."""

    @pytest.mark.parametrize("inplace", [True, False])
    def test_reindex_to_current_index(self, survey, inplace):
        """Reindexation to the current index must return the same survey."""
        index = survey.headers.index.names
        survey_reindexed = survey.reindex(index, inplace=inplace)
        assert_surveys_equal(survey, survey_reindexed)
        assert_survey_processed_inplace(survey, survey_reindexed, inplace)

    @pytest.mark.parametrize("new_index", ["CDP", ["CDP_X", "CDP_Y"]])
    def test_reindex_and_back(self, survey, new_index):
        """Reindexation to a new index and back must return the same survey."""
        index = survey.headers.index.names
        survey_reindexed = survey.reindex(new_index, inplace=False).reindex(index, inplace=False)
        assert_surveys_equal(survey, survey_reindexed, ignore_column_order=True, ignore_dtypes=True)

    @pytest.mark.parametrize("first_index", ["TRACE_SEQUENCE_FILE", ["CDP_X", "CDP_Y"]])
    @pytest.mark.parametrize("second_index", ["CDP", ["FieldRecord", "TraceNumber"]])
    def test_consecutive_reindex(self, survey, first_index, second_index):
        """Several consequent reindexations must be equal to reindexing to the last one specified."""
        survey_first_second = survey.reindex(first_index, inplace=False).reindex(second_index, inplace=False)
        survey_second = survey.reindex(second_index, inplace=False)
        assert_surveys_equal(survey_first_second, survey_second, ignore_column_order=True, ignore_dtypes=True)

    @pytest.mark.parametrize("new_index", ["TRACE_SEQUENCE_FILE", ["CDP_X", "CDP_Y"]])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_reindex(self, survey, new_index, inplace):
        """Validate `reindex`:
        1. Requested index is set,
        2. The index in monotonically increasing,
        3. The set of loaded SEG-Y headers has not changed,
        4. Values of trace headers has not changed, only their order in `survey.headers`.
        """
        survey_copy = survey.copy()
        survey_headers = set(survey.headers.index.names) | set(survey.headers.columns)
        survey_reindexed = survey.reindex(new_index, inplace=inplace)

        if isinstance(new_index, str):
            new_index = [new_index]
        assert survey_reindexed.headers.index.names == new_index
        assert survey_reindexed.headers.index.is_monotonic_increasing
        assert set(survey_reindexed.headers.index.names) | set(survey_reindexed.headers.columns) == survey_headers

        # Check that only order of rows in headers has changed
        merged_headers = pd.merge(survey_copy.headers.reset_index(), survey_reindexed.headers.reset_index(),
                                  on=list(survey_headers - {HDR_TRACE_POS}))
        assert len(merged_headers) == len(survey_copy.headers)

        # Check that all other attributes has not changed
        survey_copy.headers = survey_reindexed.headers
        assert_surveys_equal(survey_copy, survey_reindexed)
        assert_survey_processed_inplace(survey, survey_reindexed, inplace)
