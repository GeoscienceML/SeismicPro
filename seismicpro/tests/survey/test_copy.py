"""Test Survey copying and serialization"""

import pickle

from . import assert_surveys_equal, assert_surveys_not_linked


class TestCopy:
    """Test Survey copying and serialization."""

    def test_pickle(self, survey, tmp_path):
        """Test that survey picking-unpickling does not change the survey."""
        dump_path = tmp_path / "survey"
        with open(dump_path, "wb") as f:
            pickle.dump(survey, f)
        with open(dump_path, "rb") as f:
            survey_restored = pickle.load(f)
        assert_surveys_equal(survey, survey_restored)

    def test_copy(self, survey):
        """Test whether survey copy equals to the survey itself."""
        survey_copy = survey.copy()
        assert_surveys_equal(survey, survey_copy)

    def test_is_copy_deep(self, survey):
        """Test whether `survey.copy` returns a deepcopy by changing some of its mutable attributes."""
        survey_copy = survey.copy()
        assert_surveys_not_linked(survey, survey_copy)
