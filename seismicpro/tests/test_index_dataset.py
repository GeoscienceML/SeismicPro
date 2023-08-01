"""Test SeismicIndex and SeismicDataset instantiation, splitting and stats collection"""

import pytest

from seismicpro import Survey, SeismicIndex, SeismicDataset


HEADER_INDEX = ["FieldRecord", "TRACE_SEQUENCE_FILE", ["INLINE_3D", "CROSSLINE_3D"]]
HEADER_COLS = ["FieldRecord", "TraceNumber"]  # Guarantee that a 1-to-1 merge is performed


@pytest.mark.parametrize("header_index", HEADER_INDEX)
class TestInit:
    """Test `SeismicIndex` and `SeismicDataset` instantiation."""

    def test_from_survey(self, segy_path, header_index):
        """Test instantiation from a single survey."""
        survey = Survey(segy_path, header_index=header_index, validate=False)
        _ = SeismicIndex(survey)

    def test_from_index(self, segy_path, header_index):
        """Test instantiation from an already created index."""
        survey = Survey(segy_path, header_index=header_index, validate=False)
        index = SeismicIndex(survey)
        _ = SeismicIndex(index)

    def test_concat(self, segy_path, header_index):
        """Test concatenation of two surveys."""
        sur1 = Survey(segy_path, header_index=header_index, name="sur", validate=False)
        sur2 = Survey(segy_path, header_index=header_index, name="sur", validate=False)
        _ = SeismicIndex(sur1, sur2, mode="c")

    def test_concat_wrong_names_fails(self, segy_path, header_index):
        """Concat must fail if surveys have different names."""
        sur1 = Survey(segy_path, header_index=header_index, name="sur", validate=False)
        sur2 = Survey(segy_path, header_index=header_index, name="not_sur", validate=False)
        with pytest.raises(ValueError):
            _ = SeismicIndex(sur1, sur2, mode="c")

    def test_concat_wrong_index_fails(self, segy_path, header_index):
        """Concat must fail if surveys are indexed by different headers."""
        sur1 = Survey(segy_path, header_index=header_index, name="sur", validate=False)
        sur2 = Survey(segy_path, header_index="CDP", name="sur", validate=False)
        with pytest.raises(ValueError):
            _ = SeismicIndex(sur1, sur2, mode="c")

    def test_merge(self, segy_path, header_index):
        """Test merging of two surveys."""
        sur1 = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="before", validate=False)
        sur2 = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="after", validate=False)
        _ = SeismicIndex(sur1, sur2, mode="m")

    def test_merge_wrong_names_fails(self, segy_path, header_index):
        """Merge must fail if surveys have same names."""
        sur1 = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="sur", validate=False)
        sur2 = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="sur", validate=False)
        with pytest.raises(ValueError):
            _ = SeismicIndex(sur1, sur2, mode="m")

    def test_merge_wrong_index_fails(self, segy_path, header_index):
        """Merge must fail if surveys are indexed by different headers."""
        sur1 = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="before", validate=False)
        sur2 = Survey(segy_path, header_index="CDP", header_cols=HEADER_COLS, name="after", validate=False)
        with pytest.raises(ValueError):
            _ = SeismicIndex(sur1, sur2, mode="m")

    def test_merge_concat(self, segy_path, header_index):
        """Test merge followed by concat."""
        s1_before = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="before",
                           validate=False)
        s2_before = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="before",
                           validate=False)

        s1_after = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="after", validate=False)
        s2_after = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="after", validate=False)

        index_s1 = SeismicIndex(s1_before, s1_after, mode="m")
        index_s2 = SeismicIndex(s2_before, s2_after, mode="m")
        _ = SeismicIndex(index_s1, index_s2, mode="c")

    def test_concat_merge(self, segy_path, header_index):
        """Test concat followed by merge."""
        s1_before = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="before",
                           validate=False)
        s2_before = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="before",
                           validate=False)

        s1_after = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="after", validate=False)
        s2_after = Survey(segy_path, header_index=header_index, header_cols=HEADER_COLS, name="after", validate=False)

        index_before = SeismicIndex(s1_before, s2_before, mode="c")
        index_after = SeismicIndex(s1_after, s2_after, mode="c")
        _ = SeismicIndex(index_before, index_after, mode="m")


@pytest.mark.parametrize("header_index", HEADER_INDEX)
@pytest.mark.parametrize("test_class", [SeismicIndex, SeismicDataset])
def test_index_split(test_class, segy_path, header_index):
    """Test whether index or dataset `split` runs."""
    survey = Survey(segy_path, header_index=header_index, bar=False, validate=False)
    test_obj = test_class(survey)
    test_obj.split()


@pytest.mark.parametrize("header_index", HEADER_INDEX)
@pytest.mark.parametrize("test_class", [SeismicIndex, SeismicDataset])
def test_index_collect_stats(test_class, segy_path, header_index):
    """Test whether index or dataset `collect_stats` runs."""
    survey = Survey(segy_path, header_index=header_index, bar=False, validate=False)
    test_obj = test_class(survey)
    test_obj.collect_stats()


@pytest.mark.parametrize("header_index", HEADER_INDEX)
@pytest.mark.parametrize("test_class", [SeismicIndex, SeismicDataset])
def test_index_split_collect_stats(test_class, segy_path, header_index):
    """Test whether `collect_stats` runs for a subset of index or dataset."""
    survey = Survey(segy_path, header_index=header_index, bar=False, validate=False)
    test_obj = test_class(survey)
    test_obj.split(0.5)
    test_obj.train.collect_stats()
