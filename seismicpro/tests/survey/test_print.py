"""Test Survey printing routines"""


class TestPrint:
    """Test `__str__` magic and `info` method."""

    def test_str(self, survey):
        """Test that `str(survey)` works both when stats are collected and not."""
        assert str(survey)

    def test_str_prints_stats(self, survey_no_stats):
        """Test that extra lines are generated if stats are collected."""
        no_stats_str = str(survey_no_stats)
        stats_str = str(survey_no_stats.collect_stats())
        assert len(stats_str) > len(no_stats_str)
        assert stats_str.startswith(no_stats_str)

    def test_info_matches_str(self, survey, capsys):
        """Test that `print(survey)` equals to `survey.info()`."""
        survey.info()
        stdout = capsys.readouterr().out
        assert str(survey) + "\n" == stdout
