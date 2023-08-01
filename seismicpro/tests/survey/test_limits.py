"""Test Survey.set_limits method"""

import pytest

from . import assert_survey_limits_set
from ..conftest import N_SAMPLES


@pytest.mark.parametrize("limits, expected_limits", [
    # None equals to loading of whole traces
    (None, slice(0, N_SAMPLES, 1)),

    # Ints and tuples are converted to a corresponding slice
    (10, slice(0, 10, 1)),
    ((100, 200), slice(100, 200, 1)),
    ((100, 500, 5), slice(100, 500, 5)),
    ((None, 200, 3), slice(0, 200, 3)),

    # Slices with positive attributes are passed as-is
    (slice(700, 800), slice(700, 800, 1)),
    (slice(400, None, 4), slice(400, N_SAMPLES, 4)),

    # Handle negative bounds (note that that each trace has N_SAMPLES samples)
    (-100, slice(0, 900, 1)),
    (slice(0, -100), slice(0, N_SAMPLES - 100, 1)),
    (slice(-200, -100), slice(N_SAMPLES - 200, N_SAMPLES - 100, 1)),
    (slice(-200), slice(0, N_SAMPLES - 200, 1)),
])
def test_set_limits(survey, limits, expected_limits):
    """Test `Survey.set_limits` with a broad range of possible `limits` passed."""
    survey.set_limits(limits)
    assert_survey_limits_set(survey, expected_limits)


@pytest.mark.parametrize("limits", [
    # Negative step is not allowed
    (200, 100, -2),
    slice(-100, -500, -1),

    # Slicing must not return empty traces
    slice(-100, -200),
    slice(500, 100),
])
def test_set_limits_fails(survey, limits):
    """`set_limits` must fail if negative step is passed or empty traces are returned."""
    with pytest.raises(ValueError):
        survey.set_limits(limits)
