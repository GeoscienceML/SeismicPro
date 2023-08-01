"""Test univariate interpolation"""

import pytest
import numpy as np
from scipy.interpolate import interp1d as sp_interp1d

from seismicpro.utils import interp1d


@pytest.mark.parametrize("func, coords, eval_coords", [
    [lambda x: x, [0, 1], 0.5],  # single value to evaluate the interpolator at
    [np.exp, np.linspace(-10, 10, 10), np.linspace(-10, 10, 30)],  # interpolation inside data range
    [np.sin, np.arange(5), np.linspace(-10, 10, 10)],  # extrapolation outside data range
    [lambda x: 3 * np.log(np.square(x) + 10), np.linspace(-1, 1, 10), np.random.normal(size=10)],  # add randomization
])
def test_interp1d(func, coords, eval_coords):
    """Check if `interp1d` returns the same results as `scipy.interpolate.interp1d` with linear extrapolation outside
    the data range."""
    values = func(coords)
    interp = interp1d(coords, values)
    sp_interp = sp_interp1d(coords, values, fill_value="extrapolate")
    assert np.allclose(interp(eval_coords), sp_interp(eval_coords))
