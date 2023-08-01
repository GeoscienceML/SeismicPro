"""Test classes for spatial interpolation"""

# pylint: disable=redefined-outer-name
import pickle
from functools import partial

import pytest
import numpy as np

from seismicpro.utils import IDWInterpolator, DelaunayInterpolator, CloughTocherInterpolator, RBFInterpolator


def assert_arrays_equal(arr1, arr2):
    """Check whether two arrays have same shape and close values."""
    assert arr1.shape == arr2.shape
    assert np.allclose(arr1, arr2)


def custom_dist_transform(dist):
    """A user-defined callable distance transformation."""
    return dist * 10


VALUES_AGNOSTIC_INTERPOLATORS = [  # Interpolators that don't require values to be passed to __init__
    IDWInterpolator,  # Use all input coordinates for interpolation
    partial(IDWInterpolator, neighbors=1, dist_transform=2),
    partial(IDWInterpolator, neighbors=3, dist_transform=4),
    partial(IDWInterpolator, radius=0.01),  # force a fallback to neighbors due to a small radius
    partial(IDWInterpolator, radius=10, neighbors=5, dist_transform=custom_dist_transform),
    DelaunayInterpolator,
    partial(DelaunayInterpolator, neighbors=3, dist_transform=np.sqrt),
]


VALUES_AWARE_INTERPOLATORS = [  # Interpolators that require values to be passed to __init__
    CloughTocherInterpolator,
    partial(CloughTocherInterpolator, neighbors=1),
    RBFInterpolator,
]


@pytest.mark.parametrize("interpolator_class", VALUES_AGNOSTIC_INTERPOLATORS + VALUES_AWARE_INTERPOLATORS)
@pytest.mark.parametrize("values_dtype", [np.int32, np.float64])
class TestInterpolate:
    """Test spatial interpolation."""

    def test_picklable(self, interpolator_class, values_dtype):
        """Check if interpolators are picklable."""
        coords = [[0, 0], [1, 0], [0, 1]]
        values = np.array([1, 2, 3], dtype=values_dtype)
        interpolator = interpolator_class(coords, values)
        assert pickle.dumps(interpolator)

    def test_single_value(self, interpolator_class, values_dtype):
        """Test whether interpolation at known points returns corresponding known values if a single value is defined
        for a point."""
        coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
        values = np.array([1, 2, 3, 4], dtype=values_dtype)
        interpolator = interpolator_class(coords, values)
        assert_arrays_equal(interpolator(coords), values)

    def test_multiple_values(self, interpolator_class, values_dtype):
        """Test whether interpolation at known points returns corresponding known values if multiple values are defined
        for a point."""
        coords = [[1, 1], [2, 3], [4, 4], [10, 10]]
        values = np.random.normal(100, 10, size=(len(coords), 2)).astype(values_dtype)
        interpolator = interpolator_class(coords, values)
        assert_arrays_equal(interpolator(coords), values)

    def test_single_query_single_value(self, interpolator_class, values_dtype):
        """Test whether interpolation at a single known point returns the corresponding known value as a scalar if a
        single value is defined for a point."""
        coords = [[-1, -1], [0, 5], [1, 4], [3, -3], [0.5, -2], [0, 0]]
        values = np.arange(len(coords), dtype=values_dtype)
        interpolator = interpolator_class(coords, values)
        assert_arrays_equal(interpolator(coords[0]), values[0])

    def test_single_query_multiple_values(self, interpolator_class, values_dtype):
        """Test whether interpolation at a single known point returns the corresponding known values as a 1d array if
        multiple values are defined for a point."""
        coords = [[100, 50], [200, 300], [15, 40.0], [-10, -50]]
        values = np.random.uniform(10, 20, size=(len(coords), 2)).astype(values_dtype)
        interpolator = interpolator_class(coords, values)
        assert_arrays_equal(interpolator(coords[0]), values[0])

    def test_extrapolation(self, interpolator_class, values_dtype):
        """Test that all interpolators allow for extrapolation and return not-nan results."""
        coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
        values = np.array([1, 2, 3, 4], dtype=values_dtype)
        query = [[-1, -1], [-1, 2], [0.5, 0.5], [2, -1], [2, 2]]
        interpolator = interpolator_class(coords, values)
        extrapolated_values = interpolator(query)
        assert np.isfinite(extrapolated_values).all()


@pytest.mark.parametrize("interpolator_class", [IDWInterpolator, DelaunayInterpolator, CloughTocherInterpolator])
def test_coords_in_subspace(interpolator_class):
    """Test the case when `coords` lie within a 1d subspace of a 2d space. May affect all Delaunay-based
    interpolators. `RBFInterpolator` fails this test with default parameters, appropriate `kernel` and `degree` must
    be chosen manually, see `scipy` docs for more details."""
    coords = np.column_stack([np.arange(10), np.arange(10)])
    values = np.random.normal(scale=25, size=(len(coords), 5))
    interpolator = interpolator_class(coords, values)
    assert_arrays_equal(interpolator(coords), values)


@pytest.mark.parametrize("interpolator_class", VALUES_AGNOSTIC_INTERPOLATORS)
class TestGetWeights:
    """Test `get_weights` method of values-agnostic interpolators."""

    @pytest.mark.parametrize("coords", [
        np.array([[-6.76, 12.7], [13.2, 13.2], [42, -71.5], [11.8, 24]], dtype=np.float32),  # float32 coords
        np.array([[100, 34.4], [124.1, -20.6], [13.0, 66], [1, 1], [4, 8.5]], dtype=np.float64),  # float64 coords
        np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int32),  # int32 coords in a 1d subspace
    ])
    def test_interpolation_at_same_coords(self, interpolator_class, coords):
        """Check that for a known point `get_weights` returns a `dict` with the very same point in keys and a value
        close to 1."""
        interpolator = interpolator_class(coords)
        weighted_coords = interpolator.get_weights(coords)

        assert len(weighted_coords) == len(coords)
        assert all(isinstance(val, dict) for val in weighted_coords)

        # Perform approximate check since sometimes sklearn returns non-zero distance from a point to itself
        closest_coords = np.stack([max(val, key=val.get) for val in weighted_coords])
        assert closest_coords.shape == coords.shape
        assert np.allclose(closest_coords, coords)
        assert all(max(val.values()) > 0.99 for val in weighted_coords)

    def test_extrapolation(self, interpolator_class):
        """Test that during extrapolation `get_weights` returns `dict`s whose keys are a subset of known points and sum
        of values is close to 1."""
        coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
        query = [[-1, -1], [-1, 2], [0.5, 0.5], [2, -1], [2, 2]]
        interpolator = interpolator_class(coords)
        weighted_coords = interpolator.get_weights(query)

        assert len(weighted_coords) == len(query)
        assert all(isinstance(val, dict) for val in weighted_coords)

        unique_keys = set.union(*[set(val.keys()) for val in weighted_coords])
        assert unique_keys <= set(coords)

        values_sum = [sum(val.values()) for val in weighted_coords]
        assert np.allclose(values_sum, 1)
