"""Test indexer classes"""

import pytest
import numpy as np
import pandas as pd

from seismicpro.utils.indexer import create_indexer, UniqueIndexer, NonUniqueIndexer


@pytest.mark.parametrize("index", [
    pd.Index([2, 1, 3]),  # Unique, non-monotonic index
    pd.Index([4, 2, 1, 3, 5, 5]),  # Non-unique, non-monotonic index
    pd.MultiIndex.from_tuples([(1, 2), (3, 4), (0, 0)]),  # Unique, non-monotonic multiindex
    pd.MultiIndex.from_tuples([(1, 2), (3, 4), (0, 0), (1, 2)]),  # Non-unique, non-monotonic multiindex
])
def test_create_indexer_fails(index):
    """Test whether indexer instantiation fails."""
    with pytest.raises(ValueError):
        create_indexer(index)


@pytest.mark.parametrize("index, query_indices, query_pos", [
    [pd.Index([10]), [10], [0]],  # Single-element index
    [pd.Index([1, 2]), [1], [0]],  # Unique monotonically increasing index
    [pd.Index([1, 2, 3]), [1, 3], [0, 2]],  # Unique monotonically increasing index, several requests
    [pd.Index([2, 5, 7]), [7, 2, 5], [2, 0, 1]],  # Unique monotonically increasing index, reversed requests
    [pd.Index([1, 3, 5]), [1, 3, 1], [0, 1, 0]],  # Unique monotonically increasing index, duplicated requests
    [pd.Index([2, 1]), [1, 2], [1, 0]],  # Unique monotonically decreasing index, reversed requests
    [pd.Index([9, 5, 4]), [], []],  # Unique monotonically decreasing index, empty request
    [pd.MultiIndex.from_tuples([(1, 2), (3, 4)]), [(1, 2)], [0]], # Unique monotonically increasing multiindex
    [pd.MultiIndex.from_tuples([(10, 10), (6, 3)]), [(6, 3)], [1]],  # Unique monotonically decreasing multiindex
])
def test_create_unique_indexer(index, query_indices, query_pos):
    """Test whether the correct type of indexer is created and correct positions are returned for a unique `index`."""
    indexer = create_indexer(index)
    assert isinstance(indexer, UniqueIndexer)
    assert np.array_equal(indexer.get_locs_in_indices(query_indices), query_pos)
    assert np.array_equal(indexer.get_locs_in_unique_indices(query_indices), query_pos)


@pytest.mark.parametrize("index, query_indices, query_pos, query_unique_pos", [
    [pd.Index([0, 0]), [0], [0, 1], [0]],  # Single duplicated element
    [pd.Index([0, 0, 1]), [1], [2], [1]],  # Monotonically increasing index
    [pd.Index([5, 2, 1, 1]), [1, 2], [2, 3, 1], [2, 1]],  # Monotonically decreasing index
    [pd.Index([1, 2, 2, 4]), [1, 2, 2, 1], [0, 1, 2, 1, 2, 0], [0, 1, 1, 0]],  # Duplicated request
    [pd.MultiIndex.from_tuples([(3, 3), (3, 3)]), [(3, 3)], [0, 1], [0]],  # Single duplicated element
    [pd.MultiIndex.from_tuples([(0, 0), (0, 0)]), [(0, 0), (0, 0)], [0, 1, 0, 1], [0, 0]],  # Duplicated request
    [pd.MultiIndex.from_tuples([(3, 3), (3, 3), (5, 5)]), [(5, 5)], [2], [1]],  # Monotonically increasing index
    [pd.MultiIndex.from_tuples([(2, 4), (2, 4), (1, 5)]), [(2, 4)], [0, 1], [0]],  # Monotonically decreasing index
])
def test_create_non_unique_indexer(index, query_indices, query_pos, query_unique_pos):
    """Test whether the correct type of indexer is created and correct positions are returned for a non-unique
    `index`."""
    indexer = create_indexer(index)
    assert isinstance(indexer, NonUniqueIndexer)
    assert np.array_equal(indexer.get_locs_in_indices(query_indices), query_pos)
    assert np.array_equal(indexer.get_locs_in_unique_indices(query_indices), query_unique_pos)
