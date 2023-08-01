"""Implements classes to speed up `DataFrame` indexing.

Multiple `DataFrame`s in `SeismicPro`, such as headers of surveys, have sorted but non-unique index. In this case
`pandas` uses binary search with `O(logN)` complexity while performing lookups. Indexing can be significantly
accelerated by utilizing a `dict` which maps each index value to a range of its positions resulting in `O(1)` lookups.
"""

from itertools import chain

import numpy as np


class BaseIndexer:
    """Base indexer class."""
    def __init__(self, index):
        self.index = index
        self.unique_indices = None

    def get_locs_in_indices(self, indices):
        """Get locations of `indices` values in the source index."""
        _ = indices
        raise NotImplementedError

    def get_locs_in_unique_indices(self, indices):
        """Get locations of `indices` values in unique indices of the source index.

        Parameters
        ----------
        indices : array-like
            Indices to get locations for.

        Returns
        -------
        locations : np.ndarray
            Locations of the requested indices.
        """
        return self.unique_indices.get_indexer(indices)


class UniqueIndexer(BaseIndexer):
    """Construct an indexer for unique monotonic `index`. Should not be instantiated directly, use `create_indexer`
    function instead.
    """
    def __init__(self, index):
        super().__init__(index)
        self.unique_indices = index

        # Warmup of `get_locs_in_indices`: the first call is way slower than the following ones. Running
        # `get_locs_in_unique_indices` is unnecessary since both index and unique_indices refer to the same object.
        _ = self.get_locs_in_indices(index[:1])

    def get_locs_in_indices(self, indices):
        """Get locations of `indices` values in the source index.

        Parameters
        ----------
        indices : array-like
            Indices to get locations for.

        Returns
        -------
        locations : np.ndarray
            Locations of the requested indices.
        """
        return self.index.get_indexer(indices)


class NonUniqueIndexer(BaseIndexer):
    """Construct an indexer for monotonic, but non-unique `index`. Should not be instantiated directly, use
    `create_indexer` function instead.
    """
    def __init__(self, index):
        super().__init__(index)
        unique_indices_pos = np.where(~index.duplicated())[0]
        ix_start = unique_indices_pos
        ix_end = chain(unique_indices_pos[1:], [len(index)])
        self.unique_indices = index[unique_indices_pos]
        self.index_to_pos = {ix: range(*args) for ix, *args in zip(self.unique_indices, ix_start, ix_end)}

        # Warmup of `get_locs_in_unique_indices`: the first call is way slower than the following ones
        _ = self.get_locs_in_unique_indices(index[:1])

    def get_locs_in_indices(self, indices):
        """Get locations of `indices` values in the source index.

        Parameters
        ----------
        indices : array-like
            Indices to get locations for.

        Returns
        -------
        locations : np.ndarray
            Locations of the requested indices.
        """
        return list(chain.from_iterable(self.index_to_pos[item] for item in indices))


def create_indexer(index):
    """Construct an appropriate indexer for the passed `index`:
    * If `index` is monotonic and unique, default `pandas` indexer is used,
    * If `index` is monotonic but non-unique, an extra mapping from each `index` value to a range of its positions is
      constructed to speed up lookups from `O(logN)` to `O(1)`,
    * If `index` is non-monotonic, an error is raised.

    The returned object provides fast indexing interface via its `get_locs_in_indices` and `get_locs_in_unique_indices`
    methods.

    Parameters
    ----------
    index : pd.Index
        An index to construct an indexer for.

    Returns
    -------
    indexer : BaseIndexer
        The constructed indexer.
    """
    if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
        raise ValueError("Indexer can be created only for monotonic indices")
    if index.is_unique:
        return UniqueIndexer(index)
    return NonUniqueIndexer(index)
