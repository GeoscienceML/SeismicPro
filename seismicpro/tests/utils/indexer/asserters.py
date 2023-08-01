"""General indexer assertions"""

from seismicpro.utils.indexer import UniqueIndexer, NonUniqueIndexer


def assert_indexers_equal(left, right):
    """Check if two indexers are equal."""
    # Check if both left and right are of the same type
    assert type(left) is type(right)

    # Check types of passed indexers
    assert isinstance(left, (UniqueIndexer, NonUniqueIndexer))

    # Compare indexer attributes
    assert left.index.equals(right.index)
    assert left.unique_indices.equals(right.unique_indices)
    if isinstance(left, NonUniqueIndexer):
        assert left.index_to_pos == right.index_to_pos
