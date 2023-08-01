"""Test to_list function"""

import pytest

from seismicpro.utils import to_list


def func(*args):
    """Test function, won't be called."""
    _ = args
    return


lambda_func = lambda x: x  # pylint: disable=unnecessary-lambda-assignment


@pytest.mark.parametrize("test_input, expected", [
    # Single argument
    ["str", ["str"]],
    [1, [1]],
    [-2.5, [-2.5]],
    [{"1": 2, 2: "a"}, [{"1": 2, 2: "a"}]],
    [lambda_func, [lambda_func]],

    # Multiple arguments as list or tuple
    [("s1", "s2"), ["s1", "s2"]],
    [(func, func, func), [func, func, func]],

    # Mixed types
    [[1, "str", func], [1, "str", func]],
    [("1", [1, 2], lambda_func), ["1", [1, 2], lambda_func]],

    # Preserve types of inner objects
    [[(1, 2), (3, 4)], [(1, 2), (3, 4)]],
    [((1, 2), (3, 4)), [(1, 2), (3, 4)]],
    [[(0, 1, 2), (3, 4)], [(0, 1, 2), (3, 4)]],
    [[{"key": "val"}, {1: 2}], [{"key": "val"}, {1: 2}]],
    [({"key": "val", 1: 3.0}, {"1", 2}), [{"key": "val", 1: 3.0}, {"1", 2}]],
])
def test_to_list(test_input, expected):
    """Compare `to_list` result with the expected one."""
    assert to_list(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [
    [{0}, [0]],
    [{1, "str", 3.0}, [1, "str", 3.0]],
])
def test_set_to_list(test_input, expected):
    """Test set input to `to_list`. Usually appears as `ignore` arg in `copy`."""
    res = to_list(test_input)
    assert isinstance(res, list)
    assert set(res) == set(expected)
