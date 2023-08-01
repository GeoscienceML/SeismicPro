"""Miscellaneous general utility functions"""

from functools import partial
from concurrent.futures import Future, Executor

import numpy as np
import pandas as pd


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D objects, except for `str`, which won't be
    split into separate letters but transformed into a list of a single element."""
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return list(obj)
    return [obj]


def maybe_copy(obj, inplace=False, **kwargs):
    """Copy an object if `inplace` flag is set to `False`. Otherwise return the object unchanged."""
    return obj if inplace else obj.copy(**kwargs)


def align_src_dst(src, dst=None):
    """Cast both `src` and `dst` to lists and check if they have the same lengths. Assume that `dst` equals to `src` if
    not given."""
    src_list = to_list(src)
    dst_list = to_list(dst) if dst is not None else src_list
    if len(src_list) != len(dst_list):
        raise ValueError("src and dst should have the same length.")
    return src_list, dst_list


def unique_indices_sorted(arr):
    """Return indices of the first occurrences of the unique values in a sorted array."""
    mask = np.empty(len(arr), dtype=np.bool_)
    np.any(arr[1:] != arr[:-1], axis=1, out=mask[1:])
    mask[0] = True
    return np.where(mask)[0]


def align_args(reference_arg, *args):
    """Convert `reference_arg` and each arg from `args` to lists so that their lengths match the number of elements in
    the `reference_arg`. If some arg contains a single element, its value will is repeated. If some arg is an
    array-like whose length does not match the number of elements in the `reference_arg` an error is raised."""
    reference_arg = to_list(reference_arg)
    processed_args = []
    for arg in args:
        arg = to_list(arg)
        if len(arg) == 1:
            arg *= len(reference_arg)
        if len(arg) != len(reference_arg):
            raise ValueError("Lengths of all passed arguments must match")
        processed_args.append(arg)
    return reference_arg, *processed_args


def get_first_defined(*args):
    """Return the first non-`None` argument. Return `None` if no `args` are passed or all of them are `None`s."""
    return next((arg for arg in args if arg is not None), None)


def get_cols(df, cols):
    """Extract columns from `cols` from the `df` DataFrame columns or index as a new instance of `pd.DataFrame` if
    `cols` is array-like or `pd.Series` if `cols` is string."""
    if df.columns.nlevels == 1:  # Flat column index
        is_single_col = isinstance(cols, str)
        cols = to_list(cols)
    else:  # MultiIndex case: avoid to_list here since tuples must be preserved as is to be used for column selection
        is_single_col = not isinstance(cols, list)
        if is_single_col:
            cols = [cols]

    # Avoid using direct pandas indexing to speed up selection of multiple columns from small DataFrames
    res = {}
    for col in cols:
        if col in df.columns:
            col_values = df[col]
        elif col in df.index.names:
            col_values = df.index.get_level_values(col)
        else:
            raise KeyError(f"Unknown header {col}")
        res[col] = col_values.to_numpy()

    res = pd.DataFrame(res)
    if is_single_col:
        return res[cols[0]]
    return res


class MissingModule:
    """Postpone raising missing module error for `module_name` until it is being actually accessed in code."""
    def __init__(self, module_name):
        self._module_name = module_name

    def __getattr__(self, name):
        _ = name
        raise ImportError(f"No module named {self._module_name}")

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        raise ImportError(f"No module named {self._module_name}")


class ForPoolExecutor(Executor):
    """A sequential executor of tasks in a for loop. Inherits `Executor` interface thus can serve as a drop-in
    replacement for both `ThreadPoolExecutor` and `ProcessPoolExecutor` when threads or processes spawning is
    undesirable."""

    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self.task_queue = []

    def submit(self, fn, /, *args, **kwargs):
        """Schedule `fn` to be executed with given arguments."""
        future = Future()
        self.task_queue.append((future, partial(fn, *args, **kwargs)))
        return future

    def shutdown(self, *args, **kwargs):
        """Signal the executor to finish all scheduled tasks and free its resources."""
        _ = args, kwargs
        for future, fn in self.task_queue:
            future.set_result(fn())
        self.task_queue = None
