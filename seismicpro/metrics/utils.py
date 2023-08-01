"""Utility functions for coordinates and metric values processing"""

import numpy as np
import pandas as pd

from ..utils import to_list, get_first_defined, Coordinates


def parse_coords(coords, coords_cols=None):
    """Cast given `coords` to a 2d `np.ndarray` with shape [n_coords, 2] and try inferring names of both coordinates if
    `coords_cols` is not passed."""
    if isinstance(coords, pd.DataFrame):
        data_coords_cols = coords.columns.tolist()
        coords = coords.to_numpy()
    elif isinstance(coords, pd.Index):
        data_coords_cols = coords.names
        if None in data_coords_cols:  # Undefined index names, fallback to defaults
            data_coords_cols = None
        coords = coords.to_frame(index=False).to_numpy()
    elif isinstance(coords, (list, tuple, np.ndarray)):
        data_coords_cols = None

        # Try inferring coordinates columns if passed coords is an iterable of Coordinates
        if all(isinstance(coord, Coordinates) for coord in coords):
            data_coords_cols_set = {coord.names for coord in coords}
            if len(data_coords_cols_set) != 1:
                raise ValueError("Coordinates from different header columns were passed")
            data_coords_cols = data_coords_cols_set.pop()

        # Cast coords to an array. If coords is an array of arrays, convert it to an array with numeric dtype.
        coords = np.asarray(coords)
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
    else:
        raise TypeError(f"Unsupported type of coords {type(coords)}")

    coords_cols = get_first_defined(coords_cols, data_coords_cols, ("X", "Y"))
    coords_cols = to_list(coords_cols)
    if len(coords_cols) != 2:
        raise ValueError(f"List of coordinates names must have length 2 but {len(coords_cols)} was given.")
    if coords.ndim != 2:
        raise ValueError("Coordinates array must be 2-dimensional.")
    if coords.shape[1] != 2:
        raise ValueError(f"Each item of coords must have length 2 but {coords.shape[1]} was given.")
    return coords, coords_cols


def parse_index(index, index_cols=None):
    """Cast given `index` to a 2d `np.ndarray` and try inferring names index columns if `index_cols` is not passed."""
    if isinstance(index, pd.DataFrame):
        data_index_cols = index.columns.tolist()
        index = index.to_numpy()
    elif isinstance(index, pd.Index):
        data_index_cols = index.names
        if None in data_index_cols:  # Undefined index names, fallback to defaults
            data_index_cols = None
        index = index.to_frame(index=False).to_numpy()
    elif isinstance(index, (list, tuple, np.ndarray)):
        data_index_cols = None
        index = np.asarray(index)
        if index.ndim == 1:
            index = index.reshape(-1, 1)
        if index.ndim != 2:
            raise ValueError("index must be one- or two-dimensional")
    else:
        raise TypeError(f"Unsupported type of index {type(index)}")

    index_cols = get_first_defined(index_cols, data_index_cols)
    if index_cols is None:
        raise ValueError("Undefined index_cols")
    index_cols = to_list(index_cols)
    if len(index_cols) != index.shape[1]:
        raise ValueError("Length of index_cols must correspond to the shape of index")
    return index, index_cols


def parse_metric_data(values, metric=None):
    """Cast given `values` to a 1d `np.ndarray`. Create a corresponding `Metric` instance with a proper name if not
    given."""
    val_err_msg = "Metric values must be a 1-dimensional array-like."
    if isinstance(values, pd.DataFrame):
        columns = values.columns
        if len(columns) != 1:
            raise ValueError(val_err_msg)
        data_metric_name = columns[0]
        values = values.to_numpy()[:, 0]
    elif isinstance(values, pd.Series):
        data_metric_name = values.name
        values = values.to_numpy()
    else:
        data_metric_name = None
        values = np.array(values)
        if values.ndim != 1:
            raise ValueError(val_err_msg)

    from .metric import Metric  # pylint: disable=import-outside-toplevel
    type_err_msg = "metric must be str or an instance or a subclass of Metric"
    if isinstance(metric, type):
        if not issubclass(metric, Metric):
            raise TypeError(type_err_msg)
        metric = metric()  # Initialize the metric if it was passed as a class, not an instance
    if not isinstance(metric, Metric):
        if metric is not None and not isinstance(metric, str):
            raise TypeError(type_err_msg)
        metric = Metric(name=get_first_defined(metric, data_metric_name))
    return values, metric
