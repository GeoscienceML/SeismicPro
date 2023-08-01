"""Utils for gather cropping"""

import numpy as np


def _make_grid_origins(data_shape, crop_shape, stride):
    """Calculate evenly-spaced origins along a single axis.

    Parameters
    ----------
    data_shape : int
        Shape of the data to be cropped.
    crop_shape : int
        Shape of the resulting crops.
    stride : int
        A step between two adjacent crops.

    Returns
    -------
    origins : 1d np.ndarray
        An array of crop origins.
    """
    max_origin = max(data_shape - crop_shape, 0)
    return np.array(list(range(0, max_origin, stride)) + [max_origin], dtype=np.int32)


def make_origins(origins, data_shape, crop_shape, n_crops=1, stride=None):
    """Calculate an array of origins or reformat given origins to a 2d `np.ndarray`.

    The returned array has shape [n_origins, 2], where each origin represents a top-left corner of a corresponding crop
    with the shape `crop_shape` from the source data.

    Parameters
    ----------
    origins : list, tuple, np.ndarray or str
        All array-like values are cast to an `np.ndarray` and treated as origins directly, except for a 2-element tuple
        of `int`, which will be treated as a single individual origin.
        If `str`, represents a mode to calculate origins. Two options are supported:
        - "random": calculate `n_crops` crops selected randomly using a uniform distribution over the source data, so
          that no crop crosses data boundaries,
        - "grid": calculate a deterministic uniform grid of origins, whose density is determined by `stride`.
    data_shape : tuple with 2 elements
        Shape of the data to be cropped.
    crop_shape : tuple with 2 elements
        Shape of the resulting crops.
    n_crops : int, optional, defaults to 1
        The number of generated crops if `origins` is "random".
    stride : tuple with 2 elements, optional, defaults to `crop_shape`
        Steps between two adjacent crops along both axes. The lower the value is, the more dense the grid of crops will
        be. An extra origin will always be placed so that the corresponding crop will fit in the very end of an axis to
        guarantee complete data coverage with crops regardless of passed `crop_shape` and `stride`.

    Returns
    -------
    origins : 2d np.ndarray
        An array of absolute coordinates of top-left corners of crops.

    Raises
    ------
    ValueError
        If `origins` is `str`, but not "random" or "grid".
        If `origins` is array-like, but can not be cast to a 2d `np.ndarray` with shape [n_origins, 2].
    """
    if isinstance(origins, str):
        if origins == 'random':
            return np.column_stack((np.random.randint(1 + max(0, data_shape[0] - crop_shape[0]), size=n_crops),
                                    np.random.randint(1 + max(0, data_shape[1] - crop_shape[1]), size=n_crops)))
        if origins == 'grid':
            stride = crop_shape if stride is None else stride
            origins_x = _make_grid_origins(data_shape[0], crop_shape[0], stride[0])
            origins_y = _make_grid_origins(data_shape[1], crop_shape[1], stride[1])
            return np.array(np.meshgrid(origins_x, origins_y)).T.reshape(-1, 2)
        raise ValueError(f"If str, origin should be either 'random' or 'grid' but {origins} was given.")

    origins = np.atleast_2d(origins)
    if origins.ndim == 2 and origins.shape[1] == 2:
        return origins
    raise ValueError("If array-like, origins must be of a shape [n_origins, 2].")
