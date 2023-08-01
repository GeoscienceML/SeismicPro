"""Utilities for processing of vertical functions"""

import numpy as np

from .general_utils import to_list
from .interpolation import interp1d
from .coordinates import Coordinates


def read_vfunc(path, coords_cols=("INLINE_3D", "CROSSLINE_3D"), encoding="UTF-8"):
    """Read a file with vertical functions in Paradigm Echos VFUNC format.

    The file may have one or more records with the following structure:
    VFUNC [coord_x] [coord_y]
    [x1] [y1] [x2] [y2] ... [xn] [yn]

    Parameters
    ----------
    path : str
        A path to the file.
    coords_cols : tuple with 2 elements, optional, defaults to ("INLINE_3D", "CROSSLINE_3D")
        Names of SEG-Y trace headers representing coordinates of the VFUNC.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    vfunc_list : list of tuples with 3 elements
        List of loaded vertical functions. Each of them is a tuple containing coordinates as a `Coordinates` object and
        two 1d `np.ndarray`s of the same length representing `x` and `y` fields respectively.

    Raises
    ------
    ValueError
        If data length for any VFUNC record is odd.
    """
    vfunc_list = []
    with open(path, encoding=encoding) as file:
        for data in file.read().split("VFUNC")[1:]:
            data = data.split()
            coords = Coordinates((int(data[0]), int(data[1])), names=coords_cols)
            data = np.array(data[2:], dtype=np.float64)
            if len(data) % 2 != 0:
                raise ValueError("Data length for each VFUNC record must be even")
            vfunc_list.append((coords, data[::2], data[1::2]))
    return vfunc_list


def read_single_vfunc(path, coords_cols=("INLINE_3D", "CROSSLINE_3D"), encoding="UTF-8"):
    """Read a single vertical function from a file in Paradigm Echos VFUNC format.

    The file must have exactly one record with the following structure:
    VFUNC [coord_x] [coord_y]
    [x1] [y1] [x2] [y2] ... [xn] [yn]

    Parameters
    ----------
    path : str
        A path to the file.
    coords_cols : tuple with 2 elements, optional, defaults to ("INLINE_3D", "CROSSLINE_3D")
        Names of SEG-Y trace headers representing coordinates of the VFUNC.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    vfunc : tuple with 3 elements
        Coordinates of the vertical function as an instance of `Coordinates` and two 1d `np.ndarray`s of the same
        length representing `x` and `y` fields respectively.

    Raises
    ------
    ValueError
        If data length for VFUNC record is odd.
        If the file does not contain a single vfunc.
    """
    file_data = read_vfunc(path, coords_cols=coords_cols, encoding=encoding)
    if len(file_data) != 1:
        raise ValueError(f"Input file must contain a single vfunc, but {len(file_data)} were found in {path}")
    return file_data[0]


def dump_vfunc(path, vfunc_list, encoding="UTF-8"):
    """Dump vertical functions in Paradigm Echos VFUNC format to a file.

    Each passed VFUNC is a tuple with 3 elements: `coords`, `x` and `y`, where `coords` is an array-like with 2
    elements while `x` and `y` are 1d `np.ndarray`s with the same length. A block with the following structure is
    created in the resulting file for each VFUNC:
    - The first row contains 3 values: VFUNC [coords[0]] [coords[1]],
    - All other rows represent pairs of `x` and corresponding `y` values: [x1] [y1] [x2] [y2] ...
      Each row contains 4 pairs, except for the last one, which may contain less. Each value is left aligned with the
      field width of 8.

    Block example:
    VFUNC   22      33
    17      1546    150     1530    294     1672    536     1812
    760     1933    960     2000    1202    2148    1374    2251
    1574    2409    1732    2517    1942    2675

    Parameters
    ----------
    path : str
        A path to the created file.
    vfunc_list : iterable of tuples with 3 elements
        Coordinates and data values for each vertical function.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.
    """
    with open(path, "w", encoding=encoding) as f:
        for coords, x, y in vfunc_list:
            f.write(f"{'VFUNC':8}{round(coords[0]):<8}{round(coords[1]):<8}\n")
            data = np.column_stack([x, y]).ravel()
            rows = np.split(data, np.arange(8, len(data), 8))
            for row in rows:
                f.write("".join(f"{i:<8.0f}" for i in row) + "\n")


class VFUNC:
    """A class representing a single vertical function.

    A vertical function (or VFUNC) is a piecewise linear function, whose knots are defined by two arrays: `data_x` and
    `data_y`. VFUNC instances are callable and return interpolated `y` values for given values of `x`. Values outside
    the `data_x` range are linearly extrapolated.
    """
    def __init__(self, data_x, data_y, coords=None):
        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        self.validate_data()
        self.interpolator = interp1d(self.data_x, self.data_y)
        if coords is not None and not isinstance(coords, Coordinates):
            raise ValueError("coords must be either None or an instance of Coordinates")
        self.coords = coords

    def validate_data(self):
        """Validate whether `data_x` and `data_y` are 1d arrays of the same shape."""
        if self.data_x.ndim != 1 or self.data_x.shape != self.data_y.shape:
            raise ValueError("Inconsistent shapes of times and velocities")

    @property
    def has_coords(self):
        """bool: Whether VFUNC coordinates are not-None."""
        return self.coords is not None

    @classmethod
    def from_vfuncs(cls, vfuncs, weights=None, coords=None):
        """Init a vertical function by averaging other vertical functions with given weights.

        Parameters
        ----------
        vfuncs : VFUNC or list of VFUNC
            Vertical functions to be aggregated.
        weights : float or list of floats, optional
            Weight of each item in `vfuncs`. Normalized to have sum of 1 before aggregation. If not given, equal
            weights are assigned to all items and thus mean vertical function is calculated.
        coords : Coordinates, optional
            Spatial coordinates of the created vertical function.

        Returns
        -------
        self : VFUNC
            Created vertical function.
        """
        vfuncs = to_list(vfuncs)
        data_x = np.unique(np.concatenate([vfunc.data_x for vfunc in vfuncs]))
        data_y = np.average([vfunc(data_x) for vfunc in vfuncs], axis=0, weights=weights)
        return cls(data_x, data_y, coords=coords)

    @classmethod
    def from_file(cls, path, coords_cols=("INLINE_3D", "CROSSLINE_3D"), encoding="UTF-8"):
        """Init a vertical function from a file in Paradigm Echos VFUNC format.

        The file must have exactly one record with the following structure:
        VFUNC [coord_x] [coord_y]
        [x1] [y1] [x2] [y2] ... [xn] [yn]

        Parameters
        ----------
        path : str
            A path to the file.
        coords_cols : tuple with 2 elements, optional, defaults to ("INLINE_3D", "CROSSLINE_3D")
            Names of SEG-Y trace headers representing coordinates of the VFUNC.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.

        Returns
        -------
        self : VFUNC
            Loaded vertical function.
        """
        coords, data_x, data_y = read_single_vfunc(path, coords_cols=coords_cols, encoding=encoding)
        return cls(data_x, data_y, coords=coords)

    def dump(self, path, encoding="UTF-8"):
        """Dump the vertical function to a file in Paradigm Echos VFUNC format.

        Notes
        -----
        See more about the format in :func:`~dump_vfunc`.

        Parameters
        ----------
        path : str
            A path to the created file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        """
        if not self.has_coords:
            raise ValueError("VFUNC instance can be dumped only if it has well-defined coordinates")
        dump_vfunc(path, [(self.coords, self.data_x, self.data_y)], encoding=encoding)

    def __call__(self, data_x):
        """Evaluate the vertical function at given points."""
        return self.interpolator(data_x)
