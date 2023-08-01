"""Miscellaneous utility functions for refractor velocity estimation"""

import numpy as np
import pandas as pd

from ..utils import Coordinates, to_list


def get_param_names(n_refractors):
    """Return names of parameters of a near-surface velocity model describing given number of refractors."""
    return ["t0"] + [f"x{i}" for i in range(1, n_refractors)] + [f"v{i}" for i in range(1, n_refractors + 1)]


def postprocess_params(params):
    """Postprocess array of parameters of a near-surface velocity model so that the following constraints are
    satisfied:
    - Intercept time is non-negative,
    - Crossover offsets are non-negative and increasing,
    - Velocities of refractors are non-negative and increasing.
    """
    is_1d = params.ndim == 1

    # Ensure that all params are non-negative
    params = np.clip(np.atleast_2d(params), 0, None)

    # Ensure that velocities of refractors and crossover offsets are non-decreasing
    n_refractors = params.shape[1] // 2
    np.maximum.accumulate(params[:, n_refractors:], axis=1, out=params[:, n_refractors:])
    np.maximum.accumulate(params[:, 1:n_refractors], axis=1, out=params[:, 1:n_refractors])

    if is_1d:
        return params[0]
    return params


def load_refractor_velocities(path, encoding="UTF-8"):
    """Load near-surface velocity models from a file.

    The file should define near-surface velocity models at given field locations and have the following structure:
    - The first row contains names of the coordinates parameters ("name_x", "name_y", "x", "y", "is_uphole_corrected")
      and names of parameters of near-surface velocity models ("t0", "x1"..."x{n-1}", "v1"..."v{n}"). Each velocity
      model must describe the same number of refractors.
    - Each next row contains the corresponding parameters of a single near-surface velocity model.

    File example:
     name_x    name_y         x         y   is_uphole_corrected      t0        x1        v1        v2
    SourceX   SourceY   1111100   2222220                  True   50.25   1000.10   1500.25   2000.10
    ...
    SourceX   SourceY   1111100   2222220                 False   50.50   1000.20   1500.50   2000.20

    Parameters
    ----------
    path : str
        Path to a file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.

    Returns
    -------
    rv_list : list of RefractorVelocity
        A list of loaded near-surface velocity models.
    """
    #pylint: disable-next=import-outside-toplevel
    from .refractor_velocity import RefractorVelocity  # import inside to avoid the circular import
    df = pd.read_csv(path, sep=r'\s+', dtype={"is_uphole_corrected": "string"}, encoding=encoding)
    df["is_uphole_corrected"] = df["is_uphole_corrected"].map({"None": None, "True": True, "False": False})
    params_names = df.columns[5:]
    return [RefractorVelocity(**dict(zip(params_names, row[5:])), coords=Coordinates(row[2:4], row[:2]),
                              is_uphole_corrected=row[4])
            for row in df.itertuples(index=False)]


def dump_refractor_velocities(refractor_velocities, path, encoding="UTF-8"):
    """Dump parameters of passed near-surface velocity models to a file.

    Notes
    -----
    See more about the file format in :func:`~load_refractor_velocities`.

    Parameters
    ----------
    refractor_velocities : RefractorVelocity or iterable of RefractorVelocity
        Near-surface velocity models to be dumped to a file.
    path : str
        Path to the created file.
    encoding : str, optional, defaults to "UTF-8"
        File encoding.
    """
    rv_list = to_list(refractor_velocities)
    df = pd.DataFrame([rv.coords.names for rv in rv_list], columns=["name_x", "name_y"], dtype="string")
    df[["x", "y"]] = pd.DataFrame([rv.coords for rv in rv_list]).convert_dtypes()
    df["is_uphole_corrected"] = [rv.is_uphole_corrected for rv in rv_list]
    df[list(rv_list[0].params.keys())] = pd.DataFrame([rv.params.values() for rv in rv_list])
    df.to_string(buf=path, float_format=lambda x: f"{x:.2f}", index=False, encoding=encoding)
