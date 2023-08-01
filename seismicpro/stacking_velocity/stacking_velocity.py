"""Implements a StackingVelocity class which allows for velocity interpolation at given times"""

import numpy as np

from .velocity_model import calculate_stacking_velocity
from ..utils import to_list, VFUNC


class StackingVelocity(VFUNC):
    """A class representing stacking velocity at a certain point of a field.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    It can be instantiated directly by passing arrays of times and velocities defining knots of a piecewise linear
    velocity function or created from other types of data by calling a corresponding `classmethod`:
    * `from_file` - from a file in VFUNC format with time-velocity pairs,
    * `from_constant_velocity` - from a single velocity returned for all times,
    * `from_stacking_velocities` - from other stacking velocities with given weights.

    However, usually a stacking velocity instance is not created directly, but is obtained as a result of calling the
    following methods:
    * :func:`~velocity_spectrum.VerticalVelocitySpectrum.calculate_stacking_velocity` - to run an automatic algorithm
      for stacking velocity computation by vertical velocity spectrum,
    * :func:`StackingVelocityField.__call__` - to interpolate a stacking velocity at passed field coordinates given a
      created or loaded velocity field.

    The resulting object is callable and returns stacking velocities for given times.

    Examples
    --------
    Stacking velocity can be automatically calculated for a CDP gather by its velocity spectrum:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> velocity_spectrum = gather.calculate_vertical_velocity_spectrum()
    >>> velocity = velocity_spectrum.calculate_stacking_velocity()

    Or it can be interpolated from a velocity field (loaded from a file in this case):
    >>> field = StackingVelocityField.from_file(field_path).create_interpolator("idw")
    >>> coords = (inline, crossline)
    >>> velocity = field(coords)

    Parameters
    ----------
    times : 1d array-like
        An array with time values for which stacking velocity was picked. Measured in milliseconds.
    velocities : 1d array-like
        An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
    coords : Coordinates or None, optional, defaults to None
        Spatial coordinates of the stacking velocity. If not given, the created instance won't be able to be added to a
        `StackingVelocityField`.

    Attributes
    ----------
    data_x : 1d np.ndarray
        An array with time values for which stacking velocity was picked. Measured in milliseconds.
    data_y : 1d np.ndarray
        An array with stacking velocity values, matching the length of `data_x`. Measured in meters/seconds.
    interpolator : callable
        An interpolator returning velocity value by given time.
    coords : Coordinates or None
        Spatial coordinates of the stacking velocity.
    bounds : list of two StackingVelocity or None
        Left and right bounds of an area for stacking velocity picking. Defined only if the stacking velocity was
        created using `from_vertical_velocity_spectrum`.
    """
    def __init__(self, times, velocities, coords=None):
        super().__init__(times, velocities, coords=coords)
        self.bounds = None

    @property
    def times(self):
        """1d np.ndarray: An array with time values for which stacking velocity was picked. Measured in
        milliseconds."""
        return self.data_x

    @property
    def velocities(self):
        """1d np.ndarray: An array with stacking velocity values, matching the length of `times`. Measured in
        meters/seconds."""
        return self.data_y

    def validate_data(self):
        """Validate whether `times` and `velocities` are 1d arrays of the same shape and all stacking velocities are
        positive."""
        super().validate_data()
        if (self.velocities < 0).any():
            raise ValueError("Velocity values must be positive")

    @classmethod
    def from_stacking_velocities(cls, velocities, weights=None, coords=None):
        """Init stacking velocity by averaging other stacking velocities with given weights.

        Parameters
        ----------
        velocities : StackingVelocity or list of StackingVelocity
            Stacking velocities to be aggregated.
        weights : float or list of floats, optional
            Weight of each item in `velocities`. Normalized to have sum of 1 before aggregation. If not given, equal
            weights are assigned to all items and thus mean stacking velocity is calculated.
        coords : Coordinates, optional
            Spatial coordinates of the created stacking velocity. If not given, the created instance won't be able to
            be added to a `StackingVelocityField`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        return cls.from_vfuncs(velocities, weights, coords)

    @classmethod
    def from_constant_velocity(cls, velocity, coords=None):
        """Init stacking velocity from a single velocity returned for all times.

        Parameters
        ----------
        velocity : float
            Stacking velocity returned for all times.
        coords : Coordinates, optional
            Spatial coordinates of the created stacking velocity. If not given, the created instance won't be able to
            be added to a `StackingVelocityField`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        return cls([0, 10000], [velocity, velocity], coords=coords)

    @classmethod
    def from_vertical_velocity_spectrum(cls, spectrum, init=None, bounds=None, relative_margin=0.2,
                                        acceleration_bounds="auto", times_step=100, max_offset=5000,
                                        hodograph_correction_step=25, max_n_skips=2):
        """Calculate stacking velocity by vertical velocity spectrum.

        Notes
        -----
        A detailed description of the proposed algorithm and its implementation can be found in
        :func:`~velocity_model.calculate_stacking_velocity` docs.

        Parameters
        ----------
        init : StackingVelocity, optional
            A rough estimate of the stacking velocity being picked. Used to calculate `bounds` as
            [`init` * (1 - `relative_margin`), `init` * (1 + `relative_margin`)] if they are not given.
        bounds : array-like of two StackingVelocity, optional
            Left and right bounds of an area for stacking velocity picking. If not given, `init` must be passed.
        relative_margin : positive float, optional, defaults to 0.2
            A fraction of stacking velocities defined by `init` used to estimate `bounds` if they are not given.
        acceleration_bounds : tuple of two positive floats or "auto" or None, optional
            Minimal and maximal acceleration allowed for the stacking velocity function. If "auto", equals to the range
            of accelerations of stacking velocities in `bounds` extended by 50% in both directions. If `None`, only
            ensures that picked stacking velocity is monotonically increasing. Measured in meters/seconds^2.
        times_step : float, optional, defaults to 100
            A difference between two adjacent times defining graph nodes.
        max_offset : float, optional, defaults to 5000
            An offset for hodograph time estimation. Used to create graph nodes and calculate their velocities for each
            time.
        hodograph_correction_step : float, optional, defaults to 25
            The maximum difference in arrival time of two hodographs starting at the same zero-offset time and two
            adjacent velocities at `max_offset`. Used to create graph nodes and calculate their velocities for each
            time.
        max_n_skips : int, optional, defaults to 2
            Defines the maximum number of intermediate times between two nodes of the graph. Greater values increase
            computational costs, but tend to produce smoother stacking velocity.

        Returns
        -------
        stacking_velocity : StackingVelocity
            Calculated stacking velocity.
        """
        from ..velocity_spectrum import VerticalVelocitySpectrum  # pylint: disable=import-outside-toplevel
        if not isinstance(spectrum, VerticalVelocitySpectrum):
            raise ValueError("spectrum must be an instance of VerticalVelocitySpectrum")

        if init is None and bounds is None:
            raise ValueError("Either init or bounds must be passed")
        if init is not None and not isinstance(init, StackingVelocity):
            raise ValueError("init must be an instance of StackingVelocity")
        if bounds is not None:
            bounds = to_list(bounds)
            if len(bounds) != 2 or not all(isinstance(bound, StackingVelocity) for bound in bounds):
                raise ValueError("bounds must be an array-like with two StackingVelocity instances")

        kwargs = {"init": init, "bounds": bounds, "relative_margin": relative_margin,
                  "acceleration_bounds": acceleration_bounds, "times_step": times_step, "max_offset": max_offset,
                  "hodograph_correction_step": hodograph_correction_step, "max_n_skips": max_n_skips}
        stacking_velocity_params = calculate_stacking_velocity(spectrum, **kwargs)
        times, velocities, bounds_times, min_velocity_bound, max_velocity_bound = stacking_velocity_params
        coords = spectrum.coords  # Evaluate only once
        stacking_velocity = cls(times, velocities, coords=coords)
        stacking_velocity.bounds = [cls(bounds_times, min_velocity_bound, coords=coords),
                                    cls(bounds_times, max_velocity_bound, coords=coords)]
        return stacking_velocity

    def __call__(self, times):
        """Return stacking velocities for given `times`.

        Parameters
        ----------
        times : 1d array-like
            An array with time values. Measured in milliseconds.

        Returns
        -------
        velocities : 1d np.ndarray
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        """
        return np.maximum(super().__call__(times), 0)
