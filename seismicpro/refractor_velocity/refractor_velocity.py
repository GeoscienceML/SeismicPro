"""Implements RefractorVelocity class for estimating the velocity model of an upper part of the seismic section"""

from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import SGDRegressor

from .utils import get_param_names, postprocess_params, dump_refractor_velocities, load_refractor_velocities
from ..muter import Muter
from ..decorators import batch_method, plotter
from ..utils import get_first_defined, set_ticks, set_text_formatting
from ..utils.interpolation import interp1d


# pylint: disable-next=too-many-instance-attributes
class RefractorVelocity:
    """A class to fit and store parameters of a velocity model of an upper part of the section.

    Near-surface velocity model is used to estimate a depth model of the very first layers which, combined with the
    velocity model, allows calculating static corrections for each trace of a survey.

    A velocity model of the first `N` refractors can be described in terms of the following parameters:
    * `t0` - intercept time, which theoretically equals to an uphole time in case when the first layer of the model
      describes a direct wave. Measured in milliseconds.
    * `x{i}` for i from 1 to `N`-1 - crossover offsets each defining an offset where a wave refracted from i-th layer
      arrives at the same time as a wave refracted from the next underlying layer. Measured in meters.
    * `v{i}` for i from 1 to `N` - velocity of the i-th layer. Measured in meters per second.

    A velocity model can either be created from already known parameters by directly passing them to `__init__` or via
    one of the following `classmethod`s:
    * `from_constant_velocity` - to create a single-layer `RefractorVelocity` with zero intercept time and given
      velocity of the refractor,
    * `from_first_breaks` - to automatically fit a near-surface velocity model by offsets and times of first breaks.
      This methods allows one to specify initial values of some parameters or bounds for their values or simply provide
      the expected number of refractors,
    * `from_file` - to create a velocity model from parameters stored in a file.

    The resulting object is callable and returns expected arrival times for given offsets. Each model parameter can be
    obtained by accessing the corresponding attribute of the created instance.

    Examples
    --------
    Define a near-surface velocity model from known parameters:
    >>> rv = RefractorVelocity(t0=100, x1=1500, v1=1600, v2=2200)

    Usually a velocity model is estimated by offsets of traces and times of first arrivals. First, let's load a survey
    with pre-calculated first breaks and randomly select a common source gather:
    >>> survey = Survey(survey_path, header_index="FieldRecord", header_cols=["offset", "TraceNumber"])
    >>> survey = survey.load_first_breaks(first_breaks_path)
    >>> gather = survey.sample_gather()

    Now an instance of `RefractorVelocity` can be created using a `from_first_breaks` method:
    >>> offsets = gather.offsets
    >>> fb_times = gather['FirstBreak']
    >>> rv = RefractorVelocity.from_first_breaks(offsets, fb_times, n_refractors=2)

    The same can be done by calling a `calculate_refractor_velocity` method of the gather:
    >>> rv = gather.calculate_refractor_velocity(n_refractors=2)

    Fit a two-layer refractor velocity model using initial values of its parameters:
    >>> initial_params = {'t0': 100, 'x1': 1500, 'v1': 2000, 'v2': 3000}
    >>> rv = RefractorVelocity.from_first_breaks(offsets, fb_times, init=initial_params)

    Fit a single-layer model with bounded parameters:
    >>> rv = RefractorVelocity.from_first_breaks(offsets, fb_times, bounds={'t0': [0, 200], 'v1': [1000, 3000]})

    Some keys in `init` or `bounds` may be omitted if they are defined in another `dict` or `n_refractors` is given:
    >>> rv = RefractorVelocity.from_first_breaks(offsets, fb_times, init={'x1': 200, 'v1': 1000},
    ...                                          bounds={'t0': [0, 50]}, n_refractors=3)

    Parameters
    ----------
    params : misc
        Parameters of the velocity model. Passed as keyword arguments.
    coords : Coordinates, optional
        Spatial coordinates at which refractor velocity is defined.
    is_uphole_corrected : bool, optional
        Whether the velocity model is uphole corrected.

    Attributes
    ----------
    n_refractors : int
        The number of refractors described by the model.
    params : dict
        Parameters of the velocity model.
    interpolator : callable
        An interpolator returning expected arrival times for given offsets.
    piecewise_offsets : 1d ndarray
        Offsets of knots of the offset-traveltime curve. Measured in meters.
    piecewise_times : 1d ndarray
        Times of knots of the offset-traveltime curve. Measured in milliseconds.
    coords : Coordinates or None
        Spatial coordinates at which refractor velocity is defined.
    is_uphole_corrected : bool or None
        Whether the velocity model is uphole corrected. `None` if unknown.
    is_fit : bool
        Whether the model parameters were estimated using `from_first_breaks` method.
    fit_result : OptimizeResult
        Optimization result returned by `scipy.optimize.minimize`. Defined only if the model was fit.
    max_offset : float or None
        Maximum offset reliably described by the model. Defined only if the model was fit.
    init : dict
        Initial values of model parameters used to fit the velocity model. Also includes estimated values for
        parameters that were not passed in `init` argument. Defined only if the model was fit.
    bounds : dict
        Lower and upper bounds of model parameters used to fit the velocity model. Also includes estimated values for
        parameters that were not passed in `bounds` argument. Defined only if the model was fit.
    offsets : 1d ndarray
        Offsets of traces used to fit the model. Measured in meters. Defined only if the model was fit.
    times : 1d ndarray
        Time of first break for each trace. Measured in milliseconds. Defined only if the model was fit.
    """
    def __init__(self, coords=None, is_uphole_corrected=None, **params):
        self._validate_params(params)
        self.n_refractors = len(params) // 2

        # Store params in the order defined by param_names
        self.params = {name: params[name] for name in self.param_names}
        self.piecewise_offsets, self.piecewise_times = self._calc_knots_by_params(np.array(list(self.params.values())))
        self.interpolator = interp1d(self.piecewise_offsets, self.piecewise_times)
        self.coords = coords
        self.is_uphole_corrected = is_uphole_corrected

        # Fit-related attributes, set only when from_first_breaks is called
        self.is_fit = False
        self.fit_result = None
        self.max_offset = None
        self.init = None
        self.bounds = None
        self.offsets = None
        self.times = None

    @classmethod  # pylint: disable-next=too-many-arguments, too-many-statements
    def from_first_breaks(cls, offsets, times, init=None, bounds=None, n_refractors=None, max_offset=None,
                          min_velocity_step=1, min_refractor_size=1, loss="L1", huber_coef=20, tol=1e-5, coords=None,
                          is_uphole_corrected=None, **kwargs):
        """Fit a near-surface velocity model by offsets of traces and times of their first breaks.

        This methods allows specifying:
        - initial values of model parameters via `init`,
        - bounds for parameter values via `bounds`,
        - or simply the expected number of refractors via `n_refractors`.

        At least one of `init`, `bounds` or `n_refractors` must be passed. Some keys may be omitted in one of `init` or
        `bounds` dicts if they are passed in another, e.g. one can pass only bounds for `v1` without an initial value,
        which will be inferred automatically. Both `init` and `bounds` dicts may not be passed at all if `n_refractors`
        is given.

        Parameters
        ----------
        offsets : 1d ndarray
            Offsets of traces. Measured in meters.
        times : 1d ndarray
            Time of first break for each trace. Measured in milliseconds.
        init : dict, optional
            Initial values of model parameters.
        bounds : dict, optional
            Lower and upper bounds of model parameters.
        n_refractors : int, optional
            The number of refractors described by the model.
        max_offset : float, optional
            Maximum offset reliably described by the model. Inferred automatically by `offsets`, `init` and `bounds`
            provided but should be preferably explicitly passed.
        min_velocity_step : int, or 1d array-like with shape (n_refractors - 1,), optional, defaults to 1
            Minimum difference between velocities of two adjacent refractors. Default value ensures that velocities are
            strictly increasing.
        min_refractor_size : int, or 1d array-like with shape (n_refractors,), optional, defaults to 1
            Minimum offset range covered by each refractor. Default value ensures that refractors do not degenerate
            into single points.
        loss : str, optional, defaults to "L1"
            Loss function to be minimized. Should be one of "MSE", "huber", "L1", "soft_L1", or "cauchy".
        huber_coef : float, optional, default to 20
            Coefficient for Huber loss function.
        tol : float, optional, defaults to 1e-5
            Precision goal for the value of loss in the stopping criterion.
        coords : Coordinates or None, optional
            Spatial coordinates of the created refractor velocity.
        is_uphole_corrected : bool, optional
            Whether `times` are uphole corrected.
        kwargs : misc, optional
            Additional `SLSQP` options, see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html for
            more details.

        Returns
        -------
        rv : RefractorVelocity
            Constructed near-surface velocity model.

        Raises
        ------
        ValueError
            If all `init`, `bounds`, and `n_refractors` are `None`.
            If `n_refractors` is given but is less than 1.
            If provided `init` and `bounds` are insufficient.
            If any `init` values are negative.
            If any `bounds` values are negative.
            If left bound is greater than the right bound for any of model parameters.
            If initial value of a parameter is out of defined bounds.
            If any of `min_velocity_step` or `min_refractor_size` constraints are violated by initial values of
            parameters.
        """
        offsets = np.array(offsets)
        times = np.array(times)
        if (offsets.ndim != 1) or (offsets.shape != times.shape):
            raise ValueError("offsets and times must be 1-dimensional and have the same length")
        if (~np.isfinite(offsets)).any() or (~np.isfinite(times)).any():
            raise ValueError("offsets and times must contain only finite values")
        if max_offset is not None:
            valid_mask = offsets <= max_offset
            offsets = offsets[valid_mask]
            times = times[valid_mask]

        # Convert values to int to avoid numerical instability in constraint checks which may occur for small floats
        min_velocity_step = np.ceil(min_velocity_step)
        min_refractor_size = np.ceil(min_refractor_size)

        if all(param is None for param in (init, bounds, n_refractors)):
            raise ValueError("At least one of `init`, `bounds` or `n_refractors` must be defined")
        init = {} if init is None else init
        bounds = {} if bounds is None else bounds

        # Merge initial values of parameters with those defined by bounds
        init_by_bounds = {key: (val1 + val2) / 2 for key, (val1, val2) in bounds.items()}
        init = {**init_by_bounds, **init}

        # Automatically estimate all params that were not passed in init or bounds by n_refractors
        if n_refractors is not None:
            init = cls.complete_init_by_refractors(init, n_refractors, offsets, times, max_offset,
                                                   min_velocity_step, min_refractor_size)

        # Validate initial values of model parameters and calculate the number of refractors
        cls._validate_params(init, max_offset, min_velocity_step, min_refractor_size)
        n_refractors = len(init) // 2
        param_names = get_param_names(n_refractors)
        min_velocity_step = np.broadcast_to(min_velocity_step, n_refractors-1)
        min_refractor_size = np.broadcast_to(min_refractor_size, n_refractors)

        # Estimate max_offset if it was not given
        if max_offset is None:
            max_init = init.get(f"x{n_refractors - 1}", 0) + min_refractor_size[-1]
            max_bound = max((max(val) for key, val in bounds.items() if key.startswith("x")), default=0)
            max_offset = max(offsets.max(), max_init, max_bound)

        # Estimate maximum possible velocity: it should not be highly accurate, but should cover all initial velocities
        # and their bounds. Used only to early-stop a diverging optimization on poor data when optimal velocity
        # approaches infinity.
        last_refractor_velocity = init[f"v{n_refractors}"]
        velocity_bounds = [bounds.get(f"v{i}", [0, 0]) for i in range(n_refractors)]
        max_velocity_bounds_range = max(right - left for left, right in velocity_bounds)
        max_velocity = last_refractor_velocity + max(max_velocity_bounds_range, last_refractor_velocity)

        # Set default bounds for parameters that don't have them specified, validate the result for correctness
        default_t0_bounds = [[0, max(init["t0"], times.max())]]
        default_crossover_bounds = [[min_refractor_size[0], max_offset - min_refractor_size[-1]]
                                    for _ in range(n_refractors - 1)]
        default_velocity_bounds = [[0, max_velocity] for _ in range(n_refractors)]
        default_params_bounds = default_t0_bounds + default_crossover_bounds + default_velocity_bounds
        bounds = {**dict(zip(param_names, default_params_bounds)), **bounds}
        cls._validate_params_bounds(init, bounds)

        # Store init and bounds in the order defined by param_names
        init = {name: init[name] for name in param_names}
        bounds = {name: bounds[name] for name in param_names}

        # Calculate arrays of initial params and their bounds to be passed to minimize
        init_array = cls._scale_params(np.array(list(init.values()), dtype=np.float32))
        bounds_array = cls._scale_params(np.array(list(bounds.values()), dtype=np.float32))

        # Define model constraints
        constraints = []
        if n_refractors > 1:
            velocities_ascend = {
                "type": "ineq",
                "fun": lambda x: np.diff(cls._unscale_params(x)[n_refractors:]) - min_velocity_step
            }
            constraints.append(velocities_ascend)
        if n_refractors > 2:
            crossover_offsets_ascend = {
                "type": "ineq",
                "fun": lambda x: np.diff(cls._unscale_params(x)[1:n_refractors]) - min_refractor_size[1:-1]
            }
            constraints.append(crossover_offsets_ascend)

        # Fit a piecewise-linear velocity model
        loss_fn = partial(cls.calculate_loss, offsets=offsets, times=times, max_offset=max_offset,
                          loss=loss, huber_coef=huber_coef)
        fit_result = minimize(loss_fn, x0=init_array, bounds=bounds_array, constraints=constraints,
                              method="SLSQP", tol=tol, options=kwargs)
        param_values = cls._unscale_params(fit_result.x)
        param_values[1:n_refractors] = np.minimum(param_values[1:n_refractors], max_offset)
        param_values = postprocess_params(param_values)
        params = dict(zip(param_names, param_values))

        # Construct a refractor velocity instance
        self = cls(coords=coords, is_uphole_corrected=is_uphole_corrected, **params)
        self.is_fit = True
        self.fit_result = fit_result
        self.max_offset = max_offset
        self.init = init
        self.bounds = bounds
        self.offsets = offsets
        self.times = times
        return self

    @classmethod
    def from_file(cls, path, encoding="UTF-8"):
        """Load a near-surface velocity model from a file.

        Notes
        -----
        See more about the file format in :func:`~.utils.load_refractor_velocities`.

        Parameters
        ----------
        path : str
            Path to a file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.

        Returns
        -------
        self : RefractorVelocity
            Loaded velocity model.

        Raises
        ------
        ValueError
            If the file contains more than one set of parameters.
        """
        rv_list = load_refractor_velocities(path, encoding)
        if len(rv_list) != 1:
            raise ValueError("The file should contain only one set of RefractorVelocity parameters.")
        return rv_list[0]

    @classmethod
    def from_constant_velocity(cls, velocity, coords=None, is_uphole_corrected=None):
        """Define a 1-layer near-surface velocity model with given layer velocity and zero intercept time.

        Parameters
        ----------
        velocity : float
            Velocity of the first layer.
        coords : Coordinates, optional
            Spatial coordinates of the created object.
        is_uphole_corrected : bool, optional
            Whether the velocity model is uphole corrected.

        Returns
        -------
        rv : RefractorVelocity
            1-layer near-surface velocity model.

        Raises
        ------
        ValueError
            If passed `velocity` is negative.
        """
        return cls(t0=0, v1=velocity, coords=coords, is_uphole_corrected=is_uphole_corrected)

    @property
    def param_names(self):
        """list of str: Names of model parameters."""
        return get_param_names(self.n_refractors)

    @property
    def has_coords(self):
        """bool: Whether refractor velocity coordinates are not-None."""
        return self.coords is not None

    def __repr__(self):
        """String representation of the velocity model."""
        params_str = ", ".join([f"{param}={val:.0f}" for param, val in self.params.items()])
        args_str = f"{params_str}, coords={repr(self.coords)}, is_uphole_corrected={self.is_uphole_corrected}"
        return f"RefractorVelocity({args_str})"

    def __getattr__(self, key):
        """Get requested parameter of the velocity model by its name."""
        if key.startswith("__") or key not in self.params:  # Guarantee proper pickling/unpickling
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        return self.params[key]

    def __call__(self, offsets):
        """Return the expected times of first breaks for the given offsets."""
        return self.interpolator(offsets)

    # Methods to validate model parameters and their bounds for correctness

    @staticmethod
    def _validate_params_names(params):
        """Check if keys of `params` dict describe a valid velocity model. This method checks only names of parameters,
        not their values."""
        err_msg = ("The model is underdetermined. Pass t0 and v1 to define a one-layer model. "
                   "Pass t0, x1, ..., x{N-1}, v1, ..., v{N} to define an N-layer model for N >= 2.")
        n_refractors = len(params) // 2
        if n_refractors < 1:
            raise ValueError(err_msg)
        wrong_keys = set(get_param_names(n_refractors)) ^ params.keys()
        if wrong_keys:
            raise ValueError(err_msg)

    @classmethod
    def _validate_params(cls, params, max_offset=None, min_velocity_step=0, min_refractor_size=0):
        """Check if `params` dict describes a valid velocity model."""
        cls._validate_params_names(params)
        n_refractors = len(params) // 2
        param_values = np.array([params[name] for name in get_param_names(n_refractors)])
        if max_offset is None:
            max_offset = np.inf

        negative_params = {key: val for key, val in params.items() if val < 0}
        if negative_params:
            raise ValueError(f"The following parameters contain negative values: {negative_params}")

        refractor_sizes = np.diff(param_values[1:n_refractors], prepend=0, append=max_offset)
        valid_sizes = (refractor_sizes >= min_refractor_size) | np.isclose(refractor_sizes, min_refractor_size)
        if not valid_sizes.all():
            raise ValueError(f"Offset range covered by refractors must be no less than {min_refractor_size} meters")

        velocity_steps = np.diff(param_values[n_refractors:])
        valid_steps = (velocity_steps >= min_velocity_step) | np.isclose(velocity_steps, min_velocity_step)
        if not valid_steps.all():
            raise ValueError(f"Refractor velocities must increase by no less than {min_velocity_step} m/s")

    @classmethod
    def _validate_params_bounds(cls, params, bounds):
        """Check if provided `bounds` are consistent with model parameters from `params`."""
        cls._validate_params_names(bounds)
        if params.keys() != bounds.keys():
            raise ValueError("params and bounds must contain the same keys")

        negative_bounds = {key: val for key, val in bounds.items() if min(val) < 0}
        if negative_bounds:
            raise ValueError(f"The following parameters contain negative bounds: {negative_bounds}")

        reversed_bounds = {key: [left, right] for key, [left, right] in bounds.items() if left > right}
        if reversed_bounds:
            raise ValueError(f"The following parameters contain reversed bounds: {reversed_bounds}")

        out_of_bounds = {key for key, [left, right] in bounds.items() if params[key] < left or params[key] > right}
        if out_of_bounds:
            raise ValueError(f"Values of the following parameters are out of their bounds: {out_of_bounds}")

    # Methods to roughly estimate refractor velocities

    @staticmethod
    def estimate_refractor_velocity(offsets, times, refractor_bounds):
        """Perform a rough estimation of a refractor velocity and intercept time by fitting a linear regression to an
        offset-time point cloud within given offsets bounds."""
        # Avoid fitting a regression if an empty refractor is processed
        refractor_mask = (offsets > refractor_bounds[0]) & (offsets <= refractor_bounds[1])
        n_refractor_points = refractor_mask.sum()
        if n_refractor_points == 0:
            return np.nan, np.nan, n_refractor_points

        # Avoid fitting a regression if all points have constant offsets or times of first breaks
        refractor_offsets = offsets[refractor_mask]
        refractor_times = times[refractor_mask]
        mean_offset, std_offset = np.mean(refractor_offsets), np.std(refractor_offsets)
        mean_time, std_time = np.mean(refractor_times), np.std(refractor_times)
        if np.isclose([std_offset, std_time], 0).any():
            return np.nan, np.nan, n_refractor_points

        # Fit the model to obtain velocity in km/s and intercept time in ms
        scaled_offsets = (refractor_offsets - mean_offset) / std_offset
        scaled_times = (refractor_times - mean_time) / std_time
        reg = SGDRegressor(loss="huber", epsilon=0.1, penalty=None, learning_rate="optimal", alpha=0.01,
                           max_iter=1000, tol=1e-5, shuffle=True)
        reg.fit(scaled_offsets.reshape(-1, 1), scaled_times, coef_init=1, intercept_init=0)
        slope = reg.coef_[0] * std_time / std_offset
        t0 = mean_time + reg.intercept_[0] * std_time - slope * mean_offset

        # Postprocess the obtained params
        velocity = 1000 / max(0.1, slope)  # Convert slope to velocity in m/s, clip it to be in a [0, 10000] interval
        t0 = min(max(0, t0), times.max())  # Clip intercept time to lie within a [0, times.max()] interval
        return velocity, t0, n_refractor_points

    @staticmethod
    def enforce_step_constraints(values, defined_indices, min_step=0):
        """Modify values whose indices are not in `defined_indices` so that a difference between each two adjacent
        values is no less than the corresponding `min_step`. Fill all `nan` values so that this constraint is
        satisfied."""
        defined_indices = np.sort(np.atleast_1d(defined_indices))
        min_step = np.broadcast_to(min_step, len(values) - 1)

        # Refine values between each two adjacent fixed values
        for start, stop in zip(defined_indices[:-1], defined_indices[1:]):
            for pos in range(start + 1, stop):
                values[pos] = np.nanmax([values[pos], values[pos - 1] + min_step[pos - 1]])
            for pos in range(stop - 1, start, -1):
                values[pos] = np.nanmin([values[pos], values[pos + 1] - min_step[pos]])

        # Refine values with indices outside the defined_indices range
        for pos in range(defined_indices[-1] + 1, len(values)):
            values[pos] = np.nanmax([values[pos], values[pos - 1] + min_step[pos - 1]])
        for pos in range(defined_indices[0] - 1, -1, -1):
            values[pos] = np.nanmin([values[pos], values[pos + 1] - min_step[pos]])

        return values

    @classmethod
    def complete_init_by_refractors(cls, init, n_refractors, offsets, times, max_offset=None,
                                    min_velocity_step=1, min_refractor_size=1):
        """Determine all the values in `init` that are insufficient to define a valid velocity model by the expected
        number of refractors."""
        param_names = get_param_names(n_refractors)
        if init.keys() - set(param_names):
            raise ValueError("The model is overdetermined: init or bounds contain parameters inconsistent with "
                             "n_refractors passed. Maximum valid set of parameters contains only t0 and v1 keys for a "
                             "single refractor and t0, x1, ..., x{N-1}, v1, ..., v{N} keys for N >= 2 refractors.")

        min_velocity_step = np.broadcast_to(min_velocity_step, n_refractors-1)
        min_refractor_size = np.broadcast_to(min_refractor_size, n_refractors)

        cross_offsets = [0] + [init.get(f"x{i}", np.nan) for i in range(1, n_refractors)]
        if max_offset is None:
            max_defined_ix = np.nanargmax(cross_offsets)
            max_offset = max(offsets.max(), cross_offsets[max_defined_ix] + min_refractor_size[max_defined_ix:].sum())
        cross_offsets = np.array(cross_offsets + [max_offset])

        # Linearly interpolate unknown crossover offsets but enforce min_refractor_size constraint
        defined_indices = np.where(~np.isnan(cross_offsets))[0]
        cross_indices = np.arange(n_refractors + 1)
        cross_offsets = np.interp(cross_indices, cross_indices[defined_indices], cross_offsets[defined_indices])
        cross_offsets = cls.enforce_step_constraints(cross_offsets, defined_indices, min_refractor_size)

        # Fit linear regressions to estimate unknown refractor velocities
        velocities = np.array([init.get(f"v{i}", np.nan) for i in range(1, n_refractors + 1)])
        undefined_mask = np.isnan(velocities)
        estimates = [cls.estimate_refractor_velocity(offsets, times, cross_offsets[i:i+2])
                     for i in np.where(undefined_mask)[0]]
        velocities[undefined_mask] = [vel for (vel, _, _) in estimates]

        if np.isnan(velocities).all():
            # Use a dummy velocity range as an initial guess if no velocities were passed in init/bounds dicts and
            # non of them were successfully fit using estimate_refractor_velocity
            velocities = np.cumsum(np.r_[1600, min_velocity_step])
        else:
            defined_indices = np.where(~undefined_mask)[0]
            if undefined_mask.all():
                # If no velocities were passed in init, start the refinement from the refractor with maximum number of
                # points among those with properly estimated velocity. At least one of them is guaranteed to exist.
                defined_index = max(enumerate(estimates), key=lambda x: 0 if np.isnan(x[1][0]) else x[1][-1])[0]
                velocities[defined_index] = max(velocities[defined_index], min_velocity_step[:defined_index].sum())
                defined_indices = [defined_index]
            velocities = cls.enforce_step_constraints(velocities, defined_indices, min_velocity_step)

        # Estimate t0 if not given in init
        t0 = init.get("t0")
        if t0 is None:
            if undefined_mask[0]:  # regression is already fit
                t0 = estimates[0][1]
            else:
                _, t0, _ = cls.estimate_refractor_velocity(offsets, times, cross_offsets[:2])
            t0 = np.nan_to_num(t0)  # can be nan if the regression hasn't fit successfully

        return dict(zip(param_names, [t0, *cross_offsets[1:-1], *velocities]))

    # Methods to fit a piecewise-linear regression

    @staticmethod
    def _scale_params(unscaled_params):
        """Scale a vector of model parameters before passing to `scipy.optimize.minimize`."""
        scaled = np.empty_like(unscaled_params)
        scaled[0] = unscaled_params[0] / 100
        scaled[1:] = unscaled_params[1:] / 1000
        return scaled

    @staticmethod
    def _unscale_params(scaled_params):
        """Unscale results of `scipy.optimize.minimize` to the original units."""
        unscaled_params = np.empty_like(scaled_params)
        unscaled_params[0] = scaled_params[0] * 100
        unscaled_params[1:] = scaled_params[1:] * 1000
        return unscaled_params

    @staticmethod
    def _calc_knots_by_params(unscaled_params, max_offset=None):
        """Calculate coordinates of the knots of a piecewise linear function by a vector of unscaled velocity model
        parameters and `max_offset`."""
        n_refractors = len(unscaled_params) // 2
        params_max_offset = unscaled_params[n_refractors - 1] if n_refractors > 1 else 0
        if max_offset is None or max_offset < params_max_offset:
            # Artificially set max_offset in order to properly define an interpolator
            max_offset = params_max_offset + 1000

        piecewise_offsets = np.concatenate([[0], unscaled_params[1:n_refractors], [max_offset]])
        piecewise_times = np.empty(n_refractors + 1)
        piecewise_times[0] = unscaled_params[0]
        params_zip = zip(piecewise_offsets[1:], piecewise_offsets[:-1], unscaled_params[n_refractors:])
        for i, (cross, prev_cross, vel) in enumerate(params_zip):
            piecewise_times[i + 1] = piecewise_times[i] + 1000 * (cross - prev_cross) / max(0.01, vel)  # m/s to km/s
        return piecewise_offsets, piecewise_times

    @classmethod
    def calculate_loss(cls, scaled_params, offsets, times, max_offset, loss='L1', huber_coef=20):
        """Calculate the value of loss function for a given vector of model parameters scaled according to
        `cls._scale_params`.

        `scaled_params` should be a 1d `np.ndarray` with shape (2 * n_refractors,) with the following structure:
        - scaled_params[0] : intercept time,
        - scaled_params[1:n_refractors] : crossover offsets,
        - scaled_params[n_refractors:] : refractor velocities.

        Available loss functions are "MSE", "huber", "L1", "soft_L1", or "cauchy", coefficient for Huber loss is
        defined by `huber_coef` argument. All losses apply mean reduction of point-wise losses.
        """
        piecewise_offsets, piecewise_times = cls._calc_knots_by_params(cls._unscale_params(scaled_params), max_offset)
        abs_diff = np.abs(np.interp(offsets, piecewise_offsets, piecewise_times) - times)

        if loss == 'MSE':
            return (abs_diff ** 2).mean()
        if loss == 'huber':
            loss_val = np.empty_like(abs_diff)
            mask = abs_diff <= huber_coef
            loss_val[mask] = 0.5 * (abs_diff[mask] ** 2)
            loss_val[~mask] = huber_coef * abs_diff[~mask] - 0.5 * (huber_coef ** 2)
            return loss_val.mean()
        if loss == 'L1':
            return abs_diff.mean()
        if loss == 'soft_L1':
            return 2 * ((1 + abs_diff) ** 0.5 - 1).mean()
        if loss == 'cauchy':
            return np.log(abs_diff + 1).mean()
        raise ValueError("Unknown loss function")

    # General processing methods

    @batch_method(target="for", copy_src=False)
    def create_muter(self, delay=0, velocity_reduction=0):
        """Create a muter to attenuate high amplitudes immediately following the first breaks in the following way:
        1) Reduce velocity of each refractor by `velocity_reduction` to account for imperfect model fit and variability
           of first arrivals at a given offset,
        2) Take an offset-time curve defined by an adjusted model from the previous step and shift it by `delay` to
           handle near-offset traces.

        Parameters
        ----------
        delay : float, optional, defaults to 0
            Introduced constant delay. Measured in milliseconds.
        velocity_reduction : float or array-like of float, optional, defaults to 0
            A value used to decrement velocity of each refractor. If a single `float`, the same value is used for all
            refractors. Measured in meters/seconds.

        Returns
        -------
        muter : Muter
            Created muter.
        """
        return Muter.from_refractor_velocity(self, delay=delay, velocity_reduction=velocity_reduction)

    def dump(self, path, encoding="UTF-8"):
        """Dump a near-surface velocity model to a file.

        Notes
        -----
        See more about the file format in :func:`~utils.load_refractor_velocities`.
        `RefractorVelocity` instance should have well-defined `coords`.

        Parameters
        ----------
        path : str
            Path to the created file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.

        Raises
        ------
        ValueError
            If `coords` are undefined.
        """
        if not self.has_coords:
            raise ValueError("RefractorVelocity must have well-defined coordinates.")
        dump_refractor_velocities(self, path=path, encoding=encoding)

    @plotter(figsize=(10, 5), args_to_unpack="compare_to")
    def plot(self, *, ax=None, max_offset=None, title=None, x_ticker=None, y_ticker=None, show_params=True,
             threshold_times=None, compare_to=None, text_kwargs=None, **kwargs):
        """Plot an offset-traveltime curve and data used to fit the model if it was constructed from offsets and times
        of first breaks.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes of a figure to plot on. Will be created automatically if not given.
        max_offset : float, optional
            Maximum offset displayed on the plot.
        title : str, optional
            Plot title.
        x_ticker : dict, optional
            Parameters for ticks and ticklabels formatting for the x-axis; see :func:`~utils.set_ticks`
            for more details.
        y_ticker : dict, optional
            Parameters for ticks and ticklabels formatting for the y-axis; see :func:`~utils.set_ticks`
            for more details.
        show_params : bool, optional, defaults to True
            If `True` shows the velocity model parameters on the plot.
        threshold_times : float, optional
            Size of the neighborhood around the offset-time curve that will be highlighted on the plot. Won't be shown
            if the parameter is not given.
        compare_to : RefractorVelocity, dict or str, optional
            Additional velocity model to be displayed on the same axes. `RefractorVelocity` instance is used directly.
            `dict` value represents parameters of the velocity model.
            May be `str` if plotted in a pipeline: in this case it defines a component with refractor velocities to
            plot.
        text_kwargs : dict, optional
            Additional arguments to the :func:`~matplotlib.pyplot.text` to format model parameters on the plot if they
            are displayed.
        kwargs : dict, optional
            Additional keyword arguments to :func:`~utils.set_text_formatting`. Used to the modify text, title and
            ticker formatting.

        Returns
        -------
        self : RefractorVelocity
            Velocity model unchanged.
        """
        (title, x_ticker, y_ticker, text_kwargs), kwargs = set_text_formatting(title, x_ticker, y_ticker, text_kwargs,
                                                                               **kwargs)
        if kwargs:
            raise ValueError(f"kwargs contains unknown keys {kwargs.keys()}")
        set_ticks(ax, "x", "offset, m", **x_ticker)
        set_ticks(ax, "y", "Time, ms", **y_ticker)

        max_offset = get_first_defined(max_offset, self.max_offset, self.piecewise_offsets[-1])
        ax.scatter(self.offsets, self.times, s=1, color="black", label="first breaks")
        self._plot_lines(ax, max_offset=max_offset, curve_label="offset-traveltime curve", curve_color="red",
                         crossoffset_label="crossover point", crossover_color="blue", threshold_times=threshold_times)

        if show_params:
            params = [self.params[name] for name in self.param_names]
            text_info = f"t0: {params[0]:.2f} ms"
            if self.n_refractors > 1:
                text_info += f"\ncrossover offsets: {', '.join(str(round(x)) for x in params[1:self.n_refractors])} m"
            text_info += f"\nvelocities: {', '.join(f'{v:.0f}' for v in params[self.n_refractors:])} m/s"
            text_kwargs = {"fontsize": 12, "va": "top", **text_kwargs}
            text_ident = text_kwargs.pop("x", .03), text_kwargs.pop("y", .94)
            ax.text(*text_ident, text_info, transform=ax.transAxes, **text_kwargs)

        if compare_to is not None:
            if isinstance(compare_to, dict):
                compare_to = RefractorVelocity(**compare_to)
            if not isinstance(compare_to, RefractorVelocity):
                raise ValueError("compare_to must be either a dict or a RefractorVelocity instance")
            # pylint: disable-next=protected-access
            compare_to._plot_lines(ax, max_offset=max_offset, curve_label="compared offset-traveltime curve",
                                   curve_color="#ff7900", crossoffset_label="compared crossover point",
                                   crossover_color="green")

        ax.set_xlim(0, max_offset)
        ax.set_ylim(0)
        ax.legend(loc="lower right")
        ax.set_title(**{"label": None, **title})
        return self

    def _plot_lines(self, ax, max_offset, curve_label, curve_color, crossoffset_label, crossover_color,
                    threshold_times=None):
        """Plot an offset-traveltime curve and a vertical line for each crossover offset."""
        crossover_offsets = self.piecewise_offsets[1:-1]
        covered_offsets = crossover_offsets[:np.searchsorted(crossover_offsets, max_offset)]
        offsets = np.concatenate([[0], covered_offsets, [max_offset]])
        times = self(offsets)

        ax.plot(offsets, times, "-", color=curve_color, label=curve_label)
        if threshold_times is not None:
            ax.fill_between(offsets, times - threshold_times, times + threshold_times, color=curve_color, alpha=0.2,
                            label=f"+/- {threshold_times} ms threshold area")

        if len(covered_offsets) > 1:
            crossoffset_label += 's'
        for i, offset in enumerate(covered_offsets):
            label = None if i else crossoffset_label
            ax.axvline(offset, ls="--", color=crossover_color, label=label)
