"""Implements a RefractorVelocityField class which stores near-surface velocity models calculated at different field
location and allows for their spatial interpolation"""

import os
from textwrap import dedent
from functools import partial, cached_property
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .refractor_velocity import RefractorVelocity
from .metrics import REFRACTOR_VELOCITY_QC_METRICS, RefractorVelocityMetric
from .interactive_plot import FieldPlot
from .utils import get_param_names, postprocess_params, dump_refractor_velocities, load_refractor_velocities
from ..field import SpatialField
from ..metrics import initialize_metrics
from ..utils import to_list, get_coords_cols, Coordinates, IDWInterpolator, ForPoolExecutor, GEOGRAPHIC_COORDS
from ..const import HDR_FIRST_BREAK


class RefractorVelocityField(SpatialField):
    """A class for storing near-surface velocity models calculated at different field location and interpolating them
    spatially over the whole field.

    Refractor velocities used to compute a depth model of the very first layers are usually estimated on a sparse grid
    of inlines and crosslines and then interpolated over the whole field in order to reduce computational costs. Such
    interpolation can be performed by `RefractorVelocityField` which provides an interface to obtain a velocity model
    of an upper part of the section at given spatial coordinates via its `__call__` and `interpolate` methods.

    A field can be populated with velocity models in 4 main ways:
    - by passing precalculated velocities in the `__init__`,
    - by creating an empty field and then iteratively updating it with estimated velocities using `update`,
    - by loading a field from a file with parameters of velocity models using `from_file`,
    - by calculating a field directly from a survey with loaded first breaks using `from_survey`.

    After all velocities are added, field interpolator should be created to make the field callable. It can be done
    either manually by executing `create_interpolator` method or automatically during the first call to the field if
    `auto_create_interpolator` flag was set to `True` upon field instantiation. Manual interpolator creation is useful
    when one wants to fine-tune its parameters or the field should be later passed to different processes (e.g. in a
    pipeline with prefetch with `mpc` target) since otherwise the interpolator will be independently created in all the
    processes.

    Examples
    --------
    A field can be created empty and updated with instances of `RefractorVelocity` class:
    >>> field = RefractorVelocityField()
    >>> rv = RefractorVelocity(t0=100, x1=1500, v1=1600, v2=2200,
                               coords=Coordinates((150, 80), names=("INLINE_3D", "CROSSLINE_3D")))
    >>> field.update(rv)

    Or created from precalculated instances:
    >>> field = RefractorVelocityField(list_of_rv)

    Or created directly from a survey with preloaded first breaks:
    >>> field = RefractorVelocityField.from_survey(survey, n_refractors=2)

    Or simply loaded from a file with parameters of near-surface velocity models:
    >>> field = RefractorVelocityField.from_file(path_to_file)

    Note that all velocity models in the field must describe the same number of refractors.

    Velocity models of an upper part of the section are usually estimated independently of one another and thus may
    appear inconsistent. `refine` method allows utilizing local information about near-surface conditions to refit
    the field:
    >>> field = field.refine()

    Only fields that were constructed directly from offset-traveltime data can be refined.

    Field interpolator will be created automatically upon the first call by default, but one may do it explicitly by
    executing `create_interpolator` method:
    >>> field.create_interpolator("rbf")

    Now the field allows for velocity interpolation at given coordinates:
    >>> rv = field((100, 100))

    Or can be passed directly to some gather processing methods:
    >>> gather = survey.sample_gather().apply_lmo(field)

    Parameters
    ----------
    items : RefractorVelocity or list of RefractorVelocity, optional
        Velocity models to be added to the field on instantiation. If not given, an empty field is created.
    n_refractors : int, optional
        The number of refractors described by the field. Inferred automatically on the first update if not given.
    survey : Survey, optional
        A survey described by the field.
    is_geographic : bool, optional
        Coordinate system of the field: either geographic (e.g. (CDP_X, CDP_Y)) or line-based (e.g. (INLINE_3D,
        CROSSLINE_3D)). Inferred automatically on the first update if not given.
    auto_create_interpolator : bool, optional, defaults to True
        Whether to automatically create default interpolator (RBF for more than 3 items in the field or IDW otherwise)
        upon the first call to the field.

    Attributes
    ----------
    survey : Survey or None
        A survey described by the field. `None` if not specified during instantiation.
    n_refractors : int or None
        The number of refractors described by the field. `None` for an empty field if was not specified during
        instantiation.
    item_container : dict
        A mapping from coordinates of field items as 2-element tuples to the items themselves.
    is_geographic : bool
        Whether coordinate system of the field is geographic. `None` for an empty field if was not specified during
        instantiation.
    coords_cols : tuple with 2 elements or None
        Names of SEG-Y trace headers representing coordinates of items in the field if names are the same among all the
        items and match the geographic system of the field. ("X", "Y") for a field in geographic coordinate system if
        names of coordinates of its items are either mixed or line-based. ("INLINE_3D", "CROSSLINE_3D") for a field in
        line-based coordinate system if names of coordinates of its items are either mixed or geographic. `None` for an
        empty field.
    interpolator : SpatialInterpolator or None
        Field data interpolator.
    is_dirty_interpolator : bool
        Whether the field was updated after the interpolator was created.
    auto_create_interpolator : bool
        Whether to automatically create default interpolator (RBF for more than 3 items in the field or IDW otherwise)
        upon the first call to the field.
    """
    item_class = RefractorVelocity

    def __init__(self, items=None, n_refractors=None, survey=None, is_geographic=None, auto_create_interpolator=True):
        self.n_refractors = n_refractors
        super().__init__(items, survey, is_geographic, auto_create_interpolator)

    @property
    def param_names(self):
        """list of str: Names of model parameters."""
        if self.n_refractors is None:
            raise ValueError("The number of refractors is undefined")
        return get_param_names(self.n_refractors)

    @cached_property
    def is_uphole_corrected(self):
        """bool or None: Whether the field is uphole corrected. `None` if unknown or mixed items are stored."""
        is_uphole_corrected_set = {item.is_uphole_corrected for item in self.items}
        if len(is_uphole_corrected_set) != 1:
            return None
        return is_uphole_corrected_set.pop()

    @cached_property
    def is_fit(self):
        """bool: Whether the field was constructed directly from offset-traveltime data."""
        return all(item.is_fit for item in self.items)

    @cached_property
    def mean_velocity(self):
        """RefractorVelocity: Mean near-surface velocity model over the field."""
        return self.construct_item(self.values.mean(axis=0), coords=None)

    def __str__(self):
        """Print field metadata including descriptive statistics of the near-surface velocity model, coordinate system
        and created interpolator."""
        msg = super().__str__() + dedent(f"""\n
        Number of refractors:      {self.n_refractors}
        Is fit from first breaks:  {self.is_fit}
        Is uphole corrected:       {"Unknown" if self.is_uphole_corrected is None else self.is_uphole_corrected}
        """)

        if not self.is_empty:
            params_df = pd.DataFrame(self.values, columns=self.param_names)
            params_stats_str = params_df.describe().iloc[1:].T.to_string(col_space=8, float_format="{:.02f}".format)
            msg += f"""\nDescriptive statistics of the near-surface velocity model:\n{params_stats_str}"""

        return msg

    @staticmethod
    def _fit_refractor_velocities(rv_kwargs_list, common_kwargs):
        """Fit a separate near-surface velocity model by offsets and times of first breaks for each set of parameters
        defined in `rv_kwargs_list`. This is a helper function and is defined as a `staticmethod` only to be picklable
        so that it can be passed to `ProcessPoolExecutor.submit`."""
        return [RefractorVelocity.from_first_breaks(**rv_kwargs, **common_kwargs) for rv_kwargs in rv_kwargs_list]

    @classmethod
    def _fit_refractor_velocities_parallel(cls, rv_kwargs_list, common_kwargs=None, chunk_size=250, n_workers=None,
                                           bar=True, desc=None):
        """Fit a separate near-surface velocity model by offsets and times of first breaks for each set of parameters
        defined in `rv_kwargs_list`. Velocity model fitting is performed in parallel processes in chunks of size no
        more than `chunk_size`."""
        if common_kwargs is None:
            common_kwargs = {}
        n_velocities = len(rv_kwargs_list)
        n_chunks, mod = divmod(n_velocities, chunk_size)
        if mod:
            n_chunks += 1
        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)
        executor_class = ForPoolExecutor if n_workers == 1 else ProcessPoolExecutor

        futures = []
        with tqdm(total=n_velocities, desc=desc, disable=not bar) as pbar:
            with executor_class(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    chunk_kwargs = rv_kwargs_list[i * chunk_size : (i + 1) * chunk_size]
                    future = pool.submit(cls._fit_refractor_velocities, chunk_kwargs, common_kwargs)
                    future.add_done_callback(lambda fut: pbar.update(len(fut.result())))
                    futures.append(future)
        return sum([future.result() for future in futures], [])

    @classmethod  # pylint: disable-next=too-many-arguments
    def from_survey(cls, survey, is_geographic=None, auto_create_interpolator=True, init=None, bounds=None,
                    n_refractors=None, min_velocity_step=1, min_refractor_size=1, loss='L1', huber_coef=20, tol=1e-5,
                    first_breaks_header=HDR_FIRST_BREAK, correct_uphole=None, chunk_size=250, n_workers=None, bar=True,
                    **kwargs):
        """Create a field by estimating a near-surface velocity model for each gather in the survey.

        The survey should contain headers with trace offsets, times of first breaks and coordinates of its gathers.
        Please refer to :class:~`.refractor_velocity.RefractorVelocity` docs for more information about velocity model
        calculation.

        Parameters
        ----------
        survey : Survey
            Survey with preloaded offsets, times of first breaks, and gather coordinates.
        is_geographic : bool, optional
            Coordinate system of the field: either geographic (e.g. (CDP_X, CDP_Y)) or line-based (e.g. (INLINE_3D,
            CROSSLINE_3D)). Inferred automatically by the type of survey gathers if not given.
        auto_create_interpolator : bool, optional, defaults to True
            Whether to automatically create default interpolator (RBF for more than 3 items in the field or IDW
            otherwise) upon the first call to the field.
        init : dict, optional
            Initial parameters for all velocity models in the field.
        bounds : dict, optional
            Lower and upper bounds of parameters for all velocity models in the field.
        n_refractors : int, optional
            The number of refractors to be described by the field.
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
        first_breaks_header : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
            emulating the case when sources are located on the surface. If not given, correction is performed if
            "SourceUpholeTime" header is loaded.
        chunk_size : int, optional, defaults to 250
            The number of velocity models estimated by each of spawned processes.
        n_workers : int, optional
            The maximum number of simultaneously spawned processes to estimate velocity models. Defaults to the number
            of cpu cores.
        bar : bool, optional, defaults to True
            Whether to show progress bar for field calculation.
        kwargs : misc, optional
            Additional `SLSQP` options, see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html for
            more details.

        Raises
        ------
        ValueError
            If the survey is empty.
            If the survey is indexed by individual traces, not gathers.
            If any gather has non-unique pair of coordinates.
        """
        if survey.is_empty:
            raise ValueError("Survey is empty")
        if survey.headers.index.is_unique:
            raise ValueError("Survey must be indexed by gathers, not individual traces")

        # Extract required headers from the survey
        coords_cols = get_coords_cols(survey.indexed_by, survey.source_id_cols, survey.receiver_id_cols)
        survey_coords = survey[coords_cols]
        survey_offsets = survey["offset"]
        survey_times = survey[first_breaks_header]
        if correct_uphole is None:
            correct_uphole = "SourceUpholeTime" in survey.available_headers
        if correct_uphole:
            survey_times = survey_times + survey["SourceUpholeTime"]

        # Check if coordinates are unique within each gather
        gather_start_ix = np.where(~survey.headers.index.duplicated(keep="first"))[0]
        gather_change_ix = gather_start_ix[1:]
        coords_change_ix = np.where(~np.isclose(np.diff(survey_coords, axis=0), 0).all(axis=1))[0] + 1
        if not np.isin(coords_change_ix, gather_change_ix).all():
            raise ValueError("Non-unique coordinates are found for some gathers in the survey")

        # Construct a dict of fit parameters for each gather in the survey
        gather_offsets = np.split(survey_offsets, gather_change_ix)
        gather_times = np.split(survey_times, gather_change_ix)
        gather_coords = [Coordinates(coords, names=coords_cols) for coords in survey_coords[gather_start_ix]]
        rv_kwargs_list = [{"offsets": offsets, "times": times, "coords": coords}
                          for offsets, times, coords in zip(gather_offsets, gather_times, gather_coords)]

        # Construct a dict of common kwargs
        common_kwargs = {"init": init, "bounds": bounds, "n_refractors": n_refractors,
                         "max_offset": survey_offsets.max(), "min_velocity_step": min_velocity_step,
                         "min_refractor_size": min_refractor_size, "loss": loss, "huber_coef": huber_coef, "tol": tol,
                         "is_uphole_corrected": correct_uphole, **kwargs}

        # Run parallel fit of velocity models
        rv_list = cls._fit_refractor_velocities_parallel(rv_kwargs_list, common_kwargs, chunk_size, n_workers, bar,
                                                         desc="Velocity models estimated")
        return cls(items=rv_list, survey=survey, is_geographic=is_geographic,
                   auto_create_interpolator=auto_create_interpolator)

    @classmethod
    def from_file(cls, path, survey=None, is_geographic=None, auto_create_interpolator=True, encoding="UTF-8"):
        """Load a field with near-surface velocity models from a file.

        Notes
        -----
        See more about the file format in :func:`~.utils.load_refractor_velocities`.

        Parameters
        ----------
        path : str
            Path to a file.
        survey : Survey, optional
            A survey described by the field.
        is_geographic : bool, optional
            Coordinate system of the field: either geographic (e.g. (CDP_X, CDP_Y)) or line-based (e.g. (INLINE_3D,
            CROSSLINE_3D)). Inferred from coordinates of the first near-surface velocity model in the file if not
            given.
        auto_create_interpolator : bool, optional, defaults to True
            Whether to automatically create default interpolator (RBF for more than 3 items in the field or IDW
            otherwise) upon the first call to the field.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.

        Returns
        -------
        self : RefractorVelocityField
            Constructed field.
        """
        return cls(load_refractor_velocities(path, encoding), survey=survey, is_geographic=is_geographic,
                   auto_create_interpolator=auto_create_interpolator)

    def validate_items(self, items):
        """Check if the field can be updated with the provided `items`."""
        super().validate_items(items)
        n_refractors_set = {item.n_refractors for item in items}
        if self.n_refractors is not None:
            n_refractors_set.add(self.n_refractors)
        if len(n_refractors_set) != 1:
            raise ValueError("Each RefractorVelocity must describe the same number of refractors as the field")

    def update(self, items):
        """Add new items to the field. All passed `items` must have not-None coordinates and describe the same number
        of refractors as the field.

        Parameters
        ----------
        items : RefractorVelocity or list of RefractorVelocity
            Items to add to the field.

        Returns
        -------
        self : RefractorVelocityField
            `self` with new items added. Changes `item_container` inplace and sets the `is_dirty_interpolator` flag to
            `True` if the `items` list is not empty. Sets `is_geographic` flag and `n_refractors` attribute during the
            first update if they were not defined during field creation. Updates `coords_cols` attribute if names of
            coordinates of any item being added does not match those of the field.

        Raises
        ------
        TypeError
            If wrong type of items were found.
        ValueError
            If any of the passed items have `None` coordinates or describe not the same number of refractors as the
            field.
        """
        items = to_list(items)
        super().update(items)
        if items:
            self.n_refractors = items[0].n_refractors
        return self

    @staticmethod
    def item_to_values(item):
        """Convert a field item to a 1d `np.ndarray` of its values being interpolated."""
        return np.array(list(item.params.values()))

    def _interpolate(self, coords):
        """Interpolate field values at given `coords` and postprocess them so that the following constraints are
        satisfied:
        - Intercept time is non-negative,
        - Crossover offsets are non-negative and increasing,
        - Velocities of refractors are non-negative and increasing.
        `coords` are guaranteed to be a 2d `np.ndarray` with shape (n_coords, 2), converted to the coordinate system of
        the field."""
        values = self.interpolator(coords)
        return postprocess_params(values)

    def construct_item(self, values, coords):
        """Construct an instance of `RefractorVelocity` from its `values` at given `coords`."""
        return self.item_class(**dict(zip(self.param_names, values)), coords=coords,
                               is_uphole_corrected=self.is_uphole_corrected)

    def _get_refined_values(self, interpolator_class, min_refractor_points=0, min_refractor_points_quantile=0):
        """Redefine parameters of velocity models for refractors that contain small number of points and may thus have
        produced noisy estimates during fitting.

        Parameters of such refractors are redefined using an interpolator of the given type constructed over all
        well-fit data of the field.

        The number of points in a refractor of a given field item is considered to be small if it is less than:
        - 2 or `min_refractor_points`,
        - A quantile of the number of points in the very same refractor over the whole field defined by
          `min_refractor_points_quantile`.
        """
        coords = self.coords
        values = self.values
        refined_values = values.copy()

        # Calculate the number of point in each refractor for velocity models that were fit
        n_refractor_points = np.full((self.n_items, self.n_refractors), fill_value=np.nan)
        for i, rv in enumerate(self.item_container.values()):
            if rv.is_fit:
                bin_edges = [0] + [rv.params[f"x{i}"] for i in range(1, rv.n_refractors)] + [rv.max_offset]
                n_refractor_points[i] = np.histogram(rv.offsets, bin_edges, density=False)[0]
        n_refractor_points[:, np.isnan(n_refractor_points).all(axis=0)] = 0

        # Calculate minimum acceptable number of points in each refractor, should be at least 2
        min_refractor_points = np.maximum(np.nanquantile(n_refractor_points, min_refractor_points_quantile, axis=0),
                                          max(2, min_refractor_points))
        ignore_mask = n_refractor_points < min_refractor_points
        ignore_mask[:, ignore_mask.all(axis=0)] = False  # Use a refractor anyway if it is ignored for all items

        # Refine t0 using only items with well-fit first refractor
        ignored_t0 = ignore_mask[:, 0]
        if ignored_t0.any():
            interpolator = interpolator_class(coords[~ignored_t0], values[~ignored_t0, 0])
            refined_values[ignored_t0, 0] = interpolator(coords[ignored_t0])

        # Refine crossover offsets using only items with well-fit neighboring refractors
        for i in range(1, self.n_refractors):
            ignored_xi = ignore_mask[:, i - 1] | ignore_mask[:, i]
            if ignored_xi.any():
                interpolator = interpolator_class(coords[~ignored_xi], values[~ignored_xi, i])
                refined_values[ignored_xi, i] = interpolator(coords[ignored_xi])

        # Refine velocities using only items with well-fit corresponding refractor
        for i in range(self.n_refractors, 2 * self.n_refractors):
            ignored_vi = ignore_mask[:, i - self.n_refractors]
            if ignored_vi.any():
                interpolator = interpolator_class(coords[~ignored_vi], values[~ignored_vi, i])
                refined_values[ignored_vi, i] = interpolator(coords[ignored_vi])

        # Postprocess refined values
        return postprocess_params(refined_values)

    def _get_smoothed_values(self, radius=None, neighbors=None, min_refractor_points=0,
                             min_refractor_points_quantile=0):
        """Average refractor parameters within a given `radius` while ignoring refractors that contain less points
        than:
        - 2 or `min_refractor_points`,
        - A quantile of the number of points in the very same refractor over the whole field defined by
          `min_refractor_points_quantile`.
        """
        if radius is None:
            radius = self.default_neighborhood_radius
        interpolator = partial(IDWInterpolator, radius=radius, neighbors=neighbors)
        refined_values = self._get_refined_values(interpolator, min_refractor_points, min_refractor_points_quantile)
        return interpolator(self.coords, refined_values, dist_transform=0)(self.coords)

    def create_interpolator(self, interpolator, min_refractor_points=0, min_refractor_points_quantile=0, **kwargs):
        """Create a field interpolator whose name is defined by `interpolator`.

        Available options are:
        - "idw" - to create `IDWInterpolator`,
        - "delaunay" - to create `DelaunayInterpolator`,
        - "ct" - to create `CloughTocherInterpolator`,
        - "rbf" - to create `RBFInterpolator`.

        Parameters
        ----------
        interpolator : str
            Name of the interpolator to create.
        min_refractor_points : int, optional, defaults to 0
            Ignore parameters of refractors with less than `min_refractor_points` points during interpolation.
        min_refractor_points_quantile : float, optional, defaults to 0
            Defines quantiles of the number of points in each refractor of the field. Parameters of refractors with
            less points than the corresponding quantile are ignored during interpolation.
        kwargs : misc, optional
            Additional keyword arguments to be passed to the constructor of interpolator class.

        Returns
        -------
        field : Field
            A field with created interpolator. Sets `is_dirty_interpolator` flag to `False`.
        """
        interpolator_class = partial(self._get_interpolator_class(interpolator), **kwargs)
        values = self._get_refined_values(interpolator_class, min_refractor_points, min_refractor_points_quantile)
        self.interpolator = interpolator_class(self.coords, values)
        self.is_dirty_interpolator = False
        return self

    def smooth(self, radius=None, neighbors=4, min_refractor_points=0, min_refractor_points_quantile=0):
        """Smooth the field by averaging its velocity models within given radius.

        Parameters
        ----------
        radius : positive float, optional
            Spatial window radius (Euclidean distance). Equals to `self.default_neighborhood_radius` if not given.
        neighbors : int, optional, defaults to 4
            The number of neighbors to use for averaging if no velocities are considered to be well-fit in given
            `radius` according to provided `min_refractor_points` and `min_refractor_points_quantile`.
        min_refractor_points : int, optional, defaults to 0
            Ignore parameters of refractors with less than `min_refractor_points` points during averaging.
        min_refractor_points_quantile : float, optional, defaults to 0
            Defines quantiles of the number of points in each refractor of the field. Parameters of refractors with
            less points than the corresponding quantile are ignored during averaging.

        Returns
        -------
        field : RefractorVelocityField
            Smoothed field.
        """
        if self.is_empty:
            return type(self)(survey=self.survey, is_geographic=self.is_geographic)

        smoothed_values = self._get_smoothed_values(radius, neighbors, min_refractor_points,
                                                    min_refractor_points_quantile)
        smoothed_items = [self.item_class(**dict(zip(self.param_names, val)), coords=rv.coords,
                                          is_uphole_corrected=rv.is_uphole_corrected)
                          for rv, val in zip(self.items, smoothed_values)]
        return type(self)(smoothed_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic, auto_create_interpolator=self.auto_create_interpolator)

    def refine(self, radius=None, neighbors=4, min_refractor_points=0, min_refractor_points_quantile=0,
               relative_bounds_size=0.25, chunk_size=250, n_workers=None, bar=True):
        """Refine the field by first smoothing it and then refitting each velocity model within narrow parameter bounds
        around smoothed values. Only fields that were constructed directly from offset-traveltime data can be refined.

        Parameters
        ----------
        radius : positive float, optional
            Spatial window radius for smoothing (Euclidean distance). Equals to `self.default_neighborhood_radius` if
            not given.
        neighbors : int, optional, defaults to 4
            The number of neighbors to use for smoothing if no velocities are considered to be well-fit in given
            `radius` according to provided `min_refractor_points` and `min_refractor_points_quantile`.
        min_refractor_points : int, optional, defaults to 0
            Ignore parameters of refractors with less than `min_refractor_points` points during smoothing.
        min_refractor_points_quantile : float, optional, defaults to 0
            Defines quantiles of the number of points in each refractor of the field. Parameters of refractors with
            less points than the corresponding quantile are ignored during smoothing.
        relative_bounds_size : float, optional, defaults to 0.25
            Size of parameters bound used to refit velocity models relative to their range in the smoothed field. The
            bounds are centered around smoothed parameter values.
        chunk_size : int, optional, defaults to 250
            The number of velocity models refined by each of spawned processes.
        n_workers : int, optional
            The maximum number of simultaneously spawned processes to refine velocity models. Defaults to the number of
            cpu cores.
        bar : bool, optional, defaults to True
            Whether to show a refinement progress bar.

        Returns
        -------
        field : RefractorVelocityField
            Refined field.
        """
        if self.is_empty:
            return type(self)(survey=self.survey, is_geographic=self.is_geographic)
        if not self.is_fit:
            raise ValueError("Only fields that were constructed directly from offset-traveltime data can be refined")

        # Smooth parameters of near-surface velocity models in the field and define bounds for their optimization
        params_init = self._get_smoothed_values(radius, neighbors, min_refractor_points, min_refractor_points_quantile)
        bounds_size = params_init.ptp(axis=0) * relative_bounds_size / 2

        # Clip all bounds to be non-negative
        params_bounds = np.stack([np.maximum(params_init - bounds_size, 0), params_init + bounds_size], axis=2)

        # Clip init and bounds for crossover offsets to be no greater than max offset
        max_offsets = np.array([rv.max_offset for rv in self.items])
        np.minimum(params_init[:, 1:self.n_refractors], max_offsets[:, None], out=params_init[:, 1:self.n_refractors])
        np.minimum(params_bounds[:, 1:self.n_refractors], max_offsets[:, None, None],
                   out=params_bounds[:, 1:self.n_refractors])

        # Construct a dict of refinement parameters for each velocity model
        rv_kwargs_list = [{"offsets": rv.offsets, "times": rv.times, "init": dict(zip(self.param_names, init)),
                           "bounds": dict(zip(self.param_names, bounds)), "max_offset": rv.max_offset,
                           "coords": rv.coords, "is_uphole_corrected": rv.is_uphole_corrected}
                          for rv, init, bounds in zip(self.items, params_init, params_bounds)]
        common_kwargs = {"min_velocity_step": 0, "min_refractor_size": 0}

        # Run parallel refinement of velocity models
        refined_items = self._fit_refractor_velocities_parallel(rv_kwargs_list, common_kwargs, chunk_size, n_workers,
                                                                bar, desc="Velocity models refined")
        return type(self)(refined_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic, auto_create_interpolator=self.auto_create_interpolator)

    def dump(self, path, encoding="UTF-8"):
        """Dump near-surface velocity models stored in the field to a file.

        Notes
        -----
        See more about the file format in :func:`~.utils.load_refractor_velocities`.

        Parameters
        ----------
        path : str
            Path to the created file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.

        Raises
        ------
        ValueError
            If the field is empty.
        """
        if self.is_empty:
            raise ValueError("Empty field can't be dumped.")
        dump_refractor_velocities(self.items, path=path, encoding=encoding)

    def plot(self, **kwargs):
        """Plot an interactive map of each parameter of a near-surface velocity model and display an offset-traveltime
        curve upon clicking on a map. If some velocity models in the field were constructed directly from first break
        data, a scatter plot of offsets and times of first breaks is also displayed.

        Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
            Size of the created figures. Measured in inches.
        refractor_velocity_plot_kwargs : dict, optional
            Additional keyword arguments to be passed to `RefractorVelocity.plot`.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `MetricMap.plot`.
        """
        FieldPlot(self, **kwargs).plot()

    #pylint: disable-next=invalid-name
    def qc(self, metrics=None, survey=None, first_breaks_header=HDR_FIRST_BREAK, correct_uphole=None,
           n_workers=None, bar=True, chunk_size=250):
        """Perform quality control of the first breaks given the near-surface velocity model.

        By default, the following metrics are calculated:
        * The first break outliers metric. A first break time is considered to be an outlier if it differs from the
          expected arrival time defined by an offset-traveltime curve by more than a given threshold.
        * Mean amplitude of the signal in the moment of first break.
        * Mean absolute deviation of the signal phase from target value in the moment of first break.
        * Mean Pearson correlation coefficient of trace with mean hodograph in window around the first break.
        * An offset after which first breaks start diverging from the expected arrival time defined by the near-surface
          velocity model.

        Parameters
        ----------
        metrics : instance or subclass of :class:`~metrics.RefractorVelocityMetric` or list of them, optional
            Metrics to calculate. Defaults to those defined in `~metrics.REFRACTOR_VELOCITY_QC_METRICS`.
        survey : Survey, optional
            Survey to load traces from. Defaults to a survey the field is linked to.
        first_breaks_header : str, optional, defaults to :const:`~const.HDR_FIRST_BREAK`
            Column name from `survey.headers` where times of first break are stored.
        correct_uphole : bool, optional
            Whether to perform uphole correction by adding values of "SourceUpholeTime" header to times of first breaks
            emulating the case when sources are located on the surface. If not given, correction is performed if
            "SourceUpholeTime" header is loaded and `self` if uphole-corrected.
        n_workers : int, optional
            The number of threads to be spawned to calculate metrics. Defaults to the number of cpu cores.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.
        chunk_size : int, optional, defaults to 250
            The number of gathers to calculate metrics for in each of spawned threads.

        Returns
        -------
        metrics_maps : MetricMap or list of MetricMap
            Calculated metrics maps. Has the same length as `metrics`.
        """
        if survey is None:
            if not self.has_survey:
                raise ValueError("`survey` must be passed if the field is not linked with a survey.")
            survey = self.survey

        metrics = REFRACTOR_VELOCITY_QC_METRICS if metrics is None else metrics
        metrics_instances, is_single_metric = initialize_metrics(metrics, metric_class=RefractorVelocityMetric)
        if correct_uphole is None:
            correct_uphole = "SourceUpholeTime" in survey.available_headers and self.is_uphole_corrected
        for metric in metrics_instances:
            metric.set_defaults(first_breaks_header=first_breaks_header, correct_uphole=correct_uphole)

        coords_cols = get_coords_cols(survey.indexed_by)
        is_geographic = coords_cols in GEOGRAPHIC_COORDS

        gather_change_ix = np.where(~survey.headers.index.duplicated(keep="first"))[0]
        gather_coords = survey[coords_cols][gather_change_ix]

        n_chunks, mod = divmod(survey.n_gathers, chunk_size)
        if mod:
            n_chunks += 1
        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)

        def calc_metrics(gather_indices_chunk, coords_chunk):
            """Calculate metrics for a given chunk of gather indices."""
            refractor_velocities = self(coords_chunk, is_geographic=is_geographic)
            results = []
            for idx, rv in zip(gather_indices_chunk, refractor_velocities):
                gather = survey.get_gather(idx)
                gather_results = [metric(gather, refractor_velocity=rv) for metric in metrics_instances]
                results.append(gather_results)
            return results

        executor_class = ForPoolExecutor if n_workers == 1 else ThreadPoolExecutor
        futures = []
        with tqdm(total=survey.n_gathers, desc="Gathers processed", disable=not bar) as pbar:
            with executor_class(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    gathers_indices_chunk = survey.indices[i * chunk_size : (i + 1) * chunk_size]
                    coords_chunk = gather_coords[i * chunk_size : (i + 1) * chunk_size]
                    future = pool.submit(calc_metrics, gathers_indices_chunk, coords_chunk)
                    future.add_done_callback(lambda fut: pbar.update(len(fut.result())))
                    futures.append(future)
        results = sum([future.result() for future in futures], [])

        metrics_maps = []
        metrics_instances = [metric.provide_context(survey=survey, field=self) for metric in metrics_instances]
        metrics_maps = [metric.construct_map(gather_coords, metric_values, index=survey.indices)
                        for metric, metric_values in zip(metrics_instances, zip(*results))]
        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
