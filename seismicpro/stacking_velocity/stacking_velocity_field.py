"""Implements a StackingVelocityField class which stores stacking velocities calculated at different field locations
and allows for spatial velocity interpolation"""

import os
from functools import cached_property

import numpy as np
from tqdm.contrib.concurrent import thread_map
from sklearn.neighbors import NearestNeighbors

from .stacking_velocity import StackingVelocity
from .metrics import VELOCITY_QC_METRICS, StackingVelocityMetric
from ..metrics import initialize_metrics
from ..field import ValuesAgnosticField, VFUNCFieldMixin
from ..utils import IDWInterpolator


class StackingVelocityField(ValuesAgnosticField, VFUNCFieldMixin):
    """A class for storing and interpolating stacking velocity data over a field.

    Velocities used for seismic cube stacking are usually picked on a sparse grid of inlines and crosslines and then
    interpolated over the whole field in order to reduce computational costs. Such interpolation can be performed by
    `StackingVelocityField` class which provides an interface to obtain stacking velocity at given spatial coordinates
    via its `__call__` method.

    A field can be populated with stacking velocities in 3 main ways:
    - by passing precalculated velocities in the `__init__`,
    - by creating an empty field and then iteratively updating it with calculated stacking velocities using `update`,
    - by loading a field from a file of vertical functions via its `from_file` `classmethod`.

    After all velocities are added, field interpolator should be created to make the field callable. It can be done
    either manually by executing `create_interpolator` method or automatically during the first call to the field if
    `auto_create_interpolator` flag was set to `True` upon field instantiation. Manual interpolator creation is useful
    when one wants to fine-tune its parameters or the field should be later passed to different processes (e.g. in a
    pipeline with prefetch with `mpc` target) since otherwise the interpolator will be independently created in all the
    processes.

    The field provides an interface to its quality control via `qc` method, which returns maps for several
    spatial-window-based metrics calculated for its stacking velocities. These maps can be interactively visualized to
    assess field quality in detail.

    Examples
    --------
    A field can be created empty and updated with instances of `StackingVelocity` class:
    >>> field = StackingVelocityField()
    >>> velocity = StackingVelocity(times=[0, 1000, 2000, 3000], velocities=[1500, 2000, 2800, 3400],
    ...                             coords=Coordinates((20, 40), names=("INLINE_3D", "CROSSLINE_3D")))
    >>> field.update(velocity)

    Or created from precalculated instances:
    >>> field = StackingVelocityField(list_of_stacking_velocities)

    Or simply loaded from a file of vertical functions:
    >>> field = StackingVelocityField.from_file(path)

    Field interpolator will be created automatically upon the first call by default, but one may do it explicitly by
    executing `create_interpolator` method:
    >>> field.create_interpolator("delaunay")

    Now the field allows for velocity interpolation at given coordinates:
    >>> velocity = field((10, 10))

    Or can be passed directly to some gather processing methods:
    >>> gather = survey.sample_gather().apply_nmo(field)

    Quality control can be performed by calling `qc` method and visualizing the resulting maps:
    >>> metrics_maps = field.qc(radius=40, times=np.arange(0, 3000, 2))
    >>> for metric_map in metrics_maps:
    >>>     metric_map.plot(interactive=True)

    Parameters
    ----------
    items : StackingVelocity or list of StackingVelocity, optional
        Stacking velocities to be added to the field on instantiation. If not given, an empty field is created.
    survey : Survey, optional
        A survey described by the field.
    is_geographic : bool, optional
        Coordinate system of the field: either geographic (e.g. (CDP_X, CDP_Y)) or line-based (e.g. (INLINE_3D,
        CROSSLINE_3D)). Inferred automatically on the first update if not given.
    auto_create_interpolator : bool, optional, defaults to True
        Whether to automatically create default interpolator (IDW) upon the first call to the field.

    Attributes
    ----------
    survey : Survey or None
        A survey described by the field. `None` if not specified during instantiation.
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
        Whether to automatically create default interpolator (IDW) upon the first call to the field.
    """
    item_class = StackingVelocity

    def construct_item(self, items, weights, coords):
        """Construct a new stacking velocity by averaging other stacking velocities with corresponding weights.

        Parameters
        ----------
        items : list of StackingVelocity
            Stacking velocities to be aggregated.
        weights : list of float
            Weight of each item in `items`.
        coords : Coordinates
            Spatial coordinates of a stacking velocity being constructed.

        Returns
        -------
        item : StackingVelocity
            Constructed stacking velocity instance.
        """
        return self.item_class.from_stacking_velocities(items, weights, coords=coords)

    @cached_property
    def mean_velocity(self):
        """StackingVelocity: Mean stacking velocity over the field."""
        return self.item_class.from_stacking_velocities(self.items)

    def smooth(self, radius=None):
        """Smooth the field by averaging its stacking velocities within given radius.

        Parameters
        ----------
        radius : positive float, optional
            Spatial window radius (Euclidean distance). Equals to `self.default_neighborhood_radius` if not given.

        Returns
        -------
        field : StackingVelocityField
            Smoothed field.
        """
        if self.is_empty:
            return type(self)(survey=self.survey, is_geographic=self.is_geographic)
        if radius is None:
            radius = self.default_neighborhood_radius
        smoother = IDWInterpolator(self.coords, radius=radius, dist_transform=0)
        weights = smoother.get_weights(self.coords)
        items_coords = [item.coords for item in self.item_container.values()]
        smoothed_items = self.weights_to_items(weights, items_coords)
        return type(self)(smoothed_items, survey=self.survey, is_geographic=self.is_geographic)

    def interpolate(self, coords, times, is_geographic=None):
        """Interpolate stacking velocities at given `coords` and `times`.

        Interpolation over a regular grid of times allows implementing a much more efficient computation strategy than
        simply iteratively calling the interpolator for each of `coords` and than evaluating the obtained stacking
        velocities at `times`.

        Parameters
        ----------
        coords : 2d np.array or list of Coordinates
            Coordinates to interpolate stacking velocities at.
        times : 1d array-like
            Times to interpolate stacking velocities at.
        is_geographic : bool, optional
            Coordinate system of all non-`Coordinates` entities of `coords`. Assumed to be in the coordinate system of
            the field by default.

        Returns
        -------
        velocities : 2d np.ndarray
            Interpolated stacking velocities at given `coords` and `times`. Has shape (n_coords, n_times).
        """
        self.validate_interpolator()
        field_coords, _, _ = self.transform_coords(coords, is_geographic=is_geographic)
        times = np.atleast_1d(times)
        weights = self.interpolator.get_weights(field_coords)
        base_velocities_coords = set.union(*[set(weights_dict.keys()) for weights_dict in weights])
        base_velocities = {coords: self.item_container[coords](times) for coords in base_velocities_coords}

        res = np.zeros((len(field_coords), len(times)))
        for i, weights_dict in enumerate(weights):
            for coord, weight in weights_dict.items():
                res[i] += base_velocities[coord] * weight
        return res

    #pylint: disable-next=invalid-name
    def qc(self, metrics=None, radius=None, coords=None, times=None, n_workers=None, bar=True):
        """Perform quality control of the velocity field by calculating spatial-window-based metrics for its stacking
        velocities evaluated at given `coords` and `times`.

        If `coords` are not given, coordinates of items in the field are used, but interpolation is performed anyway.
        If `times` are not given, samples of the underlying `Survey` are used if it is defined.

        By default, the following metrics are calculated:
        * Presence of segments with velocity inversions,
        * Maximal deviation of instantaneous acceleration from mean acceleration over all times,
        * Maximal spatial velocity standard deviation in a window over all times,
        * Maximal absolute relative difference between central stacking velocity and the average of all remaining
          velocities in the window over all times.

        Parameters
        ----------
        metrics : StackingVelocityMetric or list of StackingVelocityMetric, optional
            Metrics to calculate. Defaults to those defined in `~metrics.VELOCITY_QC_METRICS`.
        radius : positive float, optional
            Spatial window radius (Euclidean distance). Equals to `self.default_neighborhood_radius` if not given.
        coords : 2d np.array or list of Coordinates, optional
            Spatial coordinates of stacking velocities to calculate metrics for. If not given, coordinates of items in
            the field are used.
        times : 1d array-like, optional
            Times to calculate metrics for. By default, samples of the underlying `Survey` are used. Measured in
            milliseconds.
        n_workers : int, optional
            The number of threads to be spawned to calculate metrics. Defaults to the number of cpu cores.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        metrics_maps : StackingVelocityMetricMap or list of StackingVelocityMetricMap
            Calculated metrics maps. Has the same shape as `metrics`.
        """
        if metrics is None:
            metrics = VELOCITY_QC_METRICS
        metrics, is_single_metric = initialize_metrics(metrics, metric_class=StackingVelocityMetric)

        # Set default radius, coords and times
        if radius is None:
            radius = self.default_neighborhood_radius
        if coords is None:
            coords = self.coords
        if times is None:
            if not self.has_survey:
                raise ValueError("times must be passed if the field is not linked with a survey")
            times = self.survey.times

        # Calculate stacking velocities at given times for each of coords
        velocities = self.interpolate(coords, times)

        # Select all neighboring stacking velocities for each of coords
        if n_workers is None:
            n_workers = os.cpu_count()
        coords_neighbors = NearestNeighbors(radius=radius, n_jobs=n_workers).fit(coords)
        # Sort results to guarantee that central stacking velocity of each window will have index 0
        _, windows_indices = coords_neighbors.radius_neighbors(coords, return_distance=True, sort_results=True)

        # Calculate metrics and construct maps
        def calculate_metrics(window_indices):
            window_velocities = velocities[window_indices]
            return [metric(window_velocities, times) for metric in metrics]  # pylint: disable=too-many-function-args

        results = thread_map(calculate_metrics, windows_indices, max_workers=n_workers,
                             desc="Coordinates processed", disable=not bar)
        context = {"times": times, "velocities": velocities, "coords_neighbors": coords_neighbors}
        metrics_maps = [metric.provide_context(**context).construct_map(coords, values, coords_cols=self.coords_cols)
                        for metric, values in zip(metrics, zip(*results))]
        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
