"""Implements classes and functions for 2d spatial interpolation and extrapolation.

Two types of interpolators are introduced:
- `SpatialInterpolator` - requires both `coords` and `values` to be passed during instantiation and implements
  `__call__` method which performs interpolation or extrapolation at given coordinates.
  Available options are `CloughTocherInterpolator` and `RBFInterpolator`.
- `ValuesAgnosticInterpolator` - requires only `coords` to be passed in `__init__`. It implements `get_weights` method
  which returns coordinates of reference items and their weights for each of the passed `coords`. Items should then be
  manually averaged with the corresponding weights to perform interpolation. An interpolator can be called directly
  only if `values` were passed to `__init__`.
  Available options are `IDWInterpolator` and `DelaunayInterpolator`.

Values-agnostic interpolators rely only on positions of objects, not their values. This may be useful in cases when:
- obtaining known function values may be time- or memory-consuming: values at reference coordinates may be accessed
  on-the-fly without calculating or preloading them all at once,
- objects being interpolated cannot be easily converted to a numeric vector but a convenient way exists to construct a
  new object from the existing ones by averaging them with given weights.
Unfortunately, the lack of information about function values may introduce artifacts such as bull's eye effect of IDW
interpolator or loss of smoothness at simplex boundaries of Delaunay interpolator.

On the contrary, `SpatialInterpolator`s construct a more complex and adequate interpolant, but are generally more
time-consuming, especially in case of high-dimensional values.
"""

import cv2
import numpy as np
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.spatial.qhull import Delaunay, QhullError  # pylint: disable=no-name-in-module


def parse_inputs(coords, values=None):
    """Transform passed `coords` to a 2d array with shape (n_coords, 2) and `values` to a 2d array with shape
    (n_coords, n_values). Check correctness and consistency of shapes. If `values` are not passed, only `coords` are
    processed."""
    coords = np.array(coords, order="C")
    is_coords_1d = coords.ndim == 1
    coords = np.atleast_2d(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_coords, 2) or (2,)")
    if values is None:
        return coords, is_coords_1d, None, None

    values = np.array(values, dtype=np.float64, order="C")
    is_values_1d = values.ndim == 1
    if is_values_1d and is_coords_1d:
        is_values_1d = False
        values = np.atleast_2d(coords)
    if is_values_1d:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or len(values) != len(coords):
        raise ValueError("values must have shape (n_coords,) or (n_coords, n_values)")
    return coords, is_coords_1d, values, is_values_1d


class SpatialInterpolator:
    """Base class for a spatial interpolator. Each concrete subclass must implement `_interpolate` method."""
    def __init__(self, coords, values):
        self.coords, _, self.values, self.is_values_1d = parse_inputs(coords, values)

    def _interpolate(self, coords):
        """Perform interpolation at given `coords`. `coords` are guaranteed to be 2-dimensional with shape
        (n_coords, 2)."""
        _ = coords
        raise NotImplementedError

    def __call__(self, coords):
        """Evaluate the interpolant at passed `coords`.

        Parameters
        ----------
        coords : np.ndarray with shape (2,) or (n_coords, 2)
            Coordinates to evaluate the interpolant at.

        Returns
        -------
        values : scalar or 1d np.ndarray with shape (n_values,) or 2d np.ndarray with shape (n_coords, n_values)
            Interpolation results. Output shape is determined by the shapes of `coords` passed to `__call__` and
            `values` passed to `__init__`.
        """
        coords, is_coords_1d, _, _ = parse_inputs(coords)
        values = self._interpolate(coords)
        if self.is_values_1d:
            values = values[:, 0]
        if is_coords_1d:
            return values[0]
        return values


class ValuesAgnosticInterpolator(SpatialInterpolator):
    """Base class for a values-agnostic interpolator. Each concrete subclass must implement `_interpolate` and
    `_get_weights` methods."""
    def __init__(self, coords, values=None):
        super().__init__(coords, values)

    @property
    def has_values(self):
        """bool: Whether `values` were passed during instantiation."""
        return self.values is not None

    def _get_weights(self, coords):
        """Get coordinates of reference objects and their weights for each of the passed `coords`. `coords` are
        guaranteed to be 2-dimensional with shape (n_coords, 2)."""
        _ = coords
        raise NotImplementedError

    def get_weights(self, coords):
        """Get coordinates of reference objects and their weights for each of the passed `coords`.

        Parameters
        ----------
        coords : np.ndarray with shape (2,) or (n_coords, 2)
            Coordinates to determine reference objects for.

        Returns
        -------
        weights : dict or 1d np.ndarray of dicts
            Coordinates and weights of reference objects for each of `coords`. Coordinates are stored in `dict` keys as
            a tuple with 2 elements, their weight is stored in the corresponding value.
        """
        coords, is_coords_1d, _, _ = parse_inputs(coords)
        weighted_coords = self._get_weights(coords)
        if is_coords_1d:
            return weighted_coords[0]
        return weighted_coords

    def __call__(self, coords):
        """Evaluate the interpolant at passed `coords`. Available only if `values` were passed during instantiation.

        Parameters
        ----------
        coords : np.ndarray with shape (2,) or (n_coords, 2)
            Coordinates to evaluate the interpolant at.

        Returns
        -------
        values : scalar or 1d np.ndarray with shape (n_values,) or 2d np.ndarray with shape (n_coords, n_values)
            Interpolation results. Output shape is determined by the shapes of `coords` passed to `__call__` and
            `values` passed to `__init__`.
        """
        if not self.has_values:
            raise ValueError("The interpolator requires values to be passed to be callable")
        return super().__call__(coords)


class IDWInterpolator(ValuesAgnosticInterpolator):
    """Construct an inverse distance weighting (IDW) interpolator.

    IDW assumes that function value being interpolated at some coordinates is closer to values at known points nearby
    compared to those located further away. Interpolation is performed by averaging known function values in a
    neighborhood of the requested coordinates with weights proportional to inverse distances to the corresponding
    points.

    The detailed algorithm looks as follows:
    1. Select known points in the neighborhood of the requested coordinates defined either by `radius` or `neighbors`.
       If none of them is given all data points are selected.
    2. Calculate euclidean distance from coordinates to each selected point.
    3. Transform calculated distances according to `dist_transform`. If `dist_transform` is `float` it defines the
       power to raise the distances to.
    4. Add `smoothing` value to the transformed distances and invert the results to obtain unnormalized weights. Zero
       `smoothing` results in fair interpolation: evaluating IDW at known points will return the corresponding known
       values. Positive `smoothing` allows violating this constraint and makes the interpolant more robust to outliers.
       Interpolation result will converge to the average value of the function in the selected neighborhood when
       `smoothing` approaches infinity.
    5. Normalize calculated weights so that their sum equals 1 and use them to aggregate function values.

    `IDWInterpolator` is values-agnostic and thus provides `get_weights` interface even without `values` passed in
    `__init__`.

    Parameters
    ----------
    coords : 2d np.ndarray with shape (n_coords, 2)
        Coordinates of data points.
    values : 1d np.ndarray with shape (n_coords,) or 2d np.ndarray with shape (n_coords, n_values), optional
        Data values at `coords`. If not given, only `get_weights` interface is provided.
    radius : float, optional
        Radius of the neighborhood to select known points from for each pair of coordinates. If not given or no known
        point is closer to given coordinates than `radius`, the neighborhood is defined by `neighbors`.
    neighbors : int, optional
        The number of closest data points to use for interpolation if `radius` is not given or no known points were
        found for `radius`. All the data points are used by default.
    dist_transform : float or callable, optional, defaults to 2
        A function used to transform distances before smoothing, inverting and normalizing into weights. If `float`,
        defines the power to raise the distances to.
    smoothing : float, optional, defaults to 0
        A constant to be added to transformed distances before inverting and normalizing into weights. The interpolant
        perfectly fits the data when `smoothing` is set to 0. For large values, the interpolant approaches mean
        function value in the neighborhood being considered.
    min_relative_weight : float, optional, defaults to 1e-3
        Ignore points whose weight divided by the weight of the closest point is lower than `min_relative_weight`
        during interpolation. Weights of the remaining points are renormalized so that their sum equals 1. Allows
        reducing computational costs since weights generally decrease rapidly for distant points and the number of
        points that affect the interpolation result significantly is low.

    Attributes
    ----------
    nearest_neighbors : KDTree
        An estimator of the closest known points to the passed spatial coordinates.
    use_radius : bool
        Whether a neighborhood of coordinates is defined by `radius` or the number of `neighbors`.
    dist_transform : float or callable
        A function used to transform distances before smoothing, inverting and normalizing into weights.
    smoothing : float
        A constant to be added to transformed distances before inverting and normalizing into weights.
    min_relative_weight : float
        Ignore points whose weight divided by the weight of the closest point is lower than `min_relative_weight`
        during interpolation.
    """
    def __init__(self, coords, values=None, radius=None, neighbors=None, dist_transform=2, smoothing=0,
                 min_relative_weight=1e-3):
        super().__init__(coords, values)
        if neighbors is None:
            neighbors = len(self.coords)
        self.neighbors = np.arange(min(neighbors, len(self.coords))) + 1  # One-based indices of neighbors to get
        self.radius = radius
        self.use_radius = radius is not None
        self.nearest_neighbors = KDTree(self.coords)
        self.dist_transform = dist_transform
        self.smoothing = smoothing
        self.min_relative_weight = min_relative_weight

    def _distances_to_weights(self, dist):
        """Convert distances to neighboring points into weights."""
        is_1d_dist = dist.ndim == 1
        dist = np.atleast_2d(dist)

        # Transform distances according to dist_transform and smooth them if needed
        if callable(self.dist_transform):
            dist = self.dist_transform(dist)
        else:
            dist **= self.dist_transform
        dist += self.smoothing

        # Calculate weights from distances: correctly handle case of interpolating at known coords
        zero_mask = np.isclose(dist, 0)
        dist[zero_mask] = 1  # suppress division by zero warning
        weights = 1 / dist
        weights[zero_mask.any(axis=1)] = 0
        weights[zero_mask] = 1

        # Zero out weights less than a threshold and norm the result
        weights[weights / weights.max(axis=1, keepdims=True) < self.min_relative_weight] = 0
        weights /= weights.sum(axis=1, keepdims=True)

        if is_1d_dist:
            return weights[0]
        return weights

    def _aggregate_values(self, indices, weights):
        """Average values with given `indices` with corresponding `weights`. Both `indices` and `weights` are 2d
        `np.ndarray`s with shape (n_items, n_indices)."""
        return (self.values[indices] * weights[:, :, None]).sum(axis=1).astype(self.values.dtype)

    def _get_reference_indices_neighbors(self, coords):
        """Get indices of reference data points and their weights for each item in `coords` if the neighborhood is
        defined by the number of `neighbors`."""
        dist, indices = self.nearest_neighbors.query(coords, k=self.neighbors, workers=-1)
        return indices, self._distances_to_weights(dist)

    def _interpolate_neighbors(self, coords):
        """Perform interpolation at given `coords` if the neighborhood is defined by the number of `neighbors`."""
        if len(coords) == 0:
            return np.empty((0, self.values.shape[1]), dtype=self.values.dtype)
        base_indices, base_weights = self._get_reference_indices_neighbors(coords)
        return self._aggregate_values(base_indices, base_weights)

    def _get_weights_neighbors(self, coords):
        """Get coordinates of reference objects and their weights for each of the passed `coords` if the neighborhood
        is defined by the number of `neighbors`."""
        if len(coords) == 0:
            return np.empty(0, dtype=object)
        base_indices, base_weights = self._get_reference_indices_neighbors(coords)
        base_coords = self.coords[base_indices]
        non_zero_mask = ~np.isclose(base_weights, 0)
        weighted_coords = [{tuple(coord): weight for coord, weight in zip(coords[mask], weights[mask])}
                           for coords, weights, mask in zip(base_coords, base_weights, non_zero_mask)]
        return np.array(weighted_coords, dtype=object)

    def _get_reference_indices_radius(self, coords):
        """Get indices of reference data points and their weights for each item in `coords` with non-empty neighborhood
        defined by `radius`. Also returns a mask of items in `coords` with empty neighborhood."""
        n_radius_points = self.nearest_neighbors.query_ball_point(coords, r=self.radius, return_length=True,
                                                                  workers=-1)
        empty_radius_mask = n_radius_points == 0
        if empty_radius_mask.all():
            return np.empty((0, 0)), np.empty((0, 0)), empty_radius_mask

        # Unlike query_ball_point, query sometimes does not return points exactly radius distance away from the
        # requested coordinates due to inaccuracies of float arithmetics. This may result in nan interpolation result
        # if no points are returned. Here the radius in increased according to default numpy tolerances so that all
        # returned points are either inside a circle of interpolation radius or close to its border for each of the
        # requested coordinates.
        isclose_rtol = 1e-05
        isclose_atol = 1e-08
        radius = self.radius * (1 + isclose_rtol) + isclose_atol
        neighbors = np.arange(n_radius_points.max()) + 1
        dist, indices = self.nearest_neighbors.query(coords[~empty_radius_mask], k=neighbors,
                                                     distance_upper_bound=radius, workers=-1)
        indices[np.isinf(dist)] = 0  # Set padded indices to 0 for further advanced indexing to properly work
        return indices, self._distances_to_weights(dist), empty_radius_mask

    def _interpolate_radius(self, coords):
        """Perform interpolation at given `coords` if the neighborhood is defined by `radius`. Falls back to
        `_interpolate_neighbors` for coordinates with empty neighborhood."""
        base_indices, base_weights, empty_radius_mask = self._get_reference_indices_radius(coords)
        values = np.empty((len(coords), self.values.shape[1]), dtype=self.values.dtype)
        values[empty_radius_mask] = self._interpolate_neighbors(coords[empty_radius_mask])
        if len(base_indices):
            values[~empty_radius_mask] = self._aggregate_values(base_indices, base_weights)
        return values

    def _get_weights_radius(self, coords):
        """Get coordinates of reference objects and their weights for each of the passed `coords` if the neighborhood
        is defined by `radius`. Falls back to `_get_weights_neighbors` for coordinates with empty neighborhood."""
        base_indices, base_weights, empty_radius_mask = self._get_reference_indices_radius(coords)
        values = np.empty(len(coords), dtype=object)
        values[empty_radius_mask] = self._get_weights_neighbors(coords[empty_radius_mask])

        weighted_coords = np.empty(len(base_indices), dtype=object)
        non_zero_mask = ~np.isclose(base_weights, 0)
        for i, (indices, weights, mask) in enumerate(zip(base_indices, base_weights, non_zero_mask)):
            coords = self.coords[indices[mask]]
            weights = weights[mask]
            weighted_coords[i] = {tuple(coord): weight for coord, weight in zip(coords, weights)}
        values[~empty_radius_mask] = weighted_coords
        return values

    def _interpolate(self, coords):
        """Perform interpolation at given `coords`."""
        if self.use_radius:
            return self._interpolate_radius(coords)
        return self._interpolate_neighbors(coords)

    def _get_weights(self, coords):
        """Get coordinates of reference objects and their weights for each of the passed `coords`."""
        if self.use_radius:
            return self._get_weights_radius(coords)
        return self._get_weights_neighbors(coords)


class BaseDelaunayInterpolator(SpatialInterpolator):
    """A base class for interpolators built on top of Delaunay triangulation of data points.

    The class triangulates input `coords` and stores the result in `tri` attribute of the created instance. It also
    constructs an IDW interpolator which is used to perform extrapolation for coordinates lying outside the convex hull
    of data points. Each concrete subclass must implement `_interpolate_inside_hull` method.
    """
    def __init__(self, coords, values=None, neighbors=3, dist_transform=2):
        super().__init__(coords, values)

        # Construct a convex hull of passed coords. Cast coords to float32, otherwise cv2 may fail. cv2 is used since
        # QHull can't handle degenerate hulls.
        self.coords_hull = cv2.convexHull(self.coords.astype(np.float32), returnPoints=True)

        # Construct an IDW interpolator to use outside the constructed hull
        self.idw_interpolator = IDWInterpolator(coords, values, neighbors=neighbors, dist_transform=dist_transform)

        # Triangulate input points
        try:
            self.tri = Delaunay(self.coords, incremental=False)
        except QhullError:
            # Delaunay fails in case of linearly dependent coordinates. Create artificial points in the corners of
            # given coordinate grid in order for Delaunay to work with a full rank matrix.
            min_x, min_y = np.min(self.coords, axis=0) - 1
            max_x, max_y = np.max(self.coords, axis=0) + 1
            corner_coords = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
            self.coords = np.concatenate([self.coords, corner_coords])
            if self.values is not None:
                mean_values = np.mean(self.values, axis=0, keepdims=True, dtype=self.values.dtype)
                corner_values = np.repeat(mean_values, 4, axis=0)
                self.values = np.concatenate([self.values, corner_values])
            self.tri = Delaunay(self.coords, incremental=False)

        # Perform the first auxiliary call of the tri for it to work properly in different processes.
        # Otherwise interpolation may fail if called in a pipeline with prefetch with mpc target.
        _ = self.tri.find_simplex((0, 0))

    def _is_in_hull(self, coords):
        """Check whether items in `coords` lie within the convex hull of data points."""
        coords = coords.astype(np.float32)  # Cast coords to float32 to match the type of points in the convex hull
        return np.array([cv2.pointPolygonTest(self.coords_hull, coord, measureDist=False) >= 0 for coord in coords])

    def _interpolate_inside_hull(self, coords):
        """Perform interpolation for coordinates lying inside convex hull of data points. `coords` are guaranteed to be
        2-dimensional with shape (n_coords, 2)."""
        _ = coords
        raise NotImplementedError

    def _interpolate(self, coords):
        """Perform interpolation at given `coords`. Falls back to an IDW interpolator for points lying outside the
        convex hull of data points. `coords` are guaranteed to be 2-dimensional with shape (n_coords, 2)."""
        inside_hull_mask = self._is_in_hull(coords)
        values = np.empty((len(coords), self.values.shape[1]), dtype=self.values.dtype)
        values[inside_hull_mask] = self._interpolate_inside_hull(coords[inside_hull_mask])
        # pylint: disable-next=protected-access
        values[~inside_hull_mask] = self.idw_interpolator._interpolate(coords[~inside_hull_mask])
        return values


class DelaunayInterpolator(BaseDelaunayInterpolator, ValuesAgnosticInterpolator):
    """Construct a linear barycentric interpolator.

    Interpolation at point `x` is performed in the following way:
    1. Triangulate input data points.
    2. If `x` lies within a convex hull of data points:
        2.1. Find a simplex containing `x`,
        2.2. Calculate barycentric coordinates of `x` inside the simplex,
        2.3. Aggregate values at simplex vertices with weights defined by barycentric coordinates.
    3. Otherwise fall back to an IDW interpolator.

    `DelaunayInterpolator` is values-agnostic and thus provides `get_weights` interface even without `values` passed in
    `__init__`.

    Parameters
    ----------
    coords : 2d np.ndarray with shape (n_coords, 2)
        Coordinates of data points.
    values : 1d np.ndarray with shape (n_coords,) or 2d np.ndarray with shape (n_coords, n_values), optional
        Data values at `coords`. If not given, only `get_weights` interface is provided.
    neighbors : int, optional, default to 3
        The number of closest data points to use for IDW interpolation.
    dist_transform : float or callable, optional, defaults to 2
        A function used to transform distances before smoothing, inverting and normalizing into weights during IDW
        interpolation. If `float`, defines the power to raise the distances to.

    Attributes
    ----------
    coords_hull : 3d np.ndarray
        Convex hull of `coords`.
    tri : Delaunay
        Delaunay triangulation of `coords`.
    idw_interpolator : IDWInterpolator
        An interpolator to fallback to for coordinates lying outside the `coords_hull`.
    """
    def _get_simplex_info(self, coords):
        """Return indices of simplex vertices and corresponding barycentric coordinates for each of passed `coords`."""
        simplex_ix = self.tri.find_simplex(coords)
        if np.any(simplex_ix < 0):
            raise ValueError("Some passed coords are outside convex hull of known coordinates")
        transform = self.tri.transform[simplex_ix]
        transition = transform[:, :2]
        bias = transform[:, 2]
        bar_coords = np.sum(transition * np.expand_dims(coords - bias, axis=1), axis=-1)
        bar_coords = np.column_stack([bar_coords, 1 - bar_coords.sum(axis=1)])
        return self.tri.simplices[simplex_ix], bar_coords

    def _interpolate_inside_hull(self, coords):
        """Perform linear barycentric interpolation for coordinates lying inside convex hull of data points. `coords`
        are guaranteed to be 2-dimensional with shape (n_coords, 2)."""
        simplices_indices, bar_coords = self._get_simplex_info(coords)
        return (self.values[simplices_indices] * bar_coords[:, :, None]).sum(axis=1)

    def _get_weights(self, coords):
        """Get coordinates of reference objects and their weights for each of the passed `coords`. Weights are defined
        by barycentric coordinates in the corresponding simplex for coordinates inside the convex hull of data points.
        Falls back to `self.idw_interpolator.get_weights` for coordinates outside the convex hull."""
        inside_hull_mask = self._is_in_hull(coords)
        weights = np.empty(len(coords), dtype=object)
        # pylint: disable-next=protected-access
        weights[~inside_hull_mask] = self.idw_interpolator._get_weights(coords[~inside_hull_mask])

        simplices_indices, bar_coords = self._get_simplex_info(coords[inside_hull_mask])
        simplices_coords = self.coords[simplices_indices]
        non_zero_mask = ~np.isclose(bar_coords, 0)
        weights[inside_hull_mask] = [{tuple(point): weight for point, weight in zip(simplices[mask], weights[mask])}
                                     for simplices, weights, mask in zip(simplices_coords, bar_coords, non_zero_mask)]
        return weights


class CloughTocherInterpolator(BaseDelaunayInterpolator):
    """Construct a Clough-Tocher interpolator.

    The interpolant is a piecewise cubic Bezier polynomial on each simplex of triangulation of data points. This class
    is a thin wrapper around `scipy.interpolate.CloughTocher2DInterpolator` and accepts the same arguments during
    instantiation. The only difference is that it falls back to IDW interpolator for coordinates lying outside the
    convex hull of data points to allow for extrapolation.

    Parameters
    ----------
    coords : 2d np.ndarray with shape (n_coords, 2)
        Coordinates of data points.
    values : 1d np.ndarray with shape (n_coords,) or 2d np.ndarray with shape (n_coords, n_values)
        Data values at `coords`.
    neighbors : int, optional, default to 3
        The number of closest data points to use for IDW interpolation.
    dist_transform : float or callable, optional, defaults to 2
        A function used to transform distances before smoothing, inverting and normalizing into weights during IDW
        interpolation. If `float`, defines the power to raise the distances to.
    kwargs : misc, optional
        Additional keyword arguments to pass to `CloughTocher2DInterpolator`.

    Attributes
    ----------
    coords_hull : 3d np.ndarray
        Convex hull of `coords`.
    tri : Delaunay
        Delaunay triangulation of `coords`.
    ct_interpolator : CloughTocher2DInterpolator
        An interpolator to use for coordinates lying inside the `coords_hull`.
    idw_interpolator : IDWInterpolator
        An interpolator to fallback to for coordinates lying outside the `coords_hull`.
    """
    def __init__(self, coords, values, neighbors=3, dist_transform=2, **kwargs):
        super().__init__(coords, values, neighbors=neighbors, dist_transform=dist_transform)
        self.ct_interpolator = interpolate.CloughTocher2DInterpolator(self.tri, self.values, **kwargs)

    def _interpolate_inside_hull(self, coords):
        """Perform interpolation for coordinates lying inside convex hull of data points. `coords` are guaranteed to be
        2-dimensional with shape (n_coords, 2)."""
        return self.ct_interpolator(coords)


class RBFInterpolator(SpatialInterpolator):
    """Construct a radial basis function (RBF) interpolator.

    This class is a thin wrapper around `scipy.interpolate.RBFInterpolator` and accepts the same arguments during
    instantiation.

    Parameters
    ----------
    coords : 2d np.ndarray with shape (n_coords, 2)
        Coordinates of data points.
    values : 1d np.ndarray with shape (n_coords,) or 2d np.ndarray with shape (n_coords, n_values)
        Data values at `coords`.
    neighbors : int, optional
        The number of closest data points to consider during interpolation. All the data points are used by default.
    smoothing : float, optional, defaults to 0
        Smoothing parameter. If set to 0, the interpolant perfectly fits the data. For large values, the interpolant
        approaches a least squares fit of a polynomial with the specified degree.
    kwargs : misc, optional
        Additional keyword arguments to pass to `RBFInterpolator`.

    Attributes
    ----------
    rbf_interpolator : RBFInterpolator
        An interpolator to be used.
    """
    def __init__(self, coords, values, neighbors=None, smoothing=0, **kwargs):
        super().__init__(coords, values)
        self.rbf_interpolator = interpolate.RBFInterpolator(self.coords, self.values, neighbors=neighbors,
                                                            smoothing=smoothing, **kwargs)

    def _interpolate(self, coords):
        """Perform interpolation at given `coords`. `coords` are guaranteed to be 2-dimensional with shape
        (n_coords, 2)."""
        return self.rbf_interpolator(coords).astype(self.values.dtype)
