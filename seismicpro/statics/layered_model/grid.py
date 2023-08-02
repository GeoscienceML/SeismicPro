from functools import partial

import numpy as np
import polars as pl

from .dataset import LayeredModelTravelTimeDataset
from ...metrics import MetricMap
from ...const import HDR_FIRST_BREAK
from ...utils import to_list, IDWInterpolator


class SpatialGrid:
    def __init__(self, coords, surface_elevations, survey=None, n_interpolation_neighbors=1):
        coords, _ = self.process_coords(coords)
        surface_elevations = np.broadcast_to(surface_elevations, len(coords))

        self.coords = coords
        self.surface_elevations = surface_elevations
        self.n_interpolation_neighbors = n_interpolation_neighbors
        self.interpolator_class = partial(IDWInterpolator, neighbors=n_interpolation_neighbors)
        self.surface_elevation_interpolator = self.interpolator_class(coords, surface_elevations)
        self.survey = survey

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, key):
        return type(self)(self.coords[key], self.surface_elevations[key], survey=self.survey,
                          n_interpolation_neighbors=self.n_interpolation_neighbors)

    @property
    def n_coords(self):
        return len(self.coords)

    @property
    def coords_tree(self):
        return self.surface_elevation_interpolator.nearest_neighbors

    @property
    def has_survey(self):
        return self.survey is not None

    @staticmethod
    def process_coords(coords):
        coords = np.array(coords)
        is_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        if coords.ndim > 2 or coords.shape[1] != 2:
            raise ValueError
        return coords, is_1d

    # IO

    @staticmethod
    def _get_survey_sensor_locations(survey, filter_azimuths=True, n_sectors=8, max_empty_sectors=2):
        sensor_cols = ["SourceX", "SourceY", "SourceSurfaceElevation",
                       "GroupX", "GroupY", "ReceiverGroupElevation"]
        headers = pl.from_pandas(survey.get_headers(sensor_cols)).lazy()

        source_locations = (headers
            .groupby("SourceX", "SourceY")
            .agg(pl.mean("SourceSurfaceElevation"))
            .rename({"SourceX": "X", "SourceY": "Y", "SourceSurfaceElevation": "Elevation"})
        )
        receiver_locations = (headers
            .groupby("GroupX", "GroupY")
            .agg(pl.mean("ReceiverGroupElevation"))
            .rename({"GroupX": "X", "GroupY": "Y", "ReceiverGroupElevation": "Elevation"})
        )
        sensor_locations = pl.concat([source_locations, receiver_locations])
        sensor_locations = sensor_locations.groupby("X", "Y").agg(pl.col("Elevation").mean())

        if filter_azimuths and survey.has_inferred_geometry and not survey.is_2d:
            azimuth = np.arctan2(pl.col("GroupY") - pl.col("SourceY"), pl.col("GroupX") - pl.col("SourceX"))
            sector = (n_sectors * (azimuth / np.pi + 1) / 2).cast(pl.Int32).clip(0, n_sectors - 1).alias("Sector")
            sector_df = headers.with_columns(sector)
            valid_sources = (sector_df
                .groupby("SourceX", "SourceY")
                .agg(pl.n_unique("Sector"))
                .filter(pl.col("Sector") >= n_sectors - max_empty_sectors)
                .rename({"SourceX": "X", "SourceY": "Y"})
            )
            valid_receivers = (sector_df
                .groupby("GroupX", "GroupY")
                .agg(pl.n_unique("Sector"))
                .filter(pl.col("Sector") >= n_sectors - max_empty_sectors)
                .rename({"GroupX": "X", "GroupY": "Y"})
            )
            valid_locations = pl.concat([valid_sources, valid_receivers])
            sensor_locations = sensor_locations.join(valid_locations, on=["X", "Y"], how="semi")

        return sensor_locations

    @classmethod
    def _get_sensor_locations(cls, survey, **kwargs):
        sensor_locations_list = [cls._get_survey_sensor_locations(sur, **kwargs) for sur in to_list(survey)]
        if len(sensor_locations_list) == 1:
            sensor_locations = sensor_locations_list[0].collect()
        else:
            sensor_locations = pl.concat(sensor_locations_list)
            sensor_locations = sensor_locations.groupby("X", "Y").agg(pl.mean("Elevation")).collect()
        coords = sensor_locations.select("X", "Y").to_numpy()
        elevations = sensor_locations.get_column("Elevation").to_numpy()
        return coords, elevations

    @classmethod
    def _get_elevation_interpolator(cls, survey, n_interpolation_neighbors=1):
        coords, elevations = cls._get_sensor_locations(survey, filter_azimuths=False)
        return IDWInterpolator(coords, elevations, neighbors=n_interpolation_neighbors)

    @classmethod
    def from_sensors(cls, survey, filter_azimuths=False, n_interpolation_neighbors=1):
        coords, elevations = cls._get_sensor_locations(survey, filter_azimuths=filter_azimuths)
        return cls(coords, elevations, survey=survey, n_interpolation_neighbors=n_interpolation_neighbors)

    @classmethod
    def from_bins(cls, survey, n_interpolation_neighbors=1):
        coords_list = [pl.from_pandas(sur.get_headers(["CDP_X", "CDP_Y"])).lazy().unique() for sur in to_list(survey)]
        coords = coords_list[0] if len(coords_list) == 1 else pl.concat(coords_list).unique()
        coords = coords.collect().to_numpy()
        elevations = cls._get_elevation_interpolator(survey, n_interpolation_neighbors)(coords)
        return cls(coords, elevations, survey=survey, n_interpolation_neighbors=n_interpolation_neighbors)

    @classmethod
    def from_regular_grid(cls, survey, grid_size=250, min_distance_to_contour=None, n_interpolation_neighbors=4):
        survey_list = to_list(survey)
        if min_distance_to_contour is not None and any(not sur.has_inferred_geometry for sur in survey_list):
            raise ValueError("Each survey must have inferred geometry if min_distance_to_contour is given")
        grid_size = np.broadcast_to(grid_size, 2)

        bin_coords = [pl.from_pandas(sur.get_headers(["CDP_X", "CDP_Y"])).lazy() for sur in survey_list]
        coords_range = [coords.select(pl.min("CDP_X").alias("x_min"), pl.min("CDP_Y").alias("y_min"),
                                      pl.max("CDP_X").alias("x_max"), pl.max("CDP_Y").alias("y_max"))
                        for coords in bin_coords]
        coords_range = pl.concat(coords_range).select(pl.col("x_min", "y_min").min(), pl.col("x_max", "y_max").max())
        x_min, y_min, x_max, y_max = coords_range.collect().row(0)

        x_range = np.arange(x_min, x_max, grid_size[0])
        y_range = np.arange(y_min, y_max, grid_size[1])
        coords = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)

        if min_distance_to_contour is not None:
            max_dist = np.column_stack([sur.dist_to_geographic_contours(coords) for sur in survey_list]).max(axis=1)
            coords = coords[max_dist >= min_distance_to_contour]

        elevations = cls._get_elevation_interpolator(survey, n_interpolation_neighbors)(coords)
        return cls(coords, elevations, survey=survey, n_interpolation_neighbors=n_interpolation_neighbors)

    @classmethod
    def from_file(cls, path, survey=None, encoding="UTF-8"):
        pass

    def dump(self, path, encoding="UTF-8"):
        pass

    # Interpolation-related methods

    def get_weathering_velocity_interpolator(self, weathering_velocity_header, radius=None, neighbors=8,
                                             dist_transform=2, smoothing=0, min_relative_weight=1e-3):
        if not self.has_survey:
            raise ValueError("A survey must be defined for the grid")

        wv_headers = [pl.from_pandas(sur.get_headers(["SourceX", "SourceY", weathering_velocity_header])).lazy()
                      for sur in to_list(self.survey)]
        wv_headers = pl.concat(wv_headers).groupby("SourceX", "SourceY").agg(pl.mean(weathering_velocity_header))
        wv_headers = wv_headers.collect()

        source_coords = wv_headers.select("SourceX", "SourceY").to_numpy()
        source_wv = wv_headers.get_column(weathering_velocity_header).to_numpy()
        return IDWInterpolator(source_coords, source_wv, radius=radius, neighbors=neighbors,
                               dist_transform=dist_transform, smoothing=smoothing,
                               min_relative_weight=min_relative_weight)

    def get_interpolation_params(self, coords):
        # pylint: disable-next=protected-access
        return self.surface_elevation_interpolator._get_reference_indices_neighbors(coords)

    def interpolate(self, values, coords):
        coords, is_1d = (coords.coords, False) if isinstance(coords, SpatialGrid) else self.process_coords(coords)
        res = self.interpolator_class(self.coords, values)(coords)
        if is_1d:
            return res[0]
        return res

    # Dataset generation

    def create_dataset(self, survey=None, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                       slowness_grid_step=500):
        if survey is None:
            if not self.has_survey:
                raise ValueError("A survey to create a dataset must be passed")
            survey = self.survey
        return LayeredModelTravelTimeDataset(survey, self, first_breaks_header, uphole_correction_method,
                                             slowness_grid_step)

    # Grid visualization

    def plot(self):
        MetricMap(self.coords, self.surface_elevations, metric="Surface elevation").plot()
