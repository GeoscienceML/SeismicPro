import math

import numpy as np
import polars as pl

from .dataset import TomoModelTravelTimeDataset
from ...const import HDR_FIRST_BREAK
from ...utils import to_list, IDWInterpolator


class Grid3D:
    def __init__(self, origin, shape, cell_size, survey=None):
        origin = np.broadcast_to(origin, 3)
        cell_size = np.broadcast_to(cell_size, 3)
        if (cell_size <= 0).any():
            raise ValueError
        shape = np.broadcast_to(shape, 3).astype(np.int32, casting="same_kind")
        if (shape <= 0).any():
            raise ValueError

        self.origin = origin
        self.cell_size = cell_size
        self.shape = shape
        self.survey = survey
        self.air_mask = None
        if survey is not None:
            # TODO: add surface elevation interpolator
            self._init_air_mask()

    @property
    def has_survey(self):
        return self.survey is not None

    def _init_air_mask(self):
        z_min, x_min, y_min = self.origin
        nz, nx, ny = self.shape
        dz, dx, dy = self.cell_size

        headers = pl.concat([sur.get_polars_headers().lazy() for sur in to_list(self.survey)], rechunk=False)
        source_elevations = headers.select([
            pl.col("SourceX").alias("X"),
            pl.col("SourceY").alias("Y"),
            pl.col("SourceSurfaceElevation").alias("Elevation")
        ])
        receiver_elevations = headers.select([
            pl.col("GroupX").alias("X"),
            pl.col("GroupY").alias("Y"),
            pl.col("ReceiverGroupElevation").alias("Elevation")
        ])
        elevations = pl.concat([source_elevations, receiver_elevations], rechunk=False).select([
            ((pl.col("X") - x_min) / dx).floor().cast(pl.Int32).alias("BinX"),
            ((pl.col("Y") - y_min) / dy).floor().cast(pl.Int32).alias("BinY"),
            pl.col("Elevation")
        ]).groupby("BinX", "BinY").agg(pl.max("Elevation"))
        max_elevations = elevations.collect().to_numpy()
        max_elevation_interp = IDWInterpolator(max_elevations[:, :2], max_elevations[:, 2], neighbors=8)
        bin_coords = np.array(np.meshgrid(np.arange(nx), np.arange(ny))).T.reshape(-1, 2)
        max_elevation_grid = max_elevation_interp(bin_coords).reshape(nx, ny)
        z_range = z_min + dz * np.arange(nz)
        self.air_mask = z_range.reshape(-1, 1, 1) >= max_elevation_grid

    # IO

    @classmethod
    def from_survey(cls, survey, z_min, cell_size, spatial_margin=3):
        survey_list = to_list(survey)
        dz, dx, dy = np.broadcast_to(cell_size, 3)

        z_max = max(sur[["SourceSurfaceElevation", "ReceiverGroupElevation"]].max() for sur in survey_list)
        if z_min >= z_max:
            raise ValueError
        nz = math.ceil((z_max - z_min) / dz)

        coords = np.concatenate([survey[["SourceX", "SourceY"]], survey[["GroupX", "GroupY"]]], axis=0)
        x_min, y_min = np.floor(coords.min(axis=0)).astype(np.int32) - spatial_margin * np.array([dx, dy])
        x_max, y_max = np.ceil(coords.max(axis=0)).astype(np.int32) + spatial_margin * np.array([dx, dy])
        nx = math.ceil((x_max - x_min) / dx)
        ny = math.ceil((y_max - y_min) / dy)

        origin = (z_min, x_min, y_min)
        shape = (nz, nx, ny)
        cell_size = (dz, dx, dy)
        return cls(origin, shape, cell_size, survey=survey)

    # Dataset generation

    def create_dataset(self, survey=None, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto"):
        if survey is None:
            if not self.has_survey:
                raise ValueError("A survey to create a dataset must be passed")
            survey = self.survey
        return TomoModelTravelTimeDataset(survey, self, first_breaks_header, uphole_correction_method)
