import math

import torch
import numpy as np
import polars as pl
from numba import njit, prange
from tqdm.auto import tqdm

from .dataloader import TensorDataLoader
from ..dataset import TravelTimeDataset
from ...const import HDR_FIRST_BREAK


class LayeredModelTravelTimeDataset(TravelTimeDataset):
    def __init__(self, survey, grid, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                 slowness_grid_step=500):
        super().__init__(survey, grid, first_breaks_header=first_breaks_header,
                         uphole_correction_method=uphole_correction_method)

        # Get indices of grid coords and their weights for each source and receiver for further interpolation of
        # layered model parameters
        self.source_indices, self.source_weights = self._get_coords_interp_params(self.source_coords, grid)
        self.receiver_indices, self.receiver_weights = self._get_coords_interp_params(self.receiver_coords, grid)

        # Get indices of grid coords, appeared either in source_indices or receiver_indices
        used_source_mask = np.bincount(self.source_indices.ravel(), minlength=self.grid.n_coords) > 0
        used_receiver_mask = np.bincount(self.receiver_indices.ravel(), minlength=self.grid.n_coords) > 0
        self.used_coords_mask = used_source_mask | used_receiver_mask
        self.used_coords_indices = np.require(np.nonzero(self.used_coords_mask)[0], dtype=np.int32)

        # Construct an interpolation grid for mean slowness estimation and get interpolation indices and weights for
        # each trace
        indices, weights = self._get_slowness_averaging_params(self.source_coords, self.receiver_coords,
                                                               grid[self.used_coords_indices], slowness_grid_step)
        self.mean_slowness_indices = self.used_coords_indices[indices]
        self.mean_slowness_weights = weights

    @staticmethod
    def _get_coords_interp_params(coords, grid):
        coords_df = pl.from_numpy(coords, schema=["X", "Y"])
        unique_coords_df = coords_df.unique()
        inverse_df = coords_df.join(unique_coords_df.with_row_count(name="i"), how="left", on=["X", "Y"])

        unique_coords = unique_coords_df.to_numpy()
        inverse = inverse_df.get_column("i").to_numpy()

        indices, weights = grid.get_interpolation_params(unique_coords)
        indices = np.require(indices, dtype=np.int32)[inverse]
        weights = np.require(weights, dtype=np.float32)[inverse]
        return indices, weights

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _get_intermediate_averaging_params(source_coords, receiver_coords, grid_size):
        min_x = min(source_coords[:, 0].min(), receiver_coords[:, 0].min())
        max_x = max(source_coords[:, 0].max(), receiver_coords[:, 0].max())
        min_y = min(source_coords[:, 1].min(), receiver_coords[:, 1].min())
        max_y = max(source_coords[:, 1].max(), receiver_coords[:, 1].max())

        grid_size_x = 1 + math.ceil((max_x - min_x) / grid_size)
        grid_size_y = 1 + math.ceil((max_y - min_y) / grid_size)
        grid_coords_x = min_x + grid_size * np.arange(grid_size_x)
        grid_coords_y = min_y + grid_size * np.arange(grid_size_y)

        coords = np.empty((grid_size_x * grid_size_y, 2))
        for i in prange(grid_size_x):
            for j in range(grid_size_y):
                coords_ix = i * grid_size_y + j
                coords[coords_ix, 0] = grid_coords_x[i]
                coords[coords_ix, 1] = grid_coords_y[j]

        n_traces = len(source_coords)
        trace_n_cells = np.empty(n_traces, dtype=np.int32)
        for i in prange(n_traces):
            offset = np.sqrt(np.sum((receiver_coords[i] - source_coords[i])**2))
            trace_n_cells[i] = 1 + math.ceil(offset / grid_size)
        max_n_cells = trace_n_cells.max()

        keep_coords_mask = np.zeros(n_traces, dtype=np.bool_)
        indices = np.zeros((n_traces, max_n_cells), dtype=np.int32)
        weights = np.zeros((n_traces, max_n_cells), dtype=np.float32)

        for i in prange(n_traces):
            n_cells = trace_n_cells[i]

            coords_x = np.linspace(source_coords[i, 0], receiver_coords[i, 0], n_cells)
            indices_x = np.round((coords_x - min_x) / grid_size).astype(np.int32)
            indices_x = np.clip(indices_x, 0, grid_size_x - 1)

            coords_y = np.linspace(source_coords[i, 1], receiver_coords[i, 1], n_cells)
            indices_y = np.round((coords_y - min_y) / grid_size).astype(np.int32)
            indices_y = np.clip(indices_y, 0, grid_size_y - 1)

            intermediate_indices = indices_x * grid_size_y + indices_y
            keep_coords_mask[intermediate_indices] = True
            indices[i, :n_cells] = intermediate_indices
            weights[i, :n_cells] = 1 / n_cells

        indices_shift = np.cumsum(~keep_coords_mask)
        for i in prange(n_traces):
            n_cells = trace_n_cells[i]
            indices[i, :n_cells] -= indices_shift[indices[i, :n_cells]]

        return coords[keep_coords_mask], indices, weights

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _combine_averaging_params(igrid_averaging_indices, igrid_to_grid_indices,
                                  igrid_averaging_weights, igrid_to_grid_weights):
        n_traces = len(igrid_averaging_indices)
        n_indices_per_trace = igrid_averaging_indices.shape[1] * igrid_to_grid_indices.shape[1]
        indices = np.empty((n_traces, n_indices_per_trace), dtype=np.int32)
        weights = np.empty((n_traces, n_indices_per_trace), dtype=np.float32)

        for i in prange(n_traces):
            trace_igrid_indices = igrid_averaging_indices[i]
            trace_igrid_weights = igrid_averaging_weights[i].reshape(-1, 1)

            indices[i] = igrid_to_grid_indices[trace_igrid_indices].ravel()
            weights[i] = (igrid_to_grid_weights[trace_igrid_indices] * trace_igrid_weights).ravel()

        return indices, weights

    @classmethod
    def _get_slowness_averaging_params(cls, source_coords, receiver_coords, grid, slowness_grid_step):
        # Construct an intermediate regular grid for faster neighbors search (igrid)
        igrid_params = cls._get_intermediate_averaging_params(source_coords, receiver_coords, slowness_grid_step)
        igrid_coords, igrid_averaging_indices, igrid_averaging_weights = igrid_params

        # Get interpolation params for each coordinate of the intermediate grid
        igrid_to_grid_indices, igrid_to_grid_weights = grid.get_interpolation_params(igrid_coords)

        # Get parameters of complex interpolation: coords -> intermediate grid -> final grid
        indices, weights = cls._combine_averaging_params(igrid_averaging_indices, igrid_to_grid_indices,
                                                         igrid_averaging_weights, igrid_to_grid_weights)
        return indices, weights

    # Loader creation

    def create_train_loader(self, batch_size, n_epochs, shuffle=True, drop_last=True, device=None, bar=True):
        train_tensors = [self.source_coords, self.source_elevations, self.source_indices, self.source_weights,
                         self.receiver_coords, self.receiver_elevations, self.receiver_indices, self.receiver_weights,
                         self.mean_slowness_indices, self.mean_slowness_weights, self.target_traveltimes]
        train_tensors = [torch.from_numpy(tensor) for tensor in train_tensors]
        loader = TensorDataLoader(*train_tensors, batch_size=batch_size, n_epochs=n_epochs,
                                  shuffle=shuffle, drop_last=drop_last, device=device)
        return tqdm(loader, desc="Iterations of model fitting", disable=not bar)

    def create_predict_loader(self, batch_size, device=None, bar=True):
        pred_tensors = [self.source_coords, self.source_elevations, self.source_indices, self.source_weights,
                        self.receiver_coords, self.receiver_elevations, self.receiver_indices, self.receiver_weights,
                        self.mean_slowness_indices, self.mean_slowness_weights, self.traveltime_corrections]
        pred_tensors = [torch.from_numpy(tensor) for tensor in pred_tensors]
        loader = TensorDataLoader(*pred_tensors, batch_size=batch_size, n_epochs=1, shuffle=False, drop_last=False,
                                  device=device)
        return tqdm(loader, desc="Iterations of model inference", disable=not bar)
