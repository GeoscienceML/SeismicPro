import math
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from numba import njit, prange
from fteikpy import Eikonal3D

from .raytracing import describe_rays
from .profile_plot import ProfilePlot


class TomoModel:
    def __init__(self, grid, velocities):
        if not np.array_equal(velocities.shape, grid.shape):
            raise ValueError

        self.grid = grid
        self.velocities_tensor = torch.tensor(velocities, dtype=torch.float32, requires_grad=True)

    # IO

    @staticmethod
    @njit(nogil=True, parallel=True)
    def interp_velocities(elevations, velocities, z_cell_centers):
        res_velocities = np.empty((len(velocities), len(z_cell_centers)), dtype=velocities.dtype)
        for i in prange(len(velocities)):
            res_velocities[i] = velocities[i, np.searchsorted(elevations[i], z_cell_centers)]
        return res_velocities

    @classmethod
    def from_layered_model(cls, grid, layered_model):
        z_min, x_min, y_min = grid.origin
        nz, nx, ny = grid.shape
        dz, dx, dy = grid.cell_size

        x_cell_centers = x_min + dx / 2 + dx * np.arange(nx)
        y_cell_centers = y_min + dy / 2 + dy * np.arange(ny)
        z_cell_centers = z_min + dz / 2 + dz * np.arange(nz)

        model_params = np.column_stack([
            layered_model.elevations_tensor.detach().cpu().numpy()[:, ::-1],
            1000 / layered_model.slownesses_tensor.detach().cpu().numpy()[:, ::-1],
            1000 / layered_model.weathering_slowness_tensor.detach().cpu().numpy(),
        ])  # elevations and velocities should be monotonically increasing for searchsorted to properly work
        spatial_coords = np.array(np.meshgrid(x_cell_centers, y_cell_centers)).T.reshape(-1, 2)
        spatial_params = layered_model.grid.interpolate(model_params, spatial_coords)
        elevations = spatial_params[:, :layered_model.n_refractors]
        velocities = spatial_params[:, layered_model.n_refractors:]

        spatial_velocities = cls.interp_velocities(elevations, velocities, z_cell_centers)
        velocity_grid = np.require(spatial_velocities.reshape((nx, ny, nz)).transpose((2, 0, 1)),
                                   dtype=np.float32, requirements="C")
        if grid.has_survey:
            velocity_grid[grid.air_mask] = 330
        return cls(grid, velocity_grid)

    @classmethod
    def from_gradient_model(cls, grid, top_velocity, bottom_velocity):
        if (top_velocity <= 0) or (bottom_velocity <= 0):
            raise ValueError
        velocities = np.linspace(bottom_velocity, top_velocity, grid.shape[0]).reshape(-1, 1, 1)
        return cls(grid, np.broadcast_to(velocities, grid.shape))

    # Traveltime estimation

    @staticmethod
    def crop_model(source, receivers, velocities, origin, cell_size, spatial_margin=3):
        min_coords = np.minimum(source[1:], receivers[:, 1:].min(axis=0))
        ix_min, iy_min = (min_coords - origin[1:]) / cell_size[1:]
        ix_min = max(math.floor(ix_min) - spatial_margin, 0)
        iy_min = max(math.floor(iy_min) - spatial_margin, 0)

        max_coords = np.maximum(source[1:], receivers[:, 1:].max(axis=0))
        ix_max, iy_max = (max_coords - origin[1:]) / cell_size[1:]
        ix_max = math.floor(ix_max) + spatial_margin
        iy_max = math.floor(iy_max) + spatial_margin

        cropped_origin = (origin[0], origin[1] + ix_min * cell_size[1], origin[2] + iy_min * cell_size[2])
        return Eikonal3D(velocities[:, ix_min:ix_max+1, iy_min:iy_max+1], cell_size, cropped_origin)

    def describe_rays(self, source, receivers, spatial_margin=3, n_sweeps=2, max_n_steps=None):
        velocities = self.velocities_tensor.detach().numpy()
        origin = np.require(self.grid.origin, dtype=np.float64)
        cell_size = np.require(self.grid.cell_size, dtype=np.float64)

        cropped_grid = self.crop_model(source, receivers, velocities, origin, cell_size, spatial_margin=spatial_margin)
        tt_grid = cropped_grid.solve(source, nsweep=n_sweeps, return_gradient=True)
        z_grad, x_grad, y_grad = tt_grid.gradient

        if max_n_steps is None:
            nz, nx, ny = tt_grid.shape
            max_n_steps = 2 * nz + nx + ny

        ray_params = describe_rays(source, receivers, velocities, origin, cell_size, z_grad.grid, x_grad.grid,
                                   y_grad.grid, tt_grid.zaxis, tt_grid.xaxis, tt_grid.yaxis, max_n_steps=max_n_steps)
        return tt_grid, *ray_params

    @torch.no_grad()
    def estimate_traveltimes(self, source, receivers, spatial_margin=3, n_sweeps=2, max_n_steps=None):
        ray_params = self.describe_rays(source, receivers, spatial_margin=spatial_margin, n_sweeps=n_sweeps,
                                        max_n_steps=max_n_steps)
        tt_grid, _, _, succeeded, trace_indices, cell_indices, cell_passes = ray_params

        n_succeeded = succeeded.sum()
        tt_pred = np.empty(len(receivers), dtype=np.float32)
        if n_succeeded != len(receivers):
            tt_pred[~succeeded] = tt_grid(receivers[~succeeded])

        tt_pred_tensor = torch.zeros(n_succeeded, dtype=torch.float64)
        trace_indices = torch.from_numpy(trace_indices)
        cell_indices = torch.from_numpy(cell_indices)
        cell_passes = torch.from_numpy(cell_passes)

        cell_velocities = torch.index_select(self.velocities_tensor.ravel(), 0, cell_indices)
        tt_pred_tensor.scatter_add_(0, trace_indices, 1000 * cell_passes / cell_velocities)
        tt_pred[succeeded] = tt_pred_tensor.detach().numpy()
        return tt_pred

    def estimate_traveltimes_batch(self, batch, spatial_margin=3, n_sweeps=2, max_n_steps=None):
        futures = []
        n_workers = min(80, len(batch))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for source, receivers in batch:
                future = pool.submit(self.estimate_traveltimes, source, receivers, spatial_margin=spatial_margin,
                                     n_sweeps=n_sweeps, max_n_steps=max_n_steps)
                futures.append(future)
        return [future.result() for future in futures]

    # Model visualization

    def plot_profile(self, **kwargs):
        return ProfilePlot(self, **kwargs).plot()
