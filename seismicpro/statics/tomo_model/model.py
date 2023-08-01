import torch
import numpy as np
from numba import njit, prange

from .profile_plot import ProfilePlot


class TomoModel:
    def __init__(self, grid, velocities):
        velocities = np.require(velocities, dtype=np.float32)
        if not np.array_equal(velocities.shape, grid.shape):
            raise ValueError

        self.grid = grid
        self.velocities_tensor = torch.from_numpy(velocities)

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

    # Model visualization

    def plot_profile(self, **kwargs):
        return ProfilePlot(self, **kwargs).plot()
