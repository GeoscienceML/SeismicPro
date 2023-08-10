import warnings

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .profile_plot import ProfilePlot
from ..statics import Statics
from ..utils import get_uphole_correction_method
from ...survey import Survey
from ...utils import to_list, align_args, IDWInterpolator
from ...const import HDR_FIRST_BREAK


class LayeredModel:
    def __init__(self, grid, velocities, elevations=None, thicknesses=None, device="cpu"):
        self.grid = grid

        # Process velocities and validate them
        velocities = self.broadcast_to_grid(velocities)
        if (np.diff(velocities, axis=1) < 0).any():
            raise ValueError("Velocities must increase in depth")
        if (velocities <= 0).any():
            raise ValueError("Layer velocities must be positive")
        slownesses = 1000 / velocities

        # Process elevations or convert thicknesses to elevations
        if elevations is not None and thicknesses is not None:
            raise ValueError("Either elevations or thicknesses should be passed")
        if elevations is None and thicknesses is None:
            elevations = np.empty((self.n_coords, 0))
        elif elevations is not None:
            elevations = self.broadcast_to_grid(elevations)
        else:
            thicknesses = self.broadcast_to_grid(thicknesses)
            depths = np.cumsum(thicknesses, axis=1)
            elevations = grid.surface_elevations.reshape(-1, 1) - depths

        if (elevations > grid.surface_elevations.reshape(-1, 1)).any():
            raise ValueError
        if (np.diff(elevations, axis=1) > 0).any():
            raise ValueError

        if slownesses.shape[1] != elevations.shape[1] + 1:
            raise ValueError

        # Convert model parameters to torch tensors and enforce model constraints
        common_kwargs = {"dtype": torch.float32, "device": device}
        self.device = device
        self.weathering_slowness_tensor = torch.tensor(slownesses[:, 0], requires_grad=True, **common_kwargs)
        self.slownesses_tensor = torch.tensor(slownesses[:, 1:], requires_grad=True, **common_kwargs)
        self.surface_elevation_tensor = torch.tensor(grid.surface_elevations, **common_kwargs)
        self.elevations_tensor = torch.tensor(elevations, requires_grad=True, **common_kwargs)
        self.enforce_constraints()

        # Define default optimization-related attributes
        self.optimizer = torch.optim.Adam([
            {"params": self.weathering_slowness_tensor, "lr": 0.1},
            {"params": self.slownesses_tensor, "lr": 0.001},
            {"params": self.elevations_tensor, "lr": 1},
        ])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, threshold=0.01, patience=25)

        # Define history-related attributes
        self.loss_hist = []
        self.velocities_reg_hist = []
        self.elevations_reg_hist = []
        self.thicknesses_reg_hist = []

    @property
    def coords(self):
        return self.grid.coords

    @property
    def n_coords(self):
        return self.grid.n_coords

    @property
    def coords_tree(self):
        return self.grid.coords_tree

    @property
    def n_refractors(self):
        return self.slownesses_tensor.shape[1]

    @property
    def n_layers(self):
        return self.n_refractors + 1

    def broadcast_to_grid(self, arr):
        arr = np.atleast_1d(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Arrays must be 2-dimensional at most")
        return np.broadcast_to(arr, (self.n_coords, arr.shape[1]))

    def process_coords(self, coords):
        coords = np.array(coords)
        is_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        if coords.ndim > 2 or coords.shape[1] not in {2, 3}:
            raise ValueError
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, self.grid.surface_elevation_interpolator(coords)])
        return coords, is_1d

    # IO

    @classmethod
    def _init_model(cls, grid, rvf_params, init_weathering_velocity=None):
        rvf_params = np.atleast_2d(rvf_params)
        rvf_params = np.broadcast_to(rvf_params, (grid.n_coords, rvf_params.shape[1]))
        n_refractors = rvf_params.shape[1] // 2

        # Initialize weathering velocity and validate layer velocities for correctness
        velocities = rvf_params[:, n_refractors:]
        if init_weathering_velocity is None:
            weathering_velocity = velocities[:, 0] / 2
        elif isinstance(init_weathering_velocity, str):
            weathering_velocity = grid.get_weathering_velocity_interpolator(init_weathering_velocity)(grid.coords)
        elif callable(init_weathering_velocity):
            weathering_velocity = init_weathering_velocity(grid.coords)
        else:
            weathering_velocity = init_weathering_velocity
        weathering_velocity = np.minimum(weathering_velocity, velocities[:, 0])
        velocities = np.column_stack([weathering_velocity, velocities])
        if (np.diff(velocities, axis=1) < 0).any():
            raise ValueError("Velocities must increase in depth")
        if (velocities <= 0).any():
            raise ValueError("Layer velocities must be positive")

        # Estimate initial layer thicknesses and convert them to layer elevations
        slownesses = 1000 / velocities
        intercept_deltas = np.cumsum(-rvf_params[:, 1:n_refractors:] * np.diff(slownesses[:, 1:], axis=1), axis=1)
        intercepts = np.column_stack([rvf_params[:, 0], rvf_params[:, :1] + intercept_deltas])
        thicknesses = []
        for i in range(n_refractors):
            prev_delay = sum(thicknesses[j] * np.sqrt(slownesses[:, j]**2 - slownesses[:, i]**2)
                             for j in range(i))
            slowness_contrast = np.maximum(np.sqrt(slownesses[:, i]**2 - slownesses[:, i + 1]**2), 0.01)
            thicknesses.append(np.maximum((intercepts[:, i] / 2 - prev_delay) / slowness_contrast, 0.01))
        thicknesses = np.column_stack(thicknesses)
        elevations = grid.surface_elevations.reshape(-1, 1) - thicknesses.cumsum(axis=1)
        return velocities, elevations

    @classmethod
    def from_refractor_velocity(cls, grid, refractor_velocity, init_weathering_velocity=None, device="cpu"):
        rvf_params = list(refractor_velocity.params.values())
        velocities, elevations = cls._init_model(grid, rvf_params, init_weathering_velocity=init_weathering_velocity)
        return cls(grid, velocities, elevations, device=device)

    @classmethod
    def from_refractor_velocity_field(cls, grid, refractor_velocity_field, init_weathering_velocity=None,
                                      device="cpu"):
        rvf_params = refractor_velocity_field.interpolate(grid.coords, is_geographic=True)
        velocities, elevations = cls._init_model(grid, rvf_params, init_weathering_velocity=init_weathering_velocity)
        return cls(grid, velocities, elevations, device=device)

#     @classmethod
#     def from_file(cls, path, device="cpu", encoding="UTF-8"):
#         params_df = load_dataframe(path, has_header=True, encoding=encoding)
#         n_layers = (len(params_df.columns) - 2) // 2
#         coords_cols = ["x", "y"]
#         elevation_cols = [f"e{i}" for i in range(n_layers)]
#         velocity_cols = [f"v{i}" for i in range(n_layers)]
#         expected_cols = coords_cols + elevation_cols + velocity_cols
#         if set(expected_cols) != set(params_df.columns):
#             raise ValueError

#         coords = params_df[coords_cols].to_numpy()
#         elevations = params_df[elevation_cols].to_numpy()
#         velocities = params_df[velocity_cols].to_numpy()
#         nsm = cls(coords, elevations, velocities, device=device)
#         return nsm

#     def dump(self, path, encoding="UTF-8"):
#         coords_cols = ["x", "y"]
#         elevation_cols = [f"e{i}" for i in range(self.n_layers)]
#         velocity_cols = [f"v{i}" for i in range(self.n_layers)]
#         params_df = pd.DataFrame(self.coords, columns=coords_cols)
#         params_df[elevation_cols[0]] = self.surface_elevation_tensor.detach().cpu().numpy()
#         params_df[elevation_cols[1:]] = self.elevations_tensor.detach().cpu().numpy()
#         params_df[velocity_cols[0]] = 1000 / self.weathering_slowness_tensor.detach().cpu().numpy()
#         params_df[velocity_cols[1:]] = 1000 / self.slownesses_tensor.detach().cpu().numpy()
#         dump_dataframe(params_df, path, has_header=True, encoding=encoding)

    # Interpolation-related methods

    def change_grid(self, grid, device="cpu"):
        self_slownesses = np.column_stack([self.weathering_slowness_tensor.detach().cpu().numpy(),
                                           self.slownesses_tensor.detach().cpu().numpy()])
        velocities = self.grid.interpolate(1000 / self_slownesses, grid)
        elevations = self.grid.interpolate(self.elevations_tensor.detach().cpu().numpy(), grid)
        return type(self)(grid, velocities=velocities, elevations=elevations, device=device)

    # Traveltime estimation

    @staticmethod
    def _describe_incident_rays(sensor_elevations, layer_slownesses, layer_elevations, layer_dips_cos, layer_dips_tan):
        batch_size = len(sensor_elevations)
        n_refractors = layer_elevations.shape[1]
        device = sensor_elevations.device

        # Calculate parameters of angles of incidence: param[:, i, j] contains a value of a given incidence parameter
        # at the border between j-th and (j+1)-th layer if critical refraction occurred at the border between i-th and
        # (i+1)-th layer. If i < j, param values are calculated as well but unused later.
        incidence_sin = torch.clip(layer_slownesses[:, 1:, None] / layer_slownesses[:, None, :-1], max=0.999)
        incidence_cos = torch.sqrt(1 - incidence_sin**2)
        incidence_tan = incidence_sin / incidence_cos

        # Calculate coefficients for ray correction to account for dipping refractors
        horizontal_correction_coef = layer_dips_cos[:, None] * (incidence_tan + layer_dips_tan[:, None])
        vertical_correction_coef = -torch.diff(layer_dips_tan, axis=1)[:, :, None]

        # Infer which layer each sensor belongs to and calculate vertical distances to all layers below
        dist_to_layers = sensor_elevations.reshape(-1, 1) - layer_elevations  # (bs, n_ref)
        layer_above_mask = dist_to_layers < 0
        dist_to_layers[layer_above_mask] = 0
        sensor_layers = layer_above_mask.sum(axis=1)  # (bs,)

        # Calculate normal passes through each layer: normal_dist[:, i, j] contains a normal distance path through
        # layer j if critical refraction occurred at the border between layers i and i+1.
        vertical_pass_dist = torch.diff(dist_to_layers, prepend=torch.zeros_like(dist_to_layers[:, :1]), axis=1)
        first_layer_normal_dist = vertical_pass_dist[:, 0, None] * layer_dips_cos[:, 0, None]
        normal_dist_list = [first_layer_normal_dist.broadcast_to(batch_size, n_refractors)]
        horizontal_correction_dist = 0  # (bs, n_ref), critical refraction index over the last axis
        for i in range(0, n_refractors - 1):
            horizontal_correction_delta = normal_dist_list[i] * horizontal_correction_coef[:, :, i]
            horizontal_correction_dist = horizontal_correction_dist + horizontal_correction_delta
            vertical_correction_delta = horizontal_correction_dist * vertical_correction_coef[:, i]
            corrected_vertical_pass_dist = vertical_pass_dist[:, i + 1, None] + vertical_correction_delta
            normal_dist = corrected_vertical_pass_dist * layer_dips_cos[:, i + 1, None]
            normal_dist_list.append(normal_dist)
        normal_dist = torch.stack(normal_dist_list, axis=-1)

        # Zero out norm[:, i, j] if i < j
        arange = torch.arange(n_refractors, device=device)
        zero_mask = torch.broadcast_to(arange.reshape(-1, 1) < arange, normal_dist.shape)
        normal_dist[zero_mask] = 0

        # Calculate passes along each previous refractor and the total incidence time for each final refractor
        incidence_times = (normal_dist / incidence_cos * layer_slownesses[:, None, :-1]).sum(axis=-1)
        paths_along_refractors = normal_dist * incidence_tan
        return sensor_layers, incidence_times, paths_along_refractors

    @staticmethod
    def _estimate_vertical_traveltimes(src_elevations, dst_elevations, layer_slownesses, layer_elevations):
        high_elevations = torch.maximum(src_elevations, dst_elevations)
        low_elevations = torch.minimum(src_elevations, dst_elevations)
        total_pass_dist = high_elevations - low_elevations

        dist_to_layers = high_elevations.reshape(-1, 1) - layer_elevations
        dist_to_layers[dist_to_layers < 0] = 0
        vertical_pass_dist = torch.diff(dist_to_layers, prepend=torch.zeros_like(layer_elevations[:, :1]), axis=1)
        vertical_pass_dist = torch.column_stack([vertical_pass_dist, total_pass_dist])

        overflow_mask = vertical_pass_dist.cumsum(axis=1) > total_pass_dist.reshape(-1, 1)
        vertical_pass_dist[overflow_mask] = 0
        residual_pass_dist = total_pass_dist - vertical_pass_dist.sum(axis=1)
        overflow_ix = overflow_mask.max(axis=1)[1]
        vertical_pass_dist[torch.arange(len(src_elevations)), overflow_ix] = residual_pass_dist

        traveltimes = (vertical_pass_dist * layer_slownesses).sum(axis=1)
        return traveltimes

    @classmethod
    def _project_to_target_layers(cls, sensor_elevations, sensor_layers, target_layers,
                                  layer_slownesses, layer_elevations):
        projected_layer_ix = torch.where(target_layers > sensor_layers, target_layers - 1, target_layers)
        projected_layer_ix = torch.clip(projected_layer_ix, max=layer_elevations.shape[1] - 1).reshape(-1, 1)
        projected_elevations = torch.gather(layer_elevations, index=projected_layer_ix, axis=1).reshape(-1)
        projected_elevations = torch.where(sensor_layers == target_layers, sensor_elevations, projected_elevations)
        projection_times = cls._estimate_vertical_traveltimes(sensor_elevations, projected_elevations,
                                                              layer_slownesses, layer_elevations)
        return projected_elevations, projection_times

    @classmethod
    def _estimate_direct_traveltimes(cls, source_layers, source_elevations, source_layer_slownesses,
                                     source_layer_elevations, receiver_layers, receiver_elevations,
                                     receiver_layer_slownesses, receiver_layer_elevations, offsets,
                                     mean_layer_slownesses):
        # Project each source and receiver to the deepest layer for each trace
        max_layers = torch.maximum(source_layers, receiver_layers)
        source_projections = cls._project_to_target_layers(source_elevations, source_layers, max_layers,
                                                           source_layer_slownesses, source_layer_elevations)
        projected_source_elevations, projection_source_times = source_projections
        receiver_projections = cls._project_to_target_layers(receiver_elevations, receiver_layers, max_layers,
                                                             receiver_layer_slownesses, receiver_layer_elevations)
        projected_receiver_elevations, projection_receiver_times = receiver_projections

        # Calculate distances passed by direct waves in the deepest layer and mean wave slownesses
        squared_direct_lens = offsets**2 + (projected_receiver_elevations - projected_source_elevations)**2
        direct_lens = torch.sqrt(torch.clip(squared_direct_lens, min=0.01))
        direct_slownesses = torch.gather(mean_layer_slownesses, index=max_layers.reshape(-1, 1), axis=1).reshape(-1)

        # Calculate an approximation of direct traveltimes as a sum of direct traveltimes in the deepest layer and
        # total vertical correction due to source/receiver projection. This approximation is exact if and only if
        # both source and receiver belong to the same layer.
        vertical_traveltimes = projection_source_times + projection_receiver_times
        direct_traveltimes = direct_lens * direct_slownesses + vertical_traveltimes
        return direct_traveltimes

    @staticmethod
    def _weight_tensor(tensor, indices, weights):
        # index_select is much faster than advanced indexing during backward
        indexed_shape = indices.shape + tensor.shape[1:]
        indexed = tensor.index_select(0, indices.ravel()).reshape(indexed_shape)
        weights_shape = weights.shape + (1,) * (tensor.ndim - 1)
        return (indexed * weights.reshape(weights_shape)).sum(axis=1)

    def _interpolate_layer_slownesses(self, indices, weights):
        return torch.column_stack([self._weight_tensor(self.weathering_slowness_tensor, indices, weights),
                                   self._weight_tensor(self.slownesses_tensor, indices, weights)])

    def _interpolate_layer_elevations(self, indices, weights):
        return self._weight_tensor(self.elevations_tensor, indices, weights)

    def _estimate_traveltimes(self, source_coords, source_elevations, source_indices, source_weights,
                              receiver_coords, receiver_elevations, receiver_indices, receiver_weights,
                              mean_slowness_indices, mean_slowness_weights):
        # Calculate an offset and mean slowness of each layer for each trace
        offsets = torch.sqrt(torch.sum((source_coords - receiver_coords)**2, axis=1))
        mean_layer_slownesses = self._interpolate_layer_slownesses(mean_slowness_indices, mean_slowness_weights)

        # Return direct traveltimes in case of a single-layer model
        if self.n_layers == 1:
            squared_direct_lens = offsets**2 + (receiver_elevations - source_elevations)**2
            direct_lens = torch.sqrt(torch.clip(squared_direct_lens, min=0.01))  # Clip to avoid nans during backward
            return direct_lens * mean_layer_slownesses[:, 0]

        # Interpolate layer slownesses and elevations at source and receiver locations
        source_layer_slownesses = self._interpolate_layer_slownesses(source_indices, source_weights)
        source_layer_elevations = self._interpolate_layer_elevations(source_indices, source_weights)
        receiver_layer_slownesses = self._interpolate_layer_slownesses(receiver_indices, receiver_weights)
        receiver_layer_elevations = self._interpolate_layer_elevations(receiver_indices, receiver_weights)

        # Estimate refractor dips
        layer_elevations_diff = receiver_layer_elevations - source_layer_elevations
        layer_dips_tan = layer_elevations_diff / torch.clip(offsets, min=0.01).reshape(-1, 1)
        layer_dips_cos = torch.sqrt(1 / (1 + layer_dips_tan**2))
        layer_dips_sin = layer_dips_cos * layer_dips_tan
        layer_dips_cos_diff = (layer_dips_cos[:, 1:] * layer_dips_cos[:, :-1] +
                               layer_dips_sin[:, 1:] * layer_dips_sin[:, :-1])
        layer_dips_cos_diff = torch.column_stack([layer_dips_cos[:, 0], layer_dips_cos_diff])

        # Calculate parameters of incident rays from sources to a given refractor and back from the refractor to the
        # corresponding receiver
        source_ray_params = self._describe_incident_rays(source_elevations, source_layer_slownesses,
                                                         source_layer_elevations, layer_dips_cos, layer_dips_tan)
        source_layers, source_incidence_times, source_paths_along_refractors = source_ray_params
        receiver_ray_params = self._describe_incident_rays(receiver_elevations, receiver_layer_slownesses,
                                                           receiver_layer_elevations, layer_dips_cos, -layer_dips_tan)
        receiver_layers, receiver_incidence_times, receiver_paths_along_refractors = receiver_ray_params

        # Calculate traveltimes of direct waves
        args = (source_layers, source_elevations, source_layer_slownesses, source_layer_elevations,
                receiver_layers, receiver_elevations, receiver_layer_slownesses, receiver_layer_elevations,
                offsets, mean_layer_slownesses)
        direct_traveltimes = self._estimate_direct_traveltimes(*args)

        # Calculate traveltimes of waves refracted from each layer for each trace
        residual_paths_along_refractors = []
        sensor_paths_along_refractors = source_paths_along_refractors + receiver_paths_along_refractors
        current_paths = offsets.reshape(-1, 1)
        for i in range(self.n_refractors):
            current_paths = current_paths * layer_dips_cos_diff[i] - sensor_paths_along_refractors[:, :, i]
            residual_paths_along_refractors.append(current_paths[:, i])
        residual_paths_along_refractors = torch.column_stack(residual_paths_along_refractors)  # (bs, n_ref)
        traveltimes_along_refractors = residual_paths_along_refractors * mean_layer_slownesses[:, 1:]
        refracted_traveltimes = traveltimes_along_refractors + source_incidence_times + receiver_incidence_times

        # Ignore impossible refractions
        min_valid_refractor = torch.maximum(source_layers, receiver_layers).reshape(-1, 1)
        shallow_refractor_mask = min_valid_refractor > torch.arange(self.n_refractors, device=self.device)
        negative_path_mask = residual_paths_along_refractors < 0
        ignore_mask = shallow_refractor_mask | negative_path_mask
        undefined_traveltime = torch.maximum(refracted_traveltimes.max(), direct_traveltimes.max()) + 1
        refracted_traveltimes = torch.where(ignore_mask, undefined_traveltime, refracted_traveltimes)

        # Return minimum feasible traveltime for each trace
        traveltimes = torch.column_stack([direct_traveltimes, refracted_traveltimes])
        return traveltimes.min(axis=1)[0]

    # Dataset generation

    def create_dataset(self, survey=None, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                       slowness_grid_step=500):
        return self.grid.create_dataset(survey, first_breaks_header, uphole_correction_method, slowness_grid_step)

    # Model fitting and inference

    @torch.no_grad()
    def enforce_constraints(self):
        self.weathering_slowness_tensor.clip_(min=0.01)
        if self.n_refractors == 0:
            return

        self.elevations_tensor.clip_(max=self.surface_elevation_tensor.reshape(-1, 1))
        if self.n_refractors > 1:
            self.elevations_tensor.data = torch.cummin(self.elevations_tensor, axis=1)[0]

        self.slownesses_tensor.clip_(min=0.01)
        if self.n_refractors > 1:
            self.slownesses_tensor.data = torch.cummin(self.slownesses_tensor, axis=1)[0]

        self.weathering_slowness_tensor.clip_(min=self.slownesses_tensor[:, 0])

    def prepare_regularization_tensors(self, dataset, n_reg_neighbors=32,  velocities_reg_coef=1,
                                       elevations_reg_coef=0.5, thicknesses_reg_coef=0.5):
        if n_reg_neighbors is None or n_reg_neighbors == 0:
            return None, None, None, None, None

        velocities_reg_coef = torch.tensor(velocities_reg_coef, dtype=torch.float32, device=self.device)
        velocities_reg_coef = torch.broadcast_to(velocities_reg_coef, (self.n_layers,))
        elevations_reg_coef = torch.tensor(elevations_reg_coef, dtype=torch.float32, device=self.device)
        elevations_reg_coef = torch.broadcast_to(elevations_reg_coef, (self.n_refractors,))
        thicknesses_reg_coef = torch.tensor(thicknesses_reg_coef, dtype=torch.float32, device=self.device)
        thicknesses_reg_coef = torch.broadcast_to(thicknesses_reg_coef, (self.n_refractors,))

        idw = IDWInterpolator(self.coords[dataset.used_coords_indices], neighbors=n_reg_neighbors + 1)
        neighbors_dist, neighbors_indices = idw.nearest_neighbors.query(self.coords, k=idw.neighbors[1:], workers=-1)
        neighbors_indices = dataset.used_coords_indices[neighbors_indices]
        neighbors_weights = idw._distances_to_weights(neighbors_dist)  # pylint: disable=protected-access
        neighbors_indices = torch.tensor(neighbors_indices, dtype=torch.int32, device=self.device)
        neighbors_weights = torch.tensor(neighbors_weights, dtype=torch.float32, device=self.device)

        return neighbors_indices, neighbors_weights, velocities_reg_coef, elevations_reg_coef, thicknesses_reg_coef

    def calculate_regularizer(self, source_indices, source_weights, receiver_indices, receiver_weights,
                              neighbors_indices, neighbors_weights, velocities_reg_coef, elevations_reg_coef,
                              thicknesses_reg_coef):
        if neighbors_indices is None:
            velocities_reg = torch.tensor(0, dtype=torch.float32, device=self.device)
            elevations_reg = torch.tensor(0, dtype=torch.float32, device=self.device)
            thicknesses_reg = torch.tensor(0, dtype=torch.float32, device=self.device)
            return velocities_reg, elevations_reg, thicknesses_reg

        valid_source_indices = source_indices[source_weights > 0]
        valid_source_mask = torch.bincount(valid_source_indices, minlength=self.n_coords) > 0
        valid_receiver_indices = receiver_indices[receiver_weights > 0]
        valid_receiver_mask = torch.bincount(valid_receiver_indices, minlength=self.n_coords) > 0
        valid_sensor_mask = valid_source_mask | valid_receiver_mask
        neighbors_indices = neighbors_indices[valid_sensor_mask]
        neighbors_weights = neighbors_weights[valid_sensor_mask][..., None]

        velocities = 1000 / torch.column_stack([self.weathering_slowness_tensor, self.slownesses_tensor])
        sensor_velocities = velocities[valid_sensor_mask]
        interp_velocities = (neighbors_weights * velocities[neighbors_indices]).sum(axis=1)
        velocities_err = torch.abs(sensor_velocities - interp_velocities) / sensor_velocities
        velocities_reg = (velocities_err * velocities_reg_coef).mean()

        sensor_elevations = self.elevations_tensor[valid_sensor_mask]
        interp_elevations = (neighbors_weights * self.elevations_tensor[neighbors_indices]).sum(axis=1)
        elevations_err = torch.abs(sensor_elevations - interp_elevations)
        elevations_reg = (elevations_err * elevations_reg_coef).mean()

        thicknesses = -torch.diff(self.elevations_tensor, prepend=self.surface_elevation_tensor.reshape(-1, 1), axis=1)
        sensor_thicknesses = thicknesses[valid_sensor_mask]
        interp_thicknesses = (neighbors_weights * thicknesses[neighbors_indices]).sum(axis=1)
        thicknesses_err = torch.abs(sensor_thicknesses - interp_thicknesses)
        thicknesses_reg = (thicknesses_err * thicknesses_reg_coef).mean()

        return velocities_reg, elevations_reg, thicknesses_reg

    def _interpolate_tensor(self, tensor, used_coords_grid, unused_coords_grid, used_coords_mask):
        used_tensor_np = tensor[used_coords_mask].detach().cpu().numpy()
        unused_tensor_np = used_coords_grid.interpolate(used_tensor_np, unused_coords_grid)
        tensor.data[~used_coords_mask] = torch.tensor(unused_tensor_np, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def interpolate_unused_points(self, dataset):
        used_coords_mask = dataset.used_coords_mask
        if used_coords_mask.all():
            return

        used_coords_grid = self.grid[used_coords_mask]
        unused_coords_grid = self.grid[~used_coords_mask]
        self._interpolate_tensor(self.weathering_slowness_tensor, used_coords_grid, unused_coords_grid,
                                 used_coords_mask)
        self._interpolate_tensor(self.slownesses_tensor, used_coords_grid, unused_coords_grid, used_coords_mask)
        self._interpolate_tensor(self.elevations_tensor, used_coords_grid, unused_coords_grid, used_coords_mask)

    def fit(self, dataset, batch_size=250000, n_epochs=5, n_reg_neighbors=32, velocities_reg_coef=1,
            elevations_reg_coef=0.5, thicknesses_reg_coef=0.5, bar=True):
        reg_tensors = self.prepare_regularization_tensors(dataset, n_reg_neighbors, velocities_reg_coef,
                                                          elevations_reg_coef, thicknesses_reg_coef)
        loader = dataset.create_train_loader(batch_size=batch_size, n_epochs=n_epochs, shuffle=True, drop_last=True,
                                             device=self.device, bar=bar)
        for *params, target_traveltimes in loader:
            prediction_traveltimes = self._estimate_traveltimes(*params)
            loss = torch.abs(prediction_traveltimes - target_traveltimes).mean()

            # Calculate regularization term
            source_indices = params[2]
            source_weights = params[3]
            receiver_indices = params[6]
            receiver_weights = params[7]
            sensor_tensors = (source_indices, source_weights, receiver_indices, receiver_weights)
            velocities_reg, elevations_reg, thicknesses_reg = self.calculate_regularizer(*sensor_tensors, *reg_tensors)

            total_loss = loss + elevations_reg + thicknesses_reg + velocities_reg
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss.item())
            self.enforce_constraints()

            self.loss_hist.append(loss.item())
            self.velocities_reg_hist.append(velocities_reg.item())
            self.elevations_reg_hist.append(elevations_reg.item())
            self.thicknesses_reg_hist.append(thicknesses_reg.item())

        self.interpolate_unused_points(dataset)

    def predict(self, dataset, batch_size=1000000, bar=True, predicted_first_breaks_header=None):
        loader = dataset.create_predict_loader(batch_size=batch_size, device=self.device, bar=bar)
        with torch.no_grad():
            pred_traveltimes = [(self._estimate_traveltimes(*params) - traveltime_corrections).cpu()
                                for *params, traveltime_corrections in loader]
        pred_traveltimes = torch.cat(pred_traveltimes).numpy()
        dataset.pred_traveltimes = np.maximum(pred_traveltimes, 0)

        if predicted_first_breaks_header is not None:
            dataset.store_predictions_to_survey(predicted_first_breaks_header)
        return dataset

    # Statics calculation

    @torch.no_grad()
    def estimate_statics(self, coords, intermediate_datum=None, intermediate_datum_refractor=None, final_datum=None,
                         replacement_velocity=None):
        # Interpolate layer elevations and thicknesses at given coords
        coords, is_1d = self.process_coords(coords)
        indices, weights = self.grid.get_interpolation_params(coords[:, :2])
        elevations = torch.tensor(coords[:, -1], dtype=torch.float32, device=self.device)
        indices = torch.tensor(indices, dtype=torch.int32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        layer_slownesses = self._interpolate_layer_slownesses(indices, weights)
        layer_elevations = self._interpolate_layer_elevations(indices, weights)

        # Get elevation of intermediate datum for each passed coordinate
        if intermediate_datum is None:
            if intermediate_datum_refractor is not None:
                intermediate_elevations = layer_elevations[:, intermediate_datum_refractor - 1]
                if replacement_velocity is None:
                    replacement_velocity = 1000 / layer_slownesses[:, intermediate_datum_refractor - 1]
            elif final_datum is not None:
                intermediate_elevations = torch.tensor(final_datum, dtype=torch.float32, device=self.device)
                final_datum = None
            else:
                raise ValueError
        else:
            if intermediate_datum_refractor is not None:
                raise ValueError
            intermediate_elevations = torch.tensor(intermediate_datum, dtype=torch.float32, device=self.device)
        intermediate_elevations = torch.broadcast_to(intermediate_elevations, (len(coords),))

        # Calculate statics from sensor locations to intermediate datum
        sign = torch.sign(elevations - intermediate_elevations)
        statics = sign * self._estimate_vertical_traveltimes(elevations, intermediate_elevations,
                                                             layer_slownesses, layer_elevations)

        # Add statics from intermediate to final datum
        if final_datum is not None:
            if replacement_velocity is None:
                raise ValueError
            statics += 1000 * (intermediate_elevations - final_datum) / replacement_velocity

        statics = statics.detach().cpu().numpy()
        if is_1d:
            return statics[0]
        return statics

    def _get_source_statics(self, survey, index_cols, uphole_correction_method, **kwargs):
        index_cols = to_list(index_cols)
        all_cols_set = set(index_cols + ["SourceX", "SourceY", "SourceSurfaceElevation"])
        if "SourceDepth" in survey.available_headers:
            all_cols_set.add("SourceDepth")
        if "SourceUpholeTime" in survey.available_headers:
            all_cols_set.add("SourceUpholeTime")
        all_cols = list(all_cols_set)
        non_index_cols = list(all_cols_set - set(index_cols))

        statics = pl.from_pandas(survey.get_headers(all_cols), rechunk=False)
        is_duplicated_expr = pl.all_horizontal([pl.n_unique(col) == 1 for col in non_index_cols]).alias("IsUnique")
        statics = statics.groupby(index_cols).agg(pl.mean(non_index_cols), is_duplicated_expr).to_pandas()

        if not statics["IsUnique"].all():
            warnings.warn("Some sources have non-unique locations or uphole data. "
                          "Calculated statics may be inaccurate.")

        source_coords = statics[["SourceX", "SourceY", "SourceSurfaceElevation"]].to_numpy()
        statics["SurfaceStatics"] = self.estimate_statics(source_coords, **kwargs)
        if uphole_correction_method == "time":
            statics["Statics"] = statics["SurfaceStatics"] - statics["SourceUpholeTime"]
        elif uphole_correction_method == "depth":
            source_elevations = source_coords[:, -1] - statics["SourceDepth"].to_numpy()
            source_coords = np.column_stack([source_coords[:, :2], source_elevations])
            statics["Statics"] = self.estimate_statics(source_coords, **kwargs)
        else:
            statics["Statics"] = statics["SurfaceStatics"]
        return statics

    def _get_receiver_statics(self, survey, index_cols, **kwargs):
        index_cols = to_list(index_cols)
        all_cols_set = set(index_cols + ["GroupX", "GroupY", "ReceiverGroupElevation"])
        all_cols = list(all_cols_set)
        non_index_cols = list(all_cols_set - set(index_cols))

        statics = pl.from_pandas(survey.get_headers(all_cols), rechunk=False)
        is_duplicated_expr = pl.all_horizontal([pl.n_unique(col) == 1 for col in non_index_cols]).alias("IsUnique")
        statics = statics.groupby(index_cols).agg(pl.mean(non_index_cols), is_duplicated_expr).to_pandas()

        if not statics["IsUnique"].all():
            warnings.warn("Some receivers have non-unique locations. Calculated statics may be inaccurate.")

        receiver_coords = statics[["GroupX", "GroupY", "ReceiverGroupElevation"]].to_numpy()
        statics["Statics"] = self.estimate_statics(receiver_coords, **kwargs)
        return statics

    def calculate_statics(self, survey=None, uphole_correction_method="auto", source_id_cols=None,
                          receiver_id_cols=None, intermediate_datum=None, intermediate_datum_refractor=None,
                          final_datum=None, replacement_velocity=None):
        if survey is None:
            if not self.grid.has_survey:
                raise ValueError("A survey to calculate statics for must be passed")
            survey = self.grid.survey
        survey_list = to_list(survey)
        is_single_survey = isinstance(survey, Survey)
        _, uphole_correction_method_list = align_args(survey_list, uphole_correction_method)

        if source_id_cols is None:
            if any(sur.source_id_cols != survey_list[0].source_id_cols for sur in survey_list):
                raise ValueError
            if survey_list[0].source_id_cols is None:
                raise ValueError
            source_id_cols = survey_list[0].source_id_cols

        if receiver_id_cols is None:
            if any(sur.receiver_id_cols != survey_list[0].receiver_id_cols for sur in survey_list):
                raise ValueError
            if survey_list[0].receiver_id_cols is None:
                raise ValueError
            receiver_id_cols = survey_list[0].receiver_id_cols

        statics_kwargs = {
            "intermediate_datum": intermediate_datum,
            "intermediate_datum_refractor": intermediate_datum_refractor,
            "final_datum": final_datum,
            "replacement_velocity": replacement_velocity,
        }
        source_statics_list = []
        receiver_statics_list = []
        for sur, correction_method in zip(survey_list, uphole_correction_method_list):
            correction_method = get_uphole_correction_method(sur, correction_method)
            source_statics = self._get_source_statics(sur, source_id_cols, correction_method, **statics_kwargs)
            source_statics_list.append(source_statics)

            receiver_statics = self._get_receiver_statics(sur, receiver_id_cols, **statics_kwargs)
            receiver_statics_list.append(receiver_statics)

        survey = survey_list[0] if is_single_survey else survey_list
        source_statics = source_statics_list[0] if is_single_survey else source_statics_list
        receiver_statics = receiver_statics_list[0] if is_single_survey else receiver_statics_list
        return Statics(survey, source_statics, source_id_cols, receiver_statics, receiver_id_cols, validate=False)

    # Model visualization

    def plot_loss(self, show_reg=True, figsize=(10, 3)):
        n_rows = 4 if show_reg else 1
        width, height = figsize
        figsize = (width, height * n_rows)
        axes = plt.subplots(nrows=n_rows, sharex=True, squeeze=False, tight_layout=True, figsize=figsize)[1].ravel()
        axes[0].plot(self.loss_hist)
        axes[0].set_title("Traveltime MAE")
        if show_reg:
            axes[1].plot(self.velocities_reg_hist)
            axes[1].set_title("Weighted velocities interpolation MAPE")
            axes[2].plot(self.elevations_reg_hist)
            axes[2].set_title("Weighted elevations interpolation MAE")
            axes[3].plot(self.thicknesses_reg_hist)
            axes[3].set_title("Weighted thicknesses interpolation MAE")

    def plot_profile(self, **kwargs):
        return ProfilePlot(self, **kwargs).plot()
