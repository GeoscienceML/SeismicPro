from functools import partial

import numpy as np

from ...metrics import MetricMap
from ...utils import get_first_defined, add_colorbar, get_text_formatting_kwargs, set_ticks
from ...utils import InteractivePlot, DropdownViewPlot, PairedPlot


class ProfilePlot(PairedPlot):
    def __init__(self, model, sampling_interval=50, min_velocity=None, max_velocity=None, figsize=(4.5, 4.5),
                 fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        self.text_kwargs = get_text_formatting_kwargs(**kwargs)

        self.n_refractors = model.n_refractors
        self.sampling_interval = sampling_interval

        elevations = np.column_stack([model.surface_elevation_tensor.detach().cpu().numpy(),
                                      model.elevations_tensor.detach().cpu().numpy()])
        min_elevation = elevations.min()
        max_elevation = elevations.max()
        elevation_gap = 0.05 * (max_elevation - min_elevation)
        self.min_elevation = min_elevation - elevation_gap
        self.max_elevation = max_elevation + elevation_gap

        weathering_velocity = 1000 / model.weathering_slowness_tensor.detach().cpu().numpy()
        layer_velocities = 1000 / model.slownesses_tensor.detach().cpu().numpy()
        velocities = np.column_stack([weathering_velocity, layer_velocities])
        self.min_velocity = get_first_defined(min_velocity, velocities.min())
        self.max_velocity = get_first_defined(max_velocity, velocities.max())

        self.model = model
        self.model_params = np.column_stack([elevations, velocities])

        self.titles = (
            ["Surface elevation"] +
            [f"Elevation of layer {i+1}" for i in range(model.n_refractors)] +
            [f"Thickness of layer {i+1}" for i in range(model.n_refractors)] +
            ["Velocity of the weathering layer"] +
            [f"Velocity of layer {i+1}" for i in range(model.n_refractors)]
        )

        param_maps = [MetricMap(model.coords, data_val, coords_cols=["X", "Y"])
                      for data_val in np.column_stack([elevations, -np.diff(elevations, axis=1), velocities]).T]
        self.plot_fn = [partial(param_map._plot, title="", **kwargs) for param_map in param_maps]

        self.figsize = figsize
        self.orientation = orientation
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct a sliceable multi-view plot of parameters of the near-surface velocity model."""
        return DropdownViewPlot(plot_fn=self.plot_fn, slice_fn=partial(self.slice_fn, **self.text_kwargs),
                                unclick_fn=self.unclick, title=self.titles, preserve_clicks_on_view_change=True,
                                figsize=self.figsize)

    def construct_aux_plot(self):
        """Construct a plot of a velocity model profile."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        aux_plot = InteractivePlot(figsize=self.figsize, toolbar_position=toolbar_position)
        aux_plot.box.layout.visibility = "hidden"
        return aux_plot

    def slice_fn(self, start_coords, stop_coords, **kwargs):
        """Display a profile of near-surface velocity model from `start_coords` to `stop_coords`."""
        x_start, y_start = start_coords
        x_stop, y_stop = stop_coords
        offset = np.sqrt((x_start - x_stop)**2 + (y_start - y_stop)**2)
        n_points = max(int(offset // self.sampling_interval), 2)
        dist_along_profile = np.linspace(0, offset, n_points)
        x_linspace = np.linspace(x_start, x_stop, n_points)
        y_linspace = np.linspace(y_start, y_stop, n_points)
        slice_coords = np.column_stack([x_linspace, y_linspace])

        data = self.model.grid.interpolate(self.model_params, slice_coords).T
        elevations = np.row_stack([data[:self.n_refractors + 1], np.full(n_points, self.min_elevation)])
        velocities = data[self.n_refractors + 1:]

        self.aux.clear()
        for i in range(self.n_refractors + 1):
            polygon = self.aux.ax.fill_between(np.arange(n_points), elevations[i], elevations[i+1], lw=0, color="none")
            verts = np.vstack([p.vertices for p in polygon.get_paths()])
            extent = [verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()]
            filling = self.aux.ax.imshow(velocities[i].reshape(1, -1), cmap="coolwarm", aspect="auto",
                                         vmin=self.min_velocity, vmax=self.max_velocity, extent=extent)
            filling.set_clip_path(polygon.get_paths()[0], transform=self.aux.ax.transData)
            self.aux.ax.plot(elevations[i], color="black")
        add_colorbar(self.aux.ax, filling, True)
        self.aux.ax.set_ylim(self.min_elevation, self.max_elevation)

        self.aux.set_title(f"A profile from ({int(x_start)}, {int(y_start)}) to ({int(x_stop)}, {int(y_stop)})")
        set_ticks(self.aux.ax, "x", label="Distance along the profile, m", major_labels=dist_along_profile, **kwargs)
        set_ticks(self.aux.ax, "y", label="Elevation, m", **kwargs)
        self.aux.box.layout.visibility = "visible"

    def unclick(self):
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"
