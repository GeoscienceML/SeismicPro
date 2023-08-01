from functools import partial

import numpy as np
from ipywidgets import widgets

from ...utils import get_first_defined, add_colorbar, get_text_formatting_kwargs, set_ticks, IDWInterpolator
from ...utils import InteractivePlot, PairedPlot, WIDGET_HEIGHT


class SlidingDepthPlot(InteractivePlot):
    def __init__(self, *, slider_min, slider_max, slider_init, slide_fn=None, **kwargs):
        min_widget = widgets.HTML(value=str(slider_min), layout=widgets.Layout(height=WIDGET_HEIGHT))
        max_widget = widgets.HTML(value=str(slider_max), layout=widgets.Layout(height=WIDGET_HEIGHT))
        self.slider = widgets.FloatSlider(value=slider_init, min=slider_min, max=slider_max, step=1, readout=False,
                                          layout=widgets.Layout(flex="1 1 auto", height=WIDGET_HEIGHT))
        self.slider.observe(slide_fn, "value")
        self.slider_box = widgets.HBox([min_widget, self.slider, max_widget],
                                       layout=widgets.Layout(width="90%", margin="auto"))
        super().__init__(**kwargs)

    def construct_header(self):
        """Append the slider below the plot header."""
        return widgets.VBox([super().construct_header(), self.slider_box], layout=widgets.Layout(overflow="hidden"))


class ProfilePlot(PairedPlot):
    def __init__(self, model, sampling_interval=None, min_velocity=None, max_velocity=None, figsize=(4.5, 4.5),
                 fontsize=8, orientation="horizontal", **kwargs):
        self.kwargs = {"fontsize": fontsize, **kwargs}
        self.text_kwargs = get_text_formatting_kwargs(**self.kwargs)

        z_min, x_min, y_min = model.grid.origin
        nz, nx, ny = model.grid.shape
        dz, dx, dy = model.grid.cell_size
        self.x_cell_centers = x_min + dx / 2 + dx * np.arange(nx)
        self.y_cell_centers = y_min + dy / 2 + dy * np.arange(ny)
        self.z_cell_centers = z_min + dz / 2 + dz * np.arange(nz)

        self.model = model
        self.sampling_interval = get_first_defined(sampling_interval, min(dx, dy))

        self.min_depth = z_min
        self.max_depth = z_min + dz * nz
        self.curr_depth = (self.max_depth + self.min_depth) / 2

        self.velocity_grid = self.model.velocities_tensor.detach().numpy()
        if self.model.grid.has_survey:
            self.velocity_grid = np.where(self.model.grid.air_mask, np.nan, self.velocity_grid)
        self.min_velocity = get_first_defined(min_velocity, np.nanmin(self.velocity_grid))
        self.max_velocity = get_first_defined(max_velocity, np.nanmax(self.velocity_grid))

        coords = np.array(np.meshgrid(self.x_cell_centers, self.y_cell_centers)).T.reshape(-1, 2)
        values = model.velocities.transpose([1, 2, 0]).reshape(-1, nz)
        self.interpolator = IDWInterpolator(coords, values, neighbors=4)

        self.figsize = figsize
        self.orientation = orientation
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct a clickable multi-view plot of parameters of the near-surface velocity model."""
        return SlidingDepthPlot(plot_fn=partial(self.plot_depth_slide, **self.kwargs),
                                slice_fn=partial(self.slice_fn, **self.text_kwargs), slide_fn=self.on_depth_change,
                                slider_min=self.min_depth, slider_max=self.max_depth, slider_init=self.curr_depth,
                                unclick_fn=self.unclick, figsize=self.figsize)

    def construct_aux_plot(self):
        """Construct a plot of a velocity model profile."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        aux_plot = InteractivePlot(figsize=self.figsize, toolbar_position=toolbar_position)
        aux_plot.box.layout.visibility = "hidden"
        return aux_plot

    def on_depth_change(self, change):
        self.curr_depth = change["new"]
        self.main.redraw()

    def plot_depth_slide(self, ax, **kwargs):
        i = min(int((self.curr_depth - self.min_depth) / self.model.grid.cell_size[0]), self.model.grid.shape[0] - 1)
        slide = self.velocity_grid[i].T
        img = ax.imshow(slide, origin="lower", aspect="auto", cmap="coolwarm",
                        vmin=self.min_velocity, vmax=self.max_velocity)
        add_colorbar(ax, img, True)
        set_ticks(ax, "x", label="X", major_labels=self.x_cell_centers, **kwargs)
        set_ticks(ax, "y", label="Y", major_labels=self.y_cell_centers, **kwargs)
        self.main.set_title(f"Seismic velocity at elevation of {int(self.curr_depth)} m")

    def slice_fn(self, start_coords, stop_coords, **kwargs):
        """Display a profile of near-surface velocity model from `start_coords` to `stop_coords`."""
        _, x_min, y_min = self.model.grid.origin
        dz, dx, dy = self.model.grid.cell_size

        ix_start, iy_start = start_coords
        ix_stop, iy_stop = stop_coords

        x_start = x_min + ix_start * dx
        x_stop = x_min + ix_stop * dx
        y_start = y_min + iy_start * dy
        y_stop = y_min + iy_stop * dy

        offset = np.sqrt((x_start - x_stop)**2 + (y_start - y_stop)**2)
        n_points = max(int(offset // self.sampling_interval), 2)
        dist_along_profile = np.linspace(0, offset, n_points)
        x_linspace = np.linspace(x_start, x_stop, n_points)
        y_linspace = np.linspace(y_start, y_stop, n_points)
        slice_coords = np.column_stack([x_linspace, y_linspace])

        self.aux.clear()
        data = self.interpolator(slice_coords).T[::-1]
        img = self.aux.ax.imshow(data, cmap="coolwarm", vmin=self.min_velocity, vmax=self.max_velocity, aspect="auto")
        add_colorbar(self.aux.ax, img, True)

        y_slide = (self.max_depth - dz / 2 - self.curr_depth) / dz
        self.aux.ax.axhline(y=y_slide, color="black")

        self.aux.set_title(f"A profile from ({int(x_start)}, {int(y_start)}) to ({int(x_stop)}, {int(y_stop)})")
        set_ticks(self.aux.ax, "x", label="Distance along the profile, m", major_labels=dist_along_profile, **kwargs)
        set_ticks(self.aux.ax, "y", label="Elevation, m", major_labels=self.z_cell_centers[::-1], **kwargs)
        self.aux.box.layout.visibility = "visible"

    def unclick(self):
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"
