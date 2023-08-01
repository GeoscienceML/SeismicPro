"""Implements interactive plots of vertical velocity spectrum and residual velocity spectrum."""

from functools import partial

import numpy as np

from ..utils import get_text_formatting_kwargs
from ..utils.interactive_plot_utils import InteractivePlot, PairedPlot


class VelocitySpectrumPlot(PairedPlot):  # pylint: disable=too-many-instance-attributes
    """Define an interactive velocity spectrum plot.

    This plot also displays the gather used to calculate the velocity spectrum. Clicking on velocity spectrum highlight
    the corresponding hodograph on the gather plot and allows performing NMO correction of the gather with the selected
    velocity by switching the view. The width of the hodograph matches the window size used to calculate the spectrum
    on both views. An initial click is performed on the maximum spectrum value.
    """
    def __init__(self, velocity_spectrum, title=None, sharey=True, gather_plot_kwargs=None,
                 figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        if gather_plot_kwargs is None:
            gather_plot_kwargs = {}
        self.gather_plot_kwargs = {"title": None, **text_kwargs, **gather_plot_kwargs}

        self.figsize = figsize
        self.orientation = orientation
        self.title = title
        self.click_time = None
        self.click_vel = None
        self.velocity_spectrum = velocity_spectrum
        self.gather = self.velocity_spectrum.gather.copy(ignore="data").sort('offset')
        self.plot_velocity_spectrum = partial(self.velocity_spectrum._plot, title=None, **kwargs)

        super().__init__(orientation=orientation)
        if sharey:
            self.aux.ax.sharey(self.main.ax)

    def construct_main_plot(self):
        """Construct a clickable velocity spectrum plot."""
        return InteractivePlot(plot_fn=self.plot_velocity_spectrum, click_fn=self.click, unclick_fn=self.unclick,
                               title=self.title, figsize=self.figsize)

    def construct_aux_plot(self):
        """Construct a correctable gather plot."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        plotter = InteractivePlot(plot_fn=[self.plot_gather, partial(self.plot_gather, corrected=True)],
                                  title=self.get_gather_title, figsize=self.figsize, toolbar_position=toolbar_position)
        plotter.view_button.disabled = True
        return plotter

    def get_gather_title(self):
        """Get title of the gather plot."""
        if (self.click_time is None) or (self.click_vel is None):
            return "Gather"
        return f"Hodograph from {self.click_time:.0f} ms with {self.click_vel:.2f} km/s velocity"

    def get_gather(self, corrected=False):
        """Get an optionally corrected gather."""
        if not corrected:
            return self.gather
        max_stretch_factor = self.velocity_spectrum.max_stretch_factor
        return self.gather.copy(ignore=["headers", "data", "samples"]) \
                          .apply_nmo(self.click_vel * 1000, max_stretch_factor=max_stretch_factor)

    def get_hodograph_times(self, corrected):
        """Get hodograph times if a click has been performed."""
        if (self.click_time is None) or (self.click_vel is None):
            return None
        if not corrected:
            return np.sqrt(self.click_time**2 + self.gather.offsets**2/self.click_vel**2)
        return np.full_like(self.gather.offsets, self.click_time)

    def plot_gather(self, ax, corrected=False):
        """Plot the gather and a hodograph if click has been performed."""
        gather = self.get_gather(corrected=corrected)
        gather.plot(ax=ax, **self.gather_plot_kwargs)

        hodograph_times = self.get_hodograph_times(corrected=corrected)
        if hodograph_times is None:
            return
        hodograph_y = self.velocity_spectrum.times_to_indices(hodograph_times) - 0.5  # Correction for pixel center
        half_window = self.velocity_spectrum.half_win_size_samples
        hodograph_low = np.clip(hodograph_y - half_window, 0, self.gather.n_times - 1)
        hodograph_high = np.clip(hodograph_y + half_window, 0, self.gather.n_times - 1)
        ax.fill_between(np.arange(len(hodograph_y)), hodograph_low, hodograph_high, color="tab:blue", alpha=0.5)

    def click(self, coords):
        """Highlight the hodograph defined by click location on the gather plot."""
        click_time, click_vel = self.velocity_spectrum.get_time_velocity_by_indices(coords[1], coords[0])
        if (click_time is None) or (click_vel is None):
            return None  # Ignore click
        self.aux.view_button.disabled = False
        self.click_time = click_time
        self.click_vel = click_vel
        self.aux.redraw()
        return coords

    def unclick(self):
        """Remove the highlighted hodograph and switch to a non-corrected view."""
        self.click_time = None
        self.click_vel = None
        self.aux.set_view(0)
        self.aux.view_button.disabled = True
