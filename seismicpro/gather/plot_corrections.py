"""Implements interactive plots for gather corrections"""

from functools import partial

from ..utils.interactive_plot_utils import WIDGET_HEIGHT, InteractivePlot
from ..utils import MissingModule, as_dict

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")


class SlidingVelocityPlot(InteractivePlot):
    """Define an interactive plot with a slider on top of the canvas. The slider becomes invisible on the last view.

    Parameters
    ----------
    slider_min : float
        Minimum slider value.
    slider_max : float
        Maximum slider value.
    slide_fn : callable
        A function called on slider move.
    kwargs : misc, optional
        Additional keyword arguments to `InteractivePlot.__init__`.
    """
    def __init__(self, *, slider_min, slider_max, slide_fn=None, **kwargs):
        min_widget = widgets.HTML(value=str(slider_min), layout=widgets.Layout(height=WIDGET_HEIGHT))
        max_widget = widgets.HTML(value=str(slider_max), layout=widgets.Layout(height=WIDGET_HEIGHT))
        self.slider = widgets.FloatSlider(min=slider_min, max=slider_max, step=1, readout=False,
                                          layout=widgets.Layout(flex="1 1 auto", height=WIDGET_HEIGHT))
        self.slider.observe(slide_fn, "value")
        self.slider_box = widgets.HBox([min_widget, self.slider, max_widget],
                                       layout=widgets.Layout(width="90%", margin="auto"))
        super().__init__(**kwargs)

    def construct_header(self):
        """Append the slider below the plot header."""
        return widgets.VBox([super().construct_header(), self.slider_box], layout=widgets.Layout(overflow="hidden"))

    def on_view_toggle(self, event):
        """Hide the slider on the last view."""
        super().on_view_toggle(event)
        if self.current_view == self.n_views - 1:
            self.slider_box.layout.visibility = "hidden"
        else:
            self.slider_box.layout.visibility = "visible"


class CorrectionPlot:
    """Base class for interactive gather corrections.

    The plot provides 2 views:
    * Corrected gather (default). Correction is performed on the fly with the velocity controlled by a slider on top of
      the plot.
    * Source gather. This view disables the velocity slider.

    Two methods should be redefined in a concrete plotter child class:
    * `get_title` - the title of the corrected view,
    * `corrected_gather` - a property returning a corrected gather.
    """
    def __init__(self, gather, min_vel, max_vel, figsize, show_grid=True, **kwargs):
        kwargs = {"fontsize": 8, **kwargs}
        event_headers = kwargs.pop("event_headers", None)
        self.event_headers = None
        if event_headers is not None:
            event_headers = {"process_outliers": "discard", **as_dict(event_headers, "headers")}
            self.event_headers = event_headers["headers"]
        self.gather = gather
        self.plotter = SlidingVelocityPlot(plot_fn=[partial(self.plot_corrected_gather, show_grid=show_grid,
                                                            event_headers=event_headers, **kwargs),
                                                    partial(self.gather.plot, **kwargs)],
                                           slide_fn=self.on_velocity_change, slider_min=min_vel, slider_max=max_vel,
                                           title=[self.get_title, "Source gather"], figsize=figsize)

    def get_title(self):
        """Get title of the corrected view."""
        raise NotImplementedError

    @property
    def corrected_gather(self):
        """Gather: corrected gather."""
        raise NotImplementedError

    def plot_corrected_gather(self, ax, show_grid=True, **kwargs):
        """Plot the corrected gather."""
        self.corrected_gather.plot(ax=ax, **kwargs)
        if show_grid:
            ax.grid(which='major', axis='y', color='k', linestyle='--')

    def on_velocity_change(self, change):
        """Redraw the plot on velocity change."""
        _ = change
        self.plotter.redraw()

    def plot(self):
        """Display the plot."""
        self.plotter.plot()


class NMOCorrectionPlot(CorrectionPlot):
    """Interactive NMO correction plot."""

    def get_title(self):
        """Get title of the NMO-corrected view."""
        return f"Normal moveout correction with {self.plotter.slider.value:.0f} m/s velocity"

    @property
    def corrected_gather(self):
        """Gather: NMO-corrected gather."""
        return self.gather.copy(ignore=["headers", "data", "samples"]).apply_nmo(self.plotter.slider.value)


class LMOCorrectionPlot(CorrectionPlot):
    """Interactive LMO correction plot."""

    def get_title(self):
        """Get title of the LMO-corrected view."""
        return f"Linear moveout correction with {(self.plotter.slider.value):.0f} m/s velocity"

    @property
    def corrected_gather(self):
        """Gather: LMO-corrected gather."""
        gather_copy = self.gather.copy(ignore=["data", "samples"])
        return gather_copy.apply_lmo(self.plotter.slider.value, event_headers=self.event_headers)
