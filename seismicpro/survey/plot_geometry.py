"""Implements a class for interactive plotting of survey geometry"""

from functools import partial

from sklearn.neighbors import NearestNeighbors

from ..utils import get_first_defined, calculate_axis_limits
from ..utils import InteractivePlot, PairedPlot, set_ticks, set_text_formatting, get_text_formatting_kwargs
from ..metrics.interactive_map import NonOverlayingIndicesPlot, OverlayingIndicesPlot


class SurveyGeometryPlot(PairedPlot):  # pylint: disable=too-many-instance-attributes
    """Interactive survey geometry plot.

    The plot provides 2 views:
    * Source view: displays locations of seismic sources. Highlights all activated receivers on click and displays the
      corresponding common source gather.
    * Receiver view: displays locations of receivers. Highlights all sources that activated the receiver on click and
      displays the corresponding common receiver gather.
    """

    #pylint: disable-next=too-many-arguments
    def __init__(self, survey, show_contour=True, keep_aspect=False, source_id_cols=None, source_sort_by=None,
                 receiver_id_cols=None, receiver_sort_by=None, sort_by=None, gather_plot_kwargs=None, x_ticker=None,
                 y_ticker=None, figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        (x_ticker, y_ticker), self.scatter_kwargs = set_text_formatting(x_ticker, y_ticker, **kwargs)
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        if gather_plot_kwargs is None:
            gather_plot_kwargs = {}
        gather_plot_kwargs = {"title": None, **text_kwargs, **gather_plot_kwargs}

        self.figsize = figsize
        self.orientation = orientation

        # Reindex the survey by source and receiver to speed up gather selection and fit nearest neighbors to project a
        # click on the map to the closest source or receiver
        self.source_map = survey.construct_fold_map(by="source", id_cols=source_id_cols)
        self.source_survey = survey.reindex(self.source_map.index_cols)
        self.source_coords = self.source_map.map_data.index.to_frame(index=False).to_numpy()
        self.source_neighbors = NearestNeighbors(n_neighbors=1).fit(self.source_coords)
        self.source_sort_by = get_first_defined(source_sort_by, sort_by)

        self.receiver_map = survey.construct_fold_map(by="receiver", id_cols=receiver_id_cols)
        self.receiver_survey = survey.reindex(self.receiver_map.index_cols)
        self.receiver_coords = self.receiver_map.map_data.index.to_frame(index=False).to_numpy()
        self.receiver_neighbors = NearestNeighbors(n_neighbors=1).fit(self.receiver_coords)
        self.receiver_sort_by = get_first_defined(receiver_sort_by, sort_by)

        # Calculate axes limits to fix them to avoid map plot shifting on view toggle
        x_lim = calculate_axis_limits([self.source_coords[:, 0].min(), self.source_coords[:, 0].max(),
                                       self.receiver_coords[:, 0].min(), self.receiver_coords[:, 0].max()])
        y_lim = calculate_axis_limits([self.source_coords[:, 1].min(), self.source_coords[:, 1].max(),
                                       self.receiver_coords[:, 1].min(), self.receiver_coords[:, 1].max()])
        contours = survey.geographic_contours if show_contour else None
        self.plot_map = partial(self._plot_map, contours=contours, keep_aspect=keep_aspect, x_lim=x_lim, y_lim=y_lim,
                                x_ticker=x_ticker, y_ticker=y_ticker, **self.scatter_kwargs)
        self.plot_gather = partial(self._plot_gather, **gather_plot_kwargs)
        self.activated_scatter = None

        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct a clickable plot of source and receiver locations."""
        return InteractivePlot(plot_fn=[self.plot_map, self.plot_map], click_fn=self.click, unclick_fn=self.unclick,
                               title=["Map of sources", "Map of receivers"], figsize=self.figsize)

    def construct_aux_plot(self):
        """Construct a gather plot."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        kwargs = {"plot_fn": self.plot_gather, "figsize": self.figsize, "toolbar_position": toolbar_position}
        has_overlays = self.source_map.has_overlaying_indices or self.receiver_map.has_overlaying_indices
        aux_plot_type = OverlayingIndicesPlot if has_overlays else NonOverlayingIndicesPlot
        aux_plot = aux_plot_type(**kwargs)
        aux_plot.box.layout.visibility = "hidden"
        return aux_plot

    @property
    def is_shot_view(self):
        """bool: Whether the current view displays shot locations."""
        return self.main.current_view == 0

    @property
    def active_map(self):
        """MetricMap: Fold map of sources or receivers depending on the current view."""
        return self.source_map if self.is_shot_view else self.receiver_map

    @property
    def survey(self):
        """Survey: A survey to get gathers from, depends on the current view."""
        return self.source_survey if self.is_shot_view else self.receiver_survey

    @property
    def coords(self):
        """np.ndarray: Coordinates of sources or receivers depending on the current view."""
        return self.source_coords if self.is_shot_view else self.receiver_coords

    @property
    def coords_neighbors(self):
        """sklearn.neighbors.NearestNeighbors: Nearest neighbors of sources or receivers depending on the current
        view."""
        return self.source_neighbors if self.is_shot_view else self.receiver_neighbors

    @property
    def sort_by(self):
        """str or list of str: Gather sorting depending on the current view."""
        return self.source_sort_by if self.is_shot_view else self.receiver_sort_by

    @property
    def activated_coords_cols(self):
        """list of 2 str: Coordinates columns describing highlighted objects depending on the current view."""
        return ["GroupX", "GroupY"] if self.is_shot_view else ["SourceX", "SourceY"]

    @property
    def main_color(self):
        """str: Color of the plotted objects depending on the current view."""
        return "tab:red" if self.is_shot_view else "tab:blue"

    @property
    def main_marker(self):
        """str: Marker of the plotted objects depending on the current view."""
        return "*" if self.is_shot_view else "v"

    @property
    def aux_color(self):
        """str: Color of the highlighted objects depending on the current view."""
        return "tab:blue" if self.is_shot_view else "tab:red"

    @property
    def aux_marker(self):
        """str: Marker of the highlighted objects depending on the current view."""
        return "v" if self.is_shot_view else "*"

    @property
    def map_x_label(self):
        """str: Label of the X map axis depending on the current view."""
        return "SourceX" if self.is_shot_view else "GroupX"

    @property
    def map_y_label(self):
        """str: Label of the Y map axis depending on the current view."""
        return "SourceY" if self.is_shot_view else "GroupY"

    def _plot_map(self, ax, contours, keep_aspect, x_lim, y_lim, x_ticker, y_ticker, **kwargs):
        """Plot locations of sources or receivers depending on the current view."""
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"

        ax.scatter(*self.coords.T, color=self.main_color, marker=self.main_marker, **kwargs)
        if contours is not None:
            for contour in contours:
                ax.fill(contour[:, 0, 0], contour[:, 0, 1], facecolor="gray", edgecolor="black", alpha=0.5)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.ticklabel_format(style="plain", useOffset=False)
        if keep_aspect:
            ax.set_aspect("equal", adjustable="box")
        set_ticks(ax, "x", self.map_x_label, **x_ticker)
        set_ticks(ax, "y", self.map_y_label, **y_ticker)

    def _plot_gather(self, ax, coords, index, **kwargs):
        """Display a gather with given index and highlight locations of activated sources or receivers."""
        _ = coords
        gather = self.survey.get_gather(index)
        if self.sort_by is not None:
            gather = gather.sort(by=self.sort_by)
        gather.plot(ax=ax, **kwargs)
        self._plot_activated(gather[self.activated_coords_cols])

    def _plot_activated(self, coords):
        """Highlight locations of activated sources or receivers."""
        if self.activated_scatter is not None:
            self.activated_scatter.remove()
        self.activated_scatter = self.main.ax.scatter(*coords.T, color=self.aux_color, marker=self.aux_marker,
                                                      **self.scatter_kwargs)
        self.main.fig.canvas.draw_idle()

    def click(self, coords):
        """Process a click on the map: display the selected gather and highlight locations of activated sources or
        receivers."""
        coords = tuple(self.coords[self.coords_neighbors.kneighbors([coords], return_distance=False).item()])
        gather_indices = self.active_map.get_indices_by_map_coords(coords).index.tolist()
        gather_coords = [coords] * len(gather_indices)
        titles = self.active_map.construct_items_titles(gather_coords, gather_indices)
        self.aux.process_map_click(gather_coords, gather_indices, titles)
        self.aux.box.layout.visibility = "visible"
        return coords

    def unclick(self):
        """Remove highlighted locations of sources or receivers, clear the gather plot and hide it."""
        if self.activated_scatter is not None:
            self.activated_scatter.remove()
            self.activated_scatter = None
        self.aux.clear()
        self.aux.box.layout.visibility = "hidden"
