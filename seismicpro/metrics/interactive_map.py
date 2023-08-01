"""Implements classes for interactive metric map plotting"""

from functools import partial

from sklearn.neighbors import NearestNeighbors

from ..utils import get_first_defined, get_text_formatting_kwargs, align_args
from ..utils.interactive_plot_utils import InteractivePlot, DropdownOptionPlot, PairedPlot


class NonOverlayingIndicesPlot(InteractivePlot):
    """Construct an interactive plot that displays data representation at click locations in case when each pair of
    spatial coordinates being plot on the map corresponds to a single metric map item. Used by `ScatterMapPlot` for
    non-binarized maps with `False` value of `has_overlaying_indices` flag. Passes the last click coordinates and an
    index of the corresponding item to the current view plotter in addition to `ax`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_coords = None
        self.current_index = None
        self.current_title = None

    @property
    def title(self):
        """str: Return the title of the plot for current item."""
        return self.current_title

    @property
    def plot_fn(self):
        """callable: Plotter of the current view with the last click coordinates and an index of the corresponding item
        passed."""
        if self.current_coords is None or self.current_index is None:
            return None
        return partial(super().plot_fn, coords=self.current_coords, index=self.current_index)

    def process_map_click(self, coords, indices, titles):
        """Handle a click on the main metric map plot. Each of `coords`, `indices` and `titles` are guaranteed to
        contain a single element."""
        self.current_coords = coords[0]
        self.current_index = indices[0]
        self.current_title = titles[0]
        self.redraw()


class OverlayingIndicesPlot(DropdownOptionPlot):
    """Construct an interactive plot that displays data representation at click locations in case when several metric
    map items are mapped to the same spatial location on the plot. Used by `ScatterMapPlot` for non-binarized maps with
    `True` value of `has_overlaying_indices` flag and `BinarizedMapPlot` for all binarized maps. Passes coordinates and
    an index of an item selected in the dropdown list to the current view plotter in addition to `ax`."""

    def process_map_click(self, coords, indices, titles):
        """Handle a click on the main metric map plot by updating the list of dropdown options and selecting the first
        one of them."""
        options = [{"coords": coord, "index": index, "option_title": title}
                   for coord, index, title in zip(coords, indices, titles)]
        self.update_state(0, options)


class MetricMapPlot(PairedPlot):  # pylint: disable=abstract-method
    """Base class for interactive metric map visualization.

    Two methods should be redefined in a concrete plotter child class:
    * `preprocess_click_coords` - transform coordinates of a click into coordinates of the metric map data.
    * `construct_aux_plot` - construct an interactive plot which displays map contents at click location. Must be an
      instance of a subclass of `InteractivePlot` and implement `process_map_click` method which should accept
      `coords`, `indices` and `titles` lists with coordinates of items at click location, their indices and titles of
      the auxiliary plot.
    """
    def __init__(self, metric_map, plot_on_click=None, plot_on_click_kwargs=None, title=None, is_lower_better=None,
                 figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        plot_on_click, plot_on_click_kwargs = align_args(plot_on_click, plot_on_click_kwargs)
        plot_on_click_kwargs = [{} if plot_kwargs is None else plot_kwargs for plot_kwargs in plot_on_click_kwargs]
        plot_on_click_kwargs = [{**text_kwargs, **plot_kwargs} for plot_kwargs in plot_on_click_kwargs]

        self.figsize = figsize
        self.orientation = orientation

        self.metric_map = metric_map
        self.is_lower_better = is_lower_better
        self.title = metric_map.plot_title if title is None else title
        self.plot_map = partial(metric_map.plot, title="", is_lower_better=is_lower_better, **kwargs)
        self.plot_on_click = [partial(plot_fn, **plot_kwargs)
                              for plot_fn, plot_kwargs in zip(plot_on_click, plot_on_click_kwargs)]
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct the metric map plot."""
        return InteractivePlot(plot_fn=self.plot_map, click_fn=self.click, title=self.title, figsize=self.figsize)

    def preprocess_click_coords(self, click_coords):
        """Transform coordinates of a click into coordinates of the metric map data."""
        _ = click_coords
        raise NotImplementedError

    def click(self, coords):
        """Handle a click on the map plot."""
        map_coords = self.preprocess_click_coords(coords)
        if map_coords is None:
            return None
        is_ascending = not get_first_defined(self.is_lower_better, self.metric_map.metric.is_lower_better, True)
        click_indices = self.metric_map.get_indices_by_map_coords(map_coords).sort_values(ascending=is_ascending)
        indices, metric_values = zip(*click_indices.items())
        coords = [self.metric_map.get_coords_by_index(index) for index in indices]
        titles = self.metric_map.construct_items_titles(coords, indices, metric_values)
        self.aux.process_map_click(coords, indices, titles)
        return map_coords

    def plot(self):
        """Display the map and perform initial clicking."""
        super().plot()
        self.main.click(self.metric_map.get_worst_coords(self.is_lower_better))


class ScatterMapPlot(MetricMapPlot):
    """Construct an interactive plot of a non-aggregated metric map."""

    def __init__(self, metric_map, plot_on_click, **kwargs):
        self.coords = metric_map.map_data.index.to_frame(index=False).to_numpy()
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)
        super().__init__(metric_map, plot_on_click, **kwargs)

    def construct_aux_plot(self):
        """Construct an interactive plot with data representation at click location."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        kwargs = {"plot_fn": self.plot_on_click, "figsize": self.figsize, "toolbar_position": toolbar_position}
        aux_plot_type = OverlayingIndicesPlot if self.metric_map.has_overlaying_indices else NonOverlayingIndicesPlot
        return aux_plot_type(**kwargs)

    def preprocess_click_coords(self, click_coords):
        """Return map coordinates closest to coordinates of the click."""
        return tuple(self.coords[self.coords_neighbors.kneighbors([click_coords], return_distance=False).item()])


class BinarizedMapPlot(MetricMapPlot):
    """Construct an interactive plot of a binarized metric map."""

    def construct_aux_plot(self):
        """Construct an interactive plot with map contents at click location."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return OverlayingIndicesPlot(plot_fn=self.plot_on_click, figsize=self.figsize,
                                     toolbar_position=toolbar_position)

    def preprocess_click_coords(self, click_coords):
        """Return coordinates of a bin corresponding to coordinates of a click. Ignore the click if it was performed
        outside the map."""
        coords = (int(click_coords[0] + 0.5), int(click_coords[1] + 0.5))
        return coords if coords in self.metric_map.map_data else None
