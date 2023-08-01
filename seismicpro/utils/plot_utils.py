"""Utility functions for visualization"""

# pylint: disable=invalid-name
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def as_dict(val, key):
    """Construct a dict with a {`key`: `val`} structure if given `val` is not a `dict`, or copy `val` otherwise."""
    return val.copy() if isinstance(val, dict) else {key: val}


def save_figure(fig, fname, dpi=100, bbox_inches="tight", pad_inches=0.1, **kwargs):
    """Save the given figure. All `args` and `kwargs` are passed directly to `matplotlib.pyplot.savefig`."""
    fig.savefig(fname, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)


def calculate_axis_limits(coords):
    """Calculate axis limits by coordinates of items being plotted. Mimics default matplotlib behavior."""
    coords = np.array(coords)
    min_coord = coords.min()
    max_coord = coords.max()
    margin_candidates = 0.05 * np.array([max_coord - min_coord, abs(max_coord), 1])
    margin = margin_candidates[~np.isclose(margin_candidates, 0)][0]
    return (min_coord - margin, max_coord + margin)


TEXT_FORMATTING_ARGS = {"fontsize", "fontfamily", "fontweight"}


def get_text_formatting_kwargs(**kwargs):
    """Get text formatting parameters from `kwargs`."""
    return {key: val for key, val in kwargs.items() if key in TEXT_FORMATTING_ARGS}


def set_text_formatting(*args, **kwargs):
    """Pop text formatting parameters from `kwargs` and set them as defaults for each of `args` tranformed to dict."""
    global_formatting = {arg: kwargs.pop(arg) for arg in TEXT_FORMATTING_ARGS if arg in kwargs}
    text_args = [{**global_formatting, **({} if arg is None else as_dict(arg, key="label"))} for arg in args]
    return text_args, kwargs


def add_colorbar(ax, artist, colorbar, divider=None, y_ticker=None):
    """Add a colorbar to the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add a colorbar to.
    artist : matplotlib.cm.ScalarMappable
        A mappable artist described by the colorbar.
    colorbar : bool or dict
        If `False` does not add a colorbar. If `True`, adds a colorbar with default parameters. If `dict`, defines
        keyword arguments for `matplotlib.figure.Figure.colorbar`.
    divider : mpl_toolkits.axes_grid1.axes_divider.AxesDivider, optional
        A divider of `ax`. If given, will be used to create child axes for the colorbar.
    y_ticker : dict, optional
        Parameters to control text formatting of y ticks of the created colorbar.
    """
    if not isinstance(colorbar, (bool, dict)):
        raise ValueError(f"colorbar must be bool or dict but {type(colorbar)} was passed")
    if colorbar is False:
        return
    colorbar = {} if colorbar is True else colorbar
    if divider is None:
        divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(artist, cax=cax, **colorbar)
    if y_ticker is not None:
        format_subplot_yticklabels(cax, **y_ticker)


def format_subplot_yticklabels(ax, fontsize=None, fontfamily=None, fontweight=None, **kwargs):
    """Set text formatting of y ticks of `ax` axes. This method is mainly used to format ticks on subplots such as a
    colorbar. It updates only font size, family and weight and does not support tick rotation."""
    _ = kwargs
    for tick in ax.get_yticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontfamily(fontfamily)
        tick.set_fontweight(fontweight)


def set_ticks(ax, axis, label='', major_labels=None, minor_labels=None, num=None,
              step_ticks=None, step_labels=None, round_to=0, **kwargs):
    """Set ticks and labels for `x` or `y` axis depending on the `axis`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        An axis on which ticks are set.
    axis : "x" or "y"
        Whether to set ticks for "x" or "y" axis of `ax`.
    label : str, optional, defaults to ''
        The label to set for `axis` axis.
    major_labels : array-like, optional, defaults to None
        An array of major labels for axis ticks.
    minor_labels : array-like, optional, defaults to None
        An array of minor labels for axis ticks.
    num : int, optional, defaults to None
        The number of evenly spaced ticks on the axis.
    step_ticks : int, optional, defaults to None
        A step between two adjacent ticks in samples (e.g. place every hundredth tick).
    step_labels : int, optional, defaults to None
        A step between two adjacent tick in the units of the corresponding labels (e.g. place a tick every 200ms for an
        axis, whose labels are measured in milliseconds). Should be None if `tick_labels` is None.
    round_to : int, optional, defaults to 0
        The number of decimal places to round tick labels to. If 0, tick labels will be cast to integers.
    kwargs : misc, optional
        Additional keyword arguments to control text formatting and rotation. Passed directly to
        `matplotlib.axis.Axis.set_label_text` and `matplotlib.axis.Axis.set_ticklabels`.

    Notes
    -----
    matplotlib does not update axes's data intervals when new artist is redrawn on the existing axes in interactive
    mode, which leads to incorrect tick positioning. To overcome this, call `ax.clear()` before drawing a new artist.

    Raises
    ------
    ValueError
        If `step_labels` is provided when tick_labels are None or not monotonically increasing.
    """
    locator, formatter = _process_ticks(labels=major_labels, num=num, step_ticks=step_ticks,
                                        step_labels=step_labels, round_to=round_to)
    rotation_kwargs = _pop_rotation_kwargs(kwargs)
    ax_obj = getattr(ax, f"{axis}axis")
    ax_obj.set_label_text(label, **kwargs)
    ax_obj.set_ticklabels([], **kwargs, **rotation_kwargs)
    ax_obj.set_major_locator(locator)
    ax_obj.set_major_formatter(formatter)

    if minor_labels is not None:
        _, formatter = _process_ticks(labels=minor_labels, round_to=round_to)
        ax_obj.set_minor_locator(ticker.AutoMinorLocator(n=4))
        ax_obj.set_minor_formatter(formatter)
        ax_obj.set_tick_params(which='minor', labelsize=kwargs.get("fontsize", plt.rcParams['font.size']) * 0.8)


def _process_ticks(labels, num=None, step_ticks=None, step_labels=None, round_to=0):
    """Create an axis locator and formatter by given `labels` and tick layout parameters."""
    if num is not None:
        locator = ticker.LinearLocator(num)
    elif step_ticks is not None:
        locator = ticker.IndexLocator(step_ticks, 0)
    elif step_labels is not None:
        if labels is None:
            raise ValueError("step_labels cannot be used: plotter does not provide labels.")
        if (np.diff(labels) < 0).any():
            raise ValueError("step_labels is valid only for monotonically increasing labels.")
        candidates = np.arange(labels[0], labels[-1], step_labels)
        ticks = np.searchsorted(labels, candidates)
        # Always include last label along the axis and remove duplicates
        ticks = np.unique(np.append(ticks, len(labels) - 1))
        locator = ticker.FixedLocator(ticks)
    else:
        locator = ticker.AutoLocator()

    def round_tick(tick, *args, round_to):
        """Format tick value."""
        _ = args
        if round_to is not None:
            return f'{tick:.{round_to}f}'
        return tick

    def get_tick_from_labels(tick, *args, labels, round_to):
        """Get tick label by its index in `labels` and format the resulting value."""
        _ = args
        if (tick < 0) or (tick > len(labels)-1):
            return None
        label_value = labels[np.round(tick).astype(np.int32)]
        return round_tick(label_value, round_to=round_to)

    if labels is None:
        formatter = partial(round_tick, round_to=round_to)
    else:
        formatter = partial(get_tick_from_labels, labels=labels, round_to=round_to)

    return locator, ticker.FuncFormatter(formatter)


def _pop_rotation_kwargs(kwargs):
    """Pop the keys responsible for text rotation from `kwargs`."""
    ROTATION_ARGS = {"ha", "rotation_mode"}
    rotation = kwargs.pop("rotation", None)
    rotation_kwargs = {arg: kwargs.pop(arg) for arg in ROTATION_ARGS if arg in kwargs}
    if rotation is not None:
        rotation_kwargs = {"rotation": rotation, "ha": "right", "rotation_mode": "anchor", **rotation_kwargs}
    return rotation_kwargs
