"""Implements an algorithm for stacking velocity computation by vertical velocity spectrum"""

import math

import numpy as np
import rustworkx as rx
from numba import njit, prange


@njit(nogil=True)
def get_path_sum(spectrum_data, start_time_ix, start_vel_ix, end_time_ix, end_vel_ix):
    """Calculate sum of spectrum values along a line connecting 2 points with indices (`start_time_ix`, `start_vel_ix`)
    and (`end_time_ix`, `end_vel_ix`). `start_vel_ix` and `end_vel_ix` may be floats."""
    n_points = end_time_ix - start_time_ix
    dv = (end_vel_ix - start_vel_ix) / n_points
    res = 0
    for i in range(n_points):
        time_ix = start_time_ix + i
        vel_ix = start_vel_ix + i * dv
        prev_vel_ix = math.floor(vel_ix)
        next_vel_ix = math.ceil(vel_ix)
        weight = next_vel_ix - vel_ix
        res += spectrum_data[time_ix, prev_vel_ix] * weight + spectrum_data[time_ix, next_vel_ix] * (1 - weight)
    return res


@njit(nogil=True)  # pylint: disable-next=too-many-arguments
def create_edges_between_layers(spectrum_data, start_time, start_time_ix, start_velocities, start_velocities_ix,
                                start_bias, end_time, end_time_ix, end_velocities, end_velocities_ix, end_bias,
                                acceleration_bounds):
    """Return edges connecting nodes with given `start_time` and `start_velocities` and nodes with given `end_time` and
    `end_velocities` with their weights."""
    dt = (end_time - start_time) / 1000
    edges = []
    for start_vel_pos, (start_vel, start_vel_ix) in enumerate(zip(start_velocities, start_velocities_ix)):
        for end_vel_pos, (end_vel, end_vel_ix) in enumerate(zip(end_velocities, end_velocities_ix)):
            acceleration = (end_vel - start_vel) / dt
            if (acceleration < acceleration_bounds[0]) | (acceleration > acceleration_bounds[1]):
                continue
            weight = get_path_sum(spectrum_data, start_time_ix, start_vel_ix, end_time_ix, end_vel_ix)
            edges.append((start_bias + start_vel_pos, end_bias + end_vel_pos, weight))
    return edges


@njit(nogil=True, parallel=True)
def create_edges(spectrum_data, layer_times, layer_times_ix, layer_velocities, layer_velocities_ix, layer_biases,
                 max_n_skips, acceleration_bounds):
    """Return edges of a graph for stacking velocity computation with their weights."""
    layer_velocities = np.split(layer_velocities, layer_biases[1:] - 1)
    layer_velocities_ix = np.split(layer_velocities_ix, layer_biases[1:] - 1)

    edges = [[(1, 1, -1.0)] for _ in range((max_n_skips + 1) * (len(layer_times_ix) - 1))]
    for ix in prange(len(edges)):  # pylint: disable=not-an-iterable
        i = ix // (max_n_skips + 1)
        j = i + 1 + ix % (max_n_skips + 1)
        if j >= len(layer_times_ix):
            continue
        edges[ix] = create_edges_between_layers(spectrum_data, layer_times[i], layer_times_ix[i], layer_velocities[i],
                                                layer_velocities_ix[i], layer_biases[i], layer_times[j],
                                                layer_times_ix[j], layer_velocities[j], layer_velocities_ix[j],
                                                layer_biases[j], acceleration_bounds)
    edges = [layer_edges for layer_edges in edges if (len(layer_edges) > 1) or (layer_edges[0][-1] >= 0)]

    # Connect start node to all nodes of the first layer
    start_node = 0
    edges.append([(start_node, i + 1, 0.0) for i in range(len(layer_velocities_ix[0]))])

    # Connect all nodes of the last layer to the end node
    end_node = layer_biases[-1] + len(layer_velocities_ix[-1])
    edges.append([(layer_biases[-1] + i, end_node, 0.0) for i in range(len(layer_velocities_ix[-1]))])
    return edges, start_node, end_node


# pylint: disable-next=too-many-statements
def calculate_stacking_velocity(spectrum, init=None, bounds=None, relative_margin=0.2, acceleration_bounds="auto",
                                times_step=100, max_offset=5000, hodograph_correction_step=25, max_n_skips=2):
    """Calculate stacking velocity by vertical velocity spectrum.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    If calculated by velocity spectrum, stacking velocity should generally meet the following conditions:
    1. It should be monotonically increasing,
    2. Its gradient should be bounded above to avoid excessive gather stretching after NMO correction,
    3. It should pass through local energy maxima on the velocity spectrum.

    In order for these conditions to be satisfied, the following algorithm is proposed:
    1. Stacking velocity is being found inside an area bounded both left and right by two stacking velocities defined
       by `bounds`. If `bounds` are not given, they are assumed to be [`init` * (1 - `relative_margin`),
       `init` * (1 + `relative_margin`)], where `init` is a rough estimate of the stacking velocity being picked.
    2. An auxiliary directed acyclic graph is constructed so that:
        1. The whole time range of the velocity spectrum is covered with a grid of times with a step of `times_step`.
        2. A range of velocities is selected for each of these times so that:
            1. They cover the whole range of allowed velocities according to selected `bounds`,
            2. A difference between two adjacent velocities is so that two hodographs starting with these velocities
               from their common zero-offset time arrive at `max_offset` within `hodograph_correction_step`
               milliseconds from each other.
        3. Each (time, velocity) pair defines a node of the graph. An edge between nodes A and B exists only if:
            1. Time at node B is greater than that of node A but by no more than (`max_n_skips` + 1) * `times_step`,
            2. Transition from node A to B occurs with acceleration within provided `acceleration_bounds`.
        4. Edge weight is defined as a sum of velocity spectrum values along its path.
    3. A path with maximal velocity spectrum sum along it between any of starting and ending nodes is found using
       Dijkstra algorithm and is considered to be the required stacking velocity.

    Parameters
    ----------
    spectrum : VerticalVelocitySpectrum
        Vertical velocity spectrum to calculate stacking velocity for.
    init : StackingVelocity, optional
        A rough estimate of the stacking velocity being picked. Used to calculate `bounds` as
        [`init` * (1 - `relative_margin`), `init` * (1 + `relative_margin`)] if they are not given.
    bounds : array-like of two StackingVelocity, optional
        Left and right bounds of an area for stacking velocity picking. If not given, `init` must be passed.
    relative_margin : positive float, optional, defaults to 0.2
        A fraction of stacking velocities defined by `init` used to estimate `bounds` if they are not given.
    acceleration_bounds : tuple of two positive floats or "auto" or None, optional
        Minimal and maximal acceleration allowed for the stacking velocity function. If "auto", equals to the range of
        accelerations of stacking velocities in `bounds` extended by 50% in both directions. If `None`, only ensures
        that picked stacking velocity is monotonically increasing. Measured in meters/seconds^2.
    times_step : float, optional, defaults to 100
        A difference between two adjacent times defining graph nodes.
    max_offset : float, optional, defaults to 5000
        An offset for hodograph time estimation. Used to create graph nodes and calculate their velocities for each
        time.
    hodograph_correction_step : float, optional, defaults to 25
        The maximum difference in arrival time of two hodographs starting at the same zero-offset time and two adjacent
        velocities at `max_offset`. Used to create graph nodes and calculate their velocities for each time.
    max_n_skips : int, optional, defaults to 2
        Defines the maximum number of intermediate times between two nodes of the graph. Greater values increase
        computational costs, but tend to produce smoother stacking velocity.

    Returns
    -------
    times : 1d np.ndarray
        Times for which stacking velocities were picked. Measured in milliseconds.
    velocities : 1d np.ndarray
        Picked stacking velocities. Matches the length of `times`. Measured in meters/seconds.
    bounds_times : 1d np.ndarray
        Times for which velocity bounds were estimated. Measured in milliseconds.
    min_velocity_bound : 1d np.ndarray
        Minimum velocity bound. Matches the length of `bounds_times`. Measured in meters/seconds.
    max_velocity_bound : 1d np.ndarray
        Maximum velocity bound. Matches the length of `bounds_times`. Measured in meters/seconds.
    """
    spectrum_data = spectrum.velocity_spectrum.max() - spectrum.velocity_spectrum
    spectrum_times = np.asarray(spectrum.times, dtype=np.float32)
    spectrum_velocities = np.asarray(spectrum.velocities, dtype=np.float32)

    # Calculate times of graph nodes
    times_step_samples = int(times_step // spectrum.sample_interval)
    layer_times_ix = np.arange(0, len(spectrum_times), times_step_samples)
    layer_times_ix[-1] = len(spectrum_times) - 1
    layer_times = spectrum_times[layer_times_ix]

    # Estimate velocity bounds for each time where nodes are placed
    if bounds is not None:
        min_velocity_bound = bounds[0](layer_times)
        max_velocity_bound = bounds[1](layer_times)
        if (min_velocity_bound > max_velocity_bound).any():
            raise ValueError("Minimum velocity bound cannot be greater than the maximum one")
    else:
        center_vel = init(layer_times)
        min_velocity_bound = center_vel * (1 - relative_margin)
        max_velocity_bound = center_vel * (1 + relative_margin)
    min_velocity_bound = np.clip(min_velocity_bound, spectrum_velocities[0], spectrum_velocities[-1])
    max_velocity_bound = np.clip(max_velocity_bound, spectrum_velocities[0], spectrum_velocities[-1])

    # Calculate allowed acceleration bounds
    if acceleration_bounds is None:
        acceleration_bounds = [0, np.inf]
    elif acceleration_bounds == "auto":
        dt = np.diff(layer_times) / 1000
        min_velocity_accelerations = np.diff(min_velocity_bound) / dt
        max_velocity_accelerations = np.diff(max_velocity_bound) / dt
        min_acceleration = min(min_velocity_accelerations.min(), max_velocity_accelerations.min())
        max_acceleration = max(min_velocity_accelerations.max(), max_velocity_accelerations.max())
        acceleration_bounds = [min_acceleration * (1 - 0.5 * np.sign(min_acceleration)),
                               max_acceleration * (1 + 0.5 * np.sign(max_acceleration))]
    acceleration_bounds = np.array(acceleration_bounds, dtype=np.float32)
    if len(acceleration_bounds) != 2:
        raise ValueError("acceleration_bounds must be an array-like with 2 elements")
    if acceleration_bounds[1] <= acceleration_bounds[0]:
        raise ValueError("Upper acceleration bound must greater than the lower one")

    # Estimate node velocities for each time
    min_correction_time = np.sqrt(layer_times**2 + (max_offset * 1000 / max_velocity_bound)**2)  # ms
    max_correction_time = np.sqrt(layer_times**2 + (max_offset * 1000 / min_velocity_bound)**2)  # ms
    spectrum_velocity_indices = np.arange(len(spectrum_velocities))
    layer_velocities = []
    layer_velocities_ix = []
    for time, min_corr, max_corr in zip(layer_times, min_correction_time, max_correction_time):
        n_vels = int((max_corr - min_corr) // hodograph_correction_step) + 1
        layer_vels = np.sqrt(max_offset**2 * 1000**2 / (np.linspace(min_corr, max_corr, n_vels)[::-1]**2 - time**2))
        layer_vels_ix = np.interp(layer_vels, spectrum_velocities, spectrum_velocity_indices)
        layer_velocities.append(layer_vels)
        layer_velocities_ix.append(layer_vels_ix)

    # Calculate edges of the graph. Concat layer_velocities and layer_velocities_ix and split them back inside
    # create_edges to make numba work
    layer_biases = np.cumsum([1] + [len(node_vels) for node_vels in layer_velocities[:-1]])
    edges, start_node, end_node = create_edges(spectrum_data, layer_times, layer_times_ix,
                                               np.concatenate(layer_velocities), np.concatenate(layer_velocities_ix),
                                               layer_biases, max_n_skips, acceleration_bounds)

    # Create a graph and find the path with maximal velocity spectrum sum along it
    graph = rx.PyDiGraph()
    for layer_edges in edges:
        graph.extend_from_weighted_edge_list(layer_edges)
    paths_dict = rx.dijkstra_shortest_paths(graph, start_node, end_node, weight_fn=float)
    if end_node not in paths_dict:
        raise ValueError("No path was found for given parameters")
    path = np.array(paths_dict[end_node], dtype=np.int32)[1:-1]

    # Convert the path to arrays of times and stacking velocities
    times_ix = np.searchsorted(layer_biases, path, side="right") - 1
    velocities_ix = path - layer_biases[times_ix]
    times = layer_times[times_ix]
    velocities = np.array([layer_velocities[tix][vix] for tix, vix in zip(times_ix, velocities_ix)])
    return times, velocities, layer_times, min_velocity_bound, max_velocity_bound
