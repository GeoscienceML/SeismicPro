import math

import numpy as np
from numba import njit
from fteikpy._interp import interp3d


TOL = 1e-6
FASTMATH_FLAGS = {"nnan", "ninf", "nsz", "arcp", "contract", "afn"}


@njit("(f8)(f8, f8, f8[:])", nogil=True, fastmath=FASTMATH_FLAGS)
def get_shrink_factor(coord, grad, bounds):
    if abs(grad) < TOL:
        return 0

    delta = bounds[1] - bounds[0]
    cell_ix = math.floor((coord - bounds[0]) / delta)

    if grad > 0:
        dist = bounds[cell_ix + 1] - coord
    else:
        dist = coord - bounds[cell_ix]

    if abs(dist) < TOL:
        dist = delta
    return dist / abs(grad)


@njit("(f8)(f8, f8[:])", nogil=True, fastmath=FASTMATH_FLAGS)
def map_to_edge(coord, bounds):
    delta = bounds[1] - bounds[0]
    cell_ix = math.floor((coord - bounds[0]) / delta)
    if abs(bounds[cell_ix] - coord) < TOL:
        return bounds[cell_ix]
    if cell_ix + 1 < len(bounds) and abs(bounds[cell_ix + 1] - coord) < TOL:
        return bounds[cell_ix + 1]
    return coord


@njit("Tuple((f8[:, :, :], i4[:], b1[:]))(f8[:], f8[:, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:], f8[:], f8[:], i4)",
      nogil=True, fastmath=FASTMATH_FLAGS)
def raytrace(src, dst_list, z_grad, x_grad, y_grad, z, x, y, max_n_steps):
    dz, dx, dy = z[1] - z[0], x[1] - x[0], y[1] - y[0]

    zsrc, xsrc, ysrc = src
    zsrc_i = math.floor((zsrc - z[0]) / dz)
    z_stop_min = z[0] + zsrc_i * dz - TOL
    z_stop_max = z[0] + (zsrc_i + 1) * dz + TOL
    xsrc_i = math.floor((xsrc - x[0]) / dx)
    x_stop_min = x[0] + xsrc_i * dx - TOL
    x_stop_max = x[0] + (xsrc_i + 1) * dx + TOL
    ysrc_i = math.floor((ysrc - y[0]) / dy)
    y_stop_min = y[0] + ysrc_i * dy - TOL
    y_stop_max = y[0] + (ysrc_i + 1) * dy + TOL

    n_dst = len(dst_list)
    rays = np.empty((n_dst, max_n_steps + 1, 3), dtype=np.float64)
    counts = np.empty(n_dst, dtype=np.int32)
    succeeded = np.zeros(n_dst, dtype=np.bool_)

    for i in range(n_dst):
        cur = dst_list[i].copy()
        ray = rays[i]
        ray[0] = cur

        for j in range(1, max_n_steps):
            gz = -interp3d(z, x, y, z_grad, cur)
            gx = -interp3d(z, x, y, x_grad, cur)
            gy = -interp3d(z, x, y, y_grad, cur)

            z_shrink = get_shrink_factor(cur[0], gz, z)
            x_shrink = get_shrink_factor(cur[1], gx, x)
            y_shrink = get_shrink_factor(cur[2], gy, y)
            factors = [factor for factor in [z_shrink, x_shrink, y_shrink] if factor > 0]
            if len(factors) == 0:
                break

            shrink = min(factors)
            cur[0] = map_to_edge(cur[0] + gz * shrink, z)
            cur[1] = map_to_edge(cur[1] + gx * shrink, x)
            cur[2] = map_to_edge(cur[2] + gy * shrink, y)
            ray[j] = cur

            z_stop = z_stop_min <= cur[0] <= z_stop_max
            x_stop = x_stop_min <= cur[1] <= x_stop_max
            y_stop = y_stop_min <= cur[2] <= y_stop_max
            if z_stop and x_stop and y_stop:
                succeeded[i] = True
                break

        if succeeded[i]:
            j += 1
            ray[j] = src
        counts[i] = j + 1

    return rays, counts, succeeded


@njit(nogil=True)
def refine_index(index, candidate_diffs, velocity_grid):
    iz, ix, iy = index
    candidates = [(iz, ix, iy)] + [(iz + dz, ix + dx, iy + dy) for (dz, dx, dy) in candidate_diffs
                                                               if (iz >= -dz) and (ix >= -dx) and (iy >= -dy)]
    velocities = np.array([velocity_grid[i] for i in candidates])
    return candidates[np.argmax(velocities)]


@njit(nogil=True, fastmath=FASTMATH_FLAGS)
def get_passes(rays, counts, succeeded, velocity_grid, origin, cell_size):
    _, nx, ny = velocity_grid.shape
    dz, dx, dy = cell_size
    z_min, x_min, y_min = origin
    n_cells = (counts[succeeded] - 1).sum()

    trace_indices = np.empty(n_cells, dtype=np.int64)
    cell_indices = np.empty(n_cells, dtype=np.int64)
    cell_passes = np.empty(n_cells, dtype=np.float64)

    offset = 0
    i_succeeded = 0

    for i in range(len(rays)):
        if not succeeded[i]:
            continue

        count = counts[i]
        ray = rays[i, :count]

        for j in range(count - 1):
            z_pass = abs(ray[j + 1, 0] - ray[j, 0])
            x_pass = abs(ray[j + 1, 1] - ray[j, 1])
            y_pass = abs(ray[j + 1, 2] - ray[j, 2])
            cell_passes[offset + j] = (z_pass * z_pass + x_pass * x_pass + y_pass * y_pass) ** 0.5

            z_mid = (ray[j + 1, 0] + ray[j, 0]) / 2
            x_mid = (ray[j + 1, 1] + ray[j, 1]) / 2
            y_mid = (ray[j + 1, 2] + ray[j, 2]) / 2
            iz = math.floor((z_mid - z_min) / dz)
            ix = math.floor((x_mid - x_min) / dx)
            iy = math.floor((y_mid - y_min) / dy)
            index = (iz, ix, iy)

            if z_pass < TOL:
                if x_pass < TOL:
                    candidate_diffs = [(0, -1, 0), (-1, -1, 0), (-1, 0, 0)]
                elif y_pass < TOL:
                    candidate_diffs = [(0, 0, -1), (-1, 0, -1), (-1, 0, 0)]
                else:
                    candidate_diffs = [(-1, 0, 0),]
                index = refine_index(index, candidate_diffs, velocity_grid)
            elif x_pass < TOL:
                if y_pass < TOL:
                    candidate_diffs = [(0, -1, 0), (0, 0, -1), (0, -1, -1)]
                else:
                    candidate_diffs = [(0, -1, 0),]
                index = refine_index(index, candidate_diffs, velocity_grid)
            elif y_pass < TOL:
                candidate_diffs = [(0, 0, -1),]
                index = refine_index(index, candidate_diffs, velocity_grid)

            iz, ix, iy = index
            cell_indices[offset + j] = iz * nx * ny + ix * ny + iy

        trace_indices[offset : offset + count - 1] = i_succeeded
        i_succeeded += 1
        offset += count - 1

    return trace_indices, cell_indices, cell_passes


@njit(nogil=True, fastmath=FASTMATH_FLAGS)
def describe_rays(src, dst_list, velocities, origin, cell_size, z_grad, x_grad, y_grad, z, x, y, max_n_steps):
    rays, counts, succeeded = raytrace(src, dst_list, z_grad, x_grad, y_grad, z, x, y, max_n_steps)
    trace_indices, cell_indices, cell_passes = get_passes(rays, counts, succeeded, velocities, origin, cell_size)
    return rays, counts, succeeded, trace_indices, cell_indices, cell_passes
