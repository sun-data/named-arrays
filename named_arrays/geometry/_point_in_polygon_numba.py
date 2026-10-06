"""Numba kernel for :func:`named_arrays.geometry.point_in_polygon`.

This module is imported lazily by :mod:`named_arrays.geometry` since
importing :mod:`numba` and :mod:`regridding` is slow.
"""

import numpy as np
import numba
import regridding

__all__ = [
    "point_in_polygon_numba",
]


@numba.njit(cache=True, parallel=True)
def point_in_polygon_numba(
    x: np.ndarray,
    y: np.ndarray,
    vertices_x: np.ndarray,
    vertices_y: np.ndarray,
    shape: np.ndarray,
    stride: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    Numba-accelerated check if a given point is inside or on the boundary of a polygon.

    Vectorized version of :func:`regridding.geometry.point_is_inside_polygon`.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the test points, flattened in C order
        from a grid of the given `shape`.
    y
        The :math:`y`-coordinates of the test points.
        Should be 1-dimensional, with the same number of elements as `x`.
    vertices_x
        The :math:`x`-coordinates of the polygons' vertices.
        Should be 2-dimensional, where the first axis is the polygon
        and the last axis is the vertex.
    vertices_y
        The :math:`y`-coordinates of the polygons' vertices.
        Should have the same shape as `vertices_x`.
    shape
        The shape of the grid of test points.
    stride
        For each axis of the grid of test points, how far along the first
        axis of `vertices_x` and `vertices_y` one step along it moves.
        Zero along the axes the polygons do not vary along.
    """

    num_pts = x.shape[0]
    num_axes = shape.shape[0]

    result = np.empty(num_pts, dtype=np.bool)

    for i in numba.prange(num_pts):
        p = 0
        inner = 1
        for a in range(num_axes - 1, -1, -1):
            p += ((i // inner) % shape[a]) * stride[a]
            inner *= shape[a]
        result[i] = regridding.geometry.point_is_inside_polygon(
            x=x[i],
            y=y[i],
            vertices_x=vertices_x[p],
            vertices_y=vertices_y[p],
        )

    return result
