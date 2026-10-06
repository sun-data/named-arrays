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
    polygon: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    Numba-accelerated check if a given point is inside or on the boundary of a polygon.

    Vectorized version of :func:`regridding.geometry.point_is_inside_polygon`.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the test points.
        Should be 1-dimensional.
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
    polygon
        The index along the first axis of `vertices_x` and `vertices_y`
        of the polygon to test each point against.
        Should be 1-dimensional, with the same number of elements as `x`.
    """

    num_pts = x.shape[0]

    result = np.empty(num_pts, dtype=np.bool)

    for i in numba.prange(num_pts):
        p = polygon[i]
        result[i] = regridding.geometry.point_is_inside_polygon(
            x=x[i],
            y=y[i],
            vertices_x=vertices_x[p],
            vertices_y=vertices_y[p],
        )

    return result
