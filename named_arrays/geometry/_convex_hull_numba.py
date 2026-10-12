"""Numba kernel for :func:`named_arrays.geometry.convex_hull`.

This module is imported lazily by :mod:`named_arrays.geometry` since
importing :mod:`numba` is slow.
"""

import numpy as np
import numba

__all__ = [
    "convex_hull_numba",
]


@numba.njit(cache=True)
def _cross(
    x: np.ndarray,
    y: np.ndarray,
    o: int,
    a: int,
    b: int,
) -> float:  # pragma: nocover
    """
    The cross product of the vectors from point `o` to points `a` and `b`.

    It is positive if `o`, `a`, `b` turn counterclockwise, negative if they
    turn clockwise, and zero if they are collinear.
    """
    return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])


@numba.njit(cache=True, parallel=True)
def convex_hull_numba(
    x: np.ndarray,
    y: np.ndarray,
    where: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: nocover
    """
    Find the convex hull of each of a batch of sets of points.

    Each hull is found with Andrew's monotone chain algorithm, which sorts
    the points and then walks along them twice, once for the lower half of
    the hull and once for the upper half, in :math:`O(n \\log n)` time.
    Points which lie on an edge of the hull, rather than at a corner, are not
    vertices of it, and neither are copies of a vertex.
    Points with a coordinate which is not finite are left out.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the points.
        Should be 2-dimensional, where the first axis is the set of points
        and the last axis is the point.
    y
        The :math:`y`-coordinates of the points.
        Should have the same shape as `x`.
    where
        Which of the points take part.
        Should have the same shape as `x`.

    Returns
    -------
    x_vertices
        For each set of points, the :math:`x`-coordinates of the vertices of
        its hull along the last axis, in counterclockwise order starting from
        the vertex with the least :math:`x`-coordinate (the least
        :math:`y`-coordinate among a tie). Every hull is padded to the number
        of vertices of the largest, and at least one, by repeating its last
        vertex. The vertices of a set with no points are NaN.
    y_vertices
        The :math:`y`-coordinates of the vertices, in the same shape.
    """
    num_sets, num_points = x.shape

    indices = np.empty((num_sets, num_points), dtype=np.int64)
    num = np.zeros(num_sets, dtype=np.int64)

    for s in numba.prange(num_sets):

        x_s = x[s]
        y_s = y[s]
        where_s = where[s]

        candidates = np.empty(num_points, dtype=np.int64)
        n = 0
        for i in range(num_points):
            if where_s[i] and np.isfinite(x_s[i]) and np.isfinite(y_s[i]):
                candidates[n] = i
                n += 1
        candidates = candidates[:n]

        if n == 0:
            continue

        if n == 1:
            indices[s, 0] = candidates[0]
            num[s] = 1
            continue

        # Sort by x, and by y among a tie in x. Stable sorts applied from
        # the least significant key to the most significant one.
        candidates = candidates[np.argsort(y_s[candidates], kind="mergesort")]
        candidates = candidates[np.argsort(x_s[candidates], kind="mergesort")]

        hull = np.empty(2 * n, dtype=np.int64)
        h = 0

        # The lower half of the hull, from left to right.
        for k in range(n):
            i = candidates[k]
            while h >= 2 and _cross(x_s, y_s, hull[h - 2], hull[h - 1], i) <= 0:
                h -= 1
            hull[h] = i
            h += 1

        # The upper half of the hull, from right to left.
        h_lower = h + 1
        for k in range(n - 2, -1, -1):
            i = candidates[k]
            while h >= h_lower and _cross(x_s, y_s, hull[h - 2], hull[h - 1], i) <= 0:
                h -= 1
            hull[h] = i
            h += 1

        # The upper half ends on the vertex the lower half started from.
        h -= 1

        # If every point coincides, the two ends of the chains are copies of
        # the same point.
        if h == 2:
            a = hull[0]
            b = hull[1]
            if x_s[a] == x_s[b] and y_s[a] == y_s[b]:
                h = 1

        indices[s, :h] = hull[:h]
        num[s] = h

    num_vertices = 1
    for s in range(num_sets):
        num_vertices = max(num_vertices, num[s])

    x_vertices = np.empty((num_sets, num_vertices))
    y_vertices = np.empty((num_sets, num_vertices))

    for s in numba.prange(num_sets):
        h = num[s]
        for v in range(num_vertices):
            if h == 0:
                x_vertices[s, v] = np.nan
                y_vertices[s, v] = np.nan
            else:
                i = indices[s, min(v, h - 1)]
                x_vertices[s, v] = x[s, i]
                y_vertices[s, v] = y[s, i]

    return x_vertices, y_vertices
