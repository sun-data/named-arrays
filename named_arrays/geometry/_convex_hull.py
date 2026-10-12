from typing import Sequence
import math
import numpy as np
import astropy.units as u
import named_arrays as na
from ._point_in_polygon import _contiguous

__all__ = [
    "convex_hull",
]


def convex_hull(
    points: "na.AbstractCartesian2dVectorArray",
    axis: str | Sequence[str],
    axis_vertex: str = "vertex",
    where: "bool | na.AbstractScalar" = True,
) -> "na.Cartesian2dVectorArray":
    """
    Find the convex hull of a set of points: the smallest convex polygon
    which contains all of them.

    The points along `axis` are reduced to the vertices of their hull, along
    the new axis `axis_vertex`, in counterclockwise order starting from the
    vertex with the least :math:`x`-coordinate (the least
    :math:`y`-coordinate among a tie). Every other axis of `points` is a
    batch axis, along which each set of points gets a hull of its own.

    Hulls have different numbers of vertices, so every hull is padded to the
    number of vertices of the largest by repeating its last vertex, which
    leaves the polygon it describes unchanged. Points which lie on an edge of
    a hull, rather than at a corner, are not vertices of it, and neither are
    copies of a vertex, so the hull of points which all coincide is that one
    point. Points with a coordinate which is not finite, like NaN, are left
    out. A set with no points has a hull whose vertices are all NaN.

    The vertices keep the unit of the :math:`x`-coordinates, or of the
    :math:`y`-coordinates if those have none, and the type of the
    coordinates if it is a floating-point type, or else are 64-bit floats.

    The hull of each set is found with Andrew's monotone chain algorithm, in
    :math:`O(n \\log n)` time for :math:`n` points.

    Parameters
    ----------
    points
        The points to find the hull of.
        Must have exactly the two components :math:`x` and :math:`y`, which
        must be scalars that are not uncertain.
    axis
        The logical axis or axes along which the points of a single set vary.
    axis_vertex
        The name of the new logical axis along which the vertices of each
        hull vary. It may be the name of one of the axes in `axis`, which
        the points no longer vary along, but not of any other axis.
    where
        Which of the points take part. Those for which it is :obj:`False`
        are left out of the hull.

    Examples
    --------

    Find the hull of a cloud of random points.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define a cloud of random points
        axis = "point"
        points = na.Cartesian2dVectorArray(
            x=na.random.normal(0, 1, shape_random={axis: 101}),
            y=na.random.normal(0, 2, shape_random={axis: 101}),
        )

        # Find the vertices of the hull of the points
        hull = na.geometry.convex_hull(points, axis=axis)

        # Plot the points and their hull
        fig, ax = plt.subplots()
        na.plt.scatter(points.x, points.y, ax=ax)
        na.plt.fill(
            hull.x,
            hull.y,
            ax=ax,
            axis="vertex",
            facecolor="none",
            edgecolor="black",
        );
    """
    return na._named_array_function(
        func=convex_hull,
        points=points,
        axis=axis,
        axis_vertex=axis_vertex,
        where=where,
    )


def _convex_hull_quantity(
    x: float | np.ndarray | u.Quantity,
    y: float | np.ndarray | u.Quantity,
    where: bool | np.ndarray,
    num_axes_points: int,
) -> tuple[np.ndarray | u.Quantity, np.ndarray | u.Quantity]:
    """
    Find the convex hull of each of a batch of sets of points.

    If the coordinates have units, both are converted to the unit of the
    :math:`x`-coordinates, or of the :math:`y`-coordinates if those have
    none, so a coordinate without one counts as dimensionless.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the points.
        Its last `num_axes_points` axes are the axes along which the points
        of a single set vary, and the others are batch axes.
    y
        The :math:`y`-coordinates of the points, which should broadcast
        against `x`.
    where
        Which of the points take part, which should broadcast against `x`.
    num_axes_points
        The number of axes, at the end of the broadcast shape of the
        arguments, along which the points of a single set vary.

    Returns
    -------
    x_vertices
        The :math:`x`-coordinates of the vertices of each hull, along the
        last axis, after the batch axes.
    y_vertices
        The :math:`y`-coordinates of the vertices, in the same shape.
    """
    from . import _convex_hull_numba

    dtype = np.result_type(np.asarray(x), np.asarray(y))
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64

    units = [a.unit for a in (x, y) if isinstance(a, u.Quantity)]
    unit = units[0] if units else None
    if unit is not None:
        x, y = (u.Quantity(a, copy=False).to_value(unit) for a in (x, y))

    shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(where))
    index_points = len(shape) - num_axes_points
    shape_batch = shape[:index_points]
    num_sets = math.prod(shape_batch)
    num_points = math.prod(shape[index_points:])
    newshape = (num_sets, num_points)

    x_vertices, y_vertices = _convex_hull_numba.convex_hull_numba(
        x=_contiguous(x, shape, newshape),
        y=_contiguous(y, shape, newshape),
        where=_contiguous(where, shape, newshape, dtype=bool),
    )

    # Every vertex is one of the points, so the vertices are given back in
    # the type of the points without rounding.
    shape_vertices = shape_batch + (x_vertices.shape[~0],)
    x_vertices = x_vertices.reshape(shape_vertices).astype(dtype, copy=False)
    y_vertices = y_vertices.reshape(shape_vertices).astype(dtype, copy=False)

    if unit is not None:
        x_vertices = x_vertices << unit
        y_vertices = y_vertices << unit

    return x_vertices, y_vertices
