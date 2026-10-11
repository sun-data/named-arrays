from typing import Sequence
import math
import numpy as np
import astropy.units as u
import named_arrays as na

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
    a hull, rather than at a corner, are not vertices of it. A set with no
    points has a hull whose vertices are all NaN.

    The hull of each set is found with Andrew's monotone chain algorithm, in
    :math:`O(n \\log n)` time for :math:`n` points.

    Parameters
    ----------
    points
        The points to find the hull of.
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
    from . import _convex_hull_numba

    if not isinstance(points, na.AbstractCartesian2dVectorArray):
        raise TypeError(
            f"`points` must be an instance of "
            f"`named_arrays.AbstractCartesian2dVectorArray`, got {type(points)}."
        )

    if isinstance(axis, str):
        axis = (axis,)
    axis = tuple(axis)

    x = na.as_named_array(points.x).explicit
    y = na.as_named_array(points.y).explicit
    where = na.as_named_array(where).explicit

    for a in (x, y, where):
        if not isinstance(a, na.ScalarArray):
            raise TypeError(
                f"the components of `points` and `where` must be scalars "
                f"which are not uncertain, got {type(a)}."
            )

    shape = na.shape_broadcasted(x, y, where)

    missing = [a for a in axis if a not in shape]
    if missing:
        raise ValueError(f"{missing=} are not axes of `points` or `where`, {shape=}.")

    shape_batch = {a: shape[a] for a in shape if a not in axis}

    if axis_vertex in shape_batch:
        raise ValueError(
            f"{axis_vertex=} is already an axis of `points` which is not "
            f"reduced, {shape_batch=}."
        )
    shape_points = {a: shape[a] for a in axis}
    shape_ordered = shape_batch | shape_points

    unit = x.unit
    if unit is None:
        unit = y.unit

    def _values(a: "na.ScalarArray") -> np.ndarray:
        result = a.ndarray_aligned(shape_ordered)
        if unit is not None:
            result = u.Quantity(result).to_value(unit)
        result = np.broadcast_to(result, tuple(shape_ordered.values()))
        return result

    num_sets = math.prod(shape_batch.values())
    num_points = math.prod(shape_points.values())

    x_values = _values(x).reshape(num_sets, num_points)
    y_values = _values(y).reshape(num_sets, num_points)
    where_values = np.broadcast_to(
        where.ndarray_aligned(shape_ordered),
        tuple(shape_ordered.values()),
    ).reshape(num_sets, num_points)

    indices, num = _convex_hull_numba.convex_hull_numba(
        x=np.ascontiguousarray(x_values, dtype=float),
        y=np.ascontiguousarray(y_values, dtype=float),
        where=np.ascontiguousarray(where_values, dtype=bool),
    )

    num_vertices = max(int(num.max(initial=0)), 1)

    # Pad each hull to the same number of vertices by repeating its last.
    position = np.minimum(
        np.arange(num_vertices)[np.newaxis, :],
        np.maximum(num - 1, 0)[:, np.newaxis],
    )
    vertex = np.take_along_axis(indices, position, axis=1)

    x_vertices = np.take_along_axis(x_values, vertex, axis=1).astype(float)
    y_vertices = np.take_along_axis(y_values, vertex, axis=1).astype(float)
    x_vertices[num == 0] = np.nan
    y_vertices[num == 0] = np.nan

    shape_result = tuple(shape_batch.values()) + (num_vertices,)
    x_vertices = x_vertices.reshape(shape_result)
    y_vertices = y_vertices.reshape(shape_result)

    if unit is not None:
        x_vertices = x_vertices << unit
        y_vertices = y_vertices << unit

    axes = tuple(shape_batch) + (axis_vertex,)

    return na.Cartesian2dVectorArray(
        x=na.ScalarArray(x_vertices, axes=axes),
        y=na.ScalarArray(y_vertices, axes=axes),
    )
