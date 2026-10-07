from typing import TypeVar
import math
import numpy as np
import astropy.units as u
import named_arrays as na

PointT = TypeVar("PointT", bound="float | u.Quantity | na.AbstractScalar")
VertexT = TypeVar("VertexT", bound="na.AbstractScalar")

def point_in_polygon(
    x: PointT,
    y: PointT,
    vertices_x: VertexT,
    vertices_y: VertexT,
    axis: str,
) -> "na.AbstractExplicitArray":
    """
    Check if a given point is inside or on the boundary of a polygon.

    This function is a wrapper around
    :func:`regridding.geometry.point_is_inside_polygon`.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the test points.
    y
        The :math:`y`-coordinates of the test points.
    vertices_x
        The :math:`x`-coordinates of the polygon's vertices.
    vertices_y
        The :math:`y`-coordinates of the polygon's vertices.
    axis
        The logical axis representing the different vertices of the polygon.

    Examples
    --------

    Check if some random points are inside a randomly-generated polygon.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define a random polygon
        axis = "vertex"
        num_vertices = 7
        radius = na.random.uniform(5, 15, shape_random={axis: num_vertices})
        angle = na.linspace(0, 2 * np.pi, axis=axis, num=num_vertices)
        vertices_x = radius * np.cos(angle)
        vertices_y = radius * np.sin(angle)

        # Define some random points
        x = na.random.uniform(-20, 20, shape_random=dict(r=1000))
        y = na.random.uniform(-20, 20, shape_random=dict(r=1000))

        # Select which points are inside the polygon
        where = na.geometry.point_in_polygon(
            x=x,
            y=y,
            vertices_x=vertices_x,
            vertices_y=vertices_y,
            axis=axis,
        )

        # Plot the results as a scatter plot
        fig, ax = plt.subplots()
        na.plt.fill(
            vertices_x,
            vertices_y,
            ax=ax,
            facecolor="none",
            edgecolor="black",
        )
        na.plt.scatter(
            x,
            y,
            where=where,
            ax=ax,
        );
    """
    return na._named_array_function(
        func=point_in_polygon,
        x=x,
        y=y,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        axis=axis,
    )


def _point_in_polygon_quantity(
    x: float | np.ndarray | u.Quantity,
    y: float | np.ndarray | u.Quantity,
    vertices_x: np.ndarray | u.Quantity,
    vertices_y: np.ndarray | u.Quantity,
) -> np.ndarray:
    """
    Check if a given point is inside or on the boundary of a polygon.

    Each point is tested against the polygon it shares its other axes with.
    The polygons are never broadcast against the points, since a copy of
    every vertex for every point would take memory in proportion to both.
    Instead, the kernel works out which polygon each point belongs to from
    where the point is in the grid.

    If any of the arguments has a unit, every argument is converted to the
    first such unit, so an argument without one counts as dimensionless.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the test points.
    y
        The :math:`y`-coordinates of the test points.
    vertices_x
        The :math:`x`-coordinates of the polygon's vertices.
        The last axis should represent the different vertices of the polygon.
        The other axes should broadcast against the test points.
    vertices_y
        The :math:`y`-coordinates of the polygon's vertices.
        The last axis should represent the different vertices of the polygon.
        The other axes should broadcast against the test points.
    """
    from . import _point_in_polygon_numba

    arrays = (x, y, vertices_x, vertices_y)
    units = [a.unit for a in arrays if isinstance(a, u.Quantity)]
    if units:
        x, y, vertices_x, vertices_y = (
            u.Quantity(a, copy=False).to_value(units[0]) for a in arrays
        )

    shape_vertices = np.broadcast_shapes(np.shape(vertices_x), np.shape(vertices_y))
    *shape_polygons, num_vertices = shape_vertices
    shape_points = np.broadcast_shapes(np.shape(x), np.shape(y), tuple(shape_polygons))

    num_axes = len(shape_points)
    shape_polygons = (1,) * (num_axes - len(shape_polygons)) + tuple(shape_polygons)
    num_polygons = math.prod(shape_polygons)

    # How far along the flattened polygons one step along each axis of the
    # points moves, which is nowhere along the axes the polygons lack.
    stride = np.zeros(num_axes, dtype=np.int64)
    step = 1
    for a in reversed(range(num_axes)):
        if shape_polygons[a] > 1:
            stride[a] = step
        step *= shape_polygons[a]

    result = _point_in_polygon_numba.point_in_polygon_numba(
        x=_contiguous(x, shape_points, (-1,)),
        y=_contiguous(y, shape_points, (-1,)),
        vertices_x=_contiguous(vertices_x, shape_vertices, (num_polygons, num_vertices)),
        vertices_y=_contiguous(vertices_y, shape_vertices, (num_polygons, num_vertices)),
        shape=np.array(shape_points, dtype=np.int64),
        stride=stride,
    )

    result = result.reshape(shape_points)

    return result


def _contiguous(
    a: float | np.ndarray,
    shape: tuple[int, ...],
    newshape: tuple[int, ...],
) -> np.ndarray:
    """
    Broadcast an array to a shape and reshape it into a C-contiguous array
    of 64-bit floats.

    The kernel is compiled once for each type and layout of its arguments,
    so they are always given to it in the same ones.  The array is copied
    only when it is not already laid out that way.

    Parameters
    ----------
    a
        The array to broadcast and reshape.
    shape
        The shape to broadcast `a` to.
    newshape
        The shape to give the broadcast array.
    """
    a = np.asarray(a)
    if a.shape != shape:
        a = np.broadcast_to(a, shape)
    return np.ascontiguousarray(a.reshape(newshape), dtype=np.float64)
