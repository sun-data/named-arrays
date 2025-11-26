from typing import TypeVar
import numpy as np
import astropy.units as u
import numba
import regridding
import named_arrays as na

PointT = TypeVar("PointT", bound="float | u.Quantity | na.AbstractScalar")
VertexT = TypeVar("VertexT", bound="na.AbstractScalar")

def point_in_polygon(
    x: PointT,
    y: PointT,
    vertices_x: VertexT,
    vertices_y: VertexT,
    axis: str,
) -> PointT | VertexT:
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
        angle = na.linspace(0, 360, axis=axis, num=num_vertices) * u.deg
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
        )
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
    x: u.Quantity,
    y: u.Quantity,
    vertices_x: u.Quantity,
    vertices_y: u.Quantity,
) -> np.ndarray:
    """
    Check if a given point is inside or on the boundary of a polygon.

    Parameters
    ----------
    x
        The :math:`x`-coordinates of the test points.
    y
        The :math:`y`-coordinates of the test points.
    vertices_x
        The :math:`x`-coordinates of the polygon's vertices.
        The last axis should represent the different vertices of the polygon.
    vertices_y
        The :math:`y`-coordinates of the polygon's vertices.
        The last axis should represent the different vertices of the polygon.
    """

    if isinstance(x, u.Quantity):
        unit = x.unit
        y = y.to_value(unit)
        vertices_x = vertices_x.to_value(unit)
        vertices_y = vertices_y.to_value(unit)

    shape_points = np.broadcast(x, y).shape
    shape_vertices = np.broadcast(vertices_x, vertices_y).shape

    num_vertices = shape_vertices[~0]

    shape_points = np.broadcast_shapes(shape_points, shape_vertices[:~0])
    shape_vertices = shape_points + (num_vertices,)

    x = np.broadcast_to(x, shape_points)
    y = np.broadcast_to(y, shape_points)

    vertices_x = np.broadcast_to(vertices_x, shape_vertices)
    vertices_y = np.broadcast_to(vertices_y, shape_vertices)

    result = _point_in_polygon_numba(
        x=x.reshape(-1),
        y=y.reshape(-1),
        vertices_x=vertices_x.reshape(-1, num_vertices),
        vertices_y=vertices_y.reshape(-1, num_vertices),
    )

    result = result.reshape(shape_points)

    return result

@numba.njit(cache=True, parallel=True)
def _point_in_polygon_numba(
    x: np.ndarray,
    y: np.ndarray,
    vertices_x: np.ndarray,
    vertices_y: np.ndarray,
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
        The :math:`x`-coordinates of the polygon's vertices.
        Should be 2-dimensional, where the first axis has the same number
        of elements as `x`.
    vertices_y
        The :math:`y`-coordinates of the polygon's vertices.
        Should be 2-dimensional, where the last axis has the same number
        of elements as `vertices_y`.
    """

    num_pts, num_vertices = vertices_x.shape

    result = np.empty(num_pts, dtype=np.bool)

    for i in numba.prange(num_pts):
        result[i] = regridding.geometry.point_is_inside_polygon(
            x=x[i],
            y=y[i],
            vertices_x=vertices_x[i],
            vertices_y=vertices_y[i],
        )

    return result
