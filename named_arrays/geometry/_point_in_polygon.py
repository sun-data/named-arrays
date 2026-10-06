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
    Instead, each point is given the index of its polygon.

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

    if isinstance(x, u.Quantity):
        unit = x.unit
        if unit != 1:
            x = x.value
            y = y.to_value(unit)
            vertices_x = vertices_x.to_value(unit)
            vertices_y = vertices_y.to_value(unit)

    shape_vertices = np.broadcast_shapes(np.shape(vertices_x), np.shape(vertices_y))
    vertices_x = np.broadcast_to(vertices_x, shape_vertices)
    vertices_y = np.broadcast_to(vertices_y, shape_vertices)

    *shape_polygons, num_vertices = shape_vertices
    shape_polygons = tuple(shape_polygons)
    num_polygons = math.prod(shape_polygons)

    shape_points = np.broadcast_shapes(np.shape(x), np.shape(y), shape_polygons)

    polygon = np.arange(num_polygons).reshape(shape_polygons)

    x = np.broadcast_to(x, shape_points)
    y = np.broadcast_to(y, shape_points)
    polygon = np.broadcast_to(polygon, shape_points)

    result = _point_in_polygon_numba.point_in_polygon_numba(
        x=x.reshape(-1),
        y=y.reshape(-1),
        vertices_x=vertices_x.reshape(num_polygons, num_vertices),
        vertices_y=vertices_y.reshape(num_polygons, num_vertices),
        polygon=polygon.reshape(-1),
    )

    result = result.reshape(shape_points)

    return result
