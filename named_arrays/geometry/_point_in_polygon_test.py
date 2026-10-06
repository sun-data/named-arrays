import tracemalloc
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

axis = "vertex"

radius = 10 * u.mm
angles = na.linspace(0, 360, axis=axis, num=11) * u.deg

circle = radius * na.Cartesian2dVectorArray(
    x=np.cos(angles),
    y=np.sin(angles),
)

@pytest.mark.parametrize(
    argnames=["x", "y", "vertices_x", "vertices_y", "axis", "result_expected"],
    argvalues=[
        (
            0 * u.mm,
            0 * u.mm,
            circle.x,
            circle.y,
            axis,
            True,
        ),
        (
            10 * u.mm,
            10 * u.mm,
            circle.x,
            circle.y,
            axis,
            False,
        ),
        (
            na.linspace(-1, 1, axis="x", num=5),
            0 * u.mm,
            circle.x,
            circle.y,
            axis,
            True,
        ),
        (
            na.UniformUncertainScalarArray(0 * u.mm, 1 * u.mm),
            0 * u.mm,
            circle.x,
            circle.y,
            axis,
            True,
        ),
        (
            0 * u.mm,
            0 * u.mm,
            circle.x + na.UniformUncertainScalarArray(0 * u.mm, 1 * u.mm),
            circle.y,
            axis,
            True,
        ),
    ]
)
def test_point_in_polygon(
    x: float | u.Quantity | na.AbstractScalar,
    y: float | u.Quantity | na.AbstractScalar,
    vertices_x: na.AbstractScalar,
    vertices_y: na.AbstractScalar,
    axis: str,
    result_expected: na.AbstractScalar,
):
    result = na.geometry.point_in_polygon(
        x=x,
        y=y,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        axis=axis,
    )

    assert np.all(result == result_expected)


@pytest.mark.parametrize("unit", [u.mm, u.cm])
def test_point_in_polygon_tests_each_point_against_its_own_polygon(
    unit: u.UnitBase,
):
    """
    Points which share an axis with the polygons are each tested against
    the polygon at the same index along it, and an axis which only the
    polygons have is added to the result, whatever the unit of the points.
    """
    half_width = na.linspace(1, 3, axis="size", num=3) * u.mm
    shift = na.linspace(0, 2, axis="shift", num=2) * u.mm
    angle = na.linspace(45, 315, axis=axis, num=4) * u.deg
    vertices_x = np.sqrt(2) * half_width * np.cos(angle) + shift
    vertices_y = np.sqrt(2) * half_width * np.sin(angle)

    x = na.linspace(-3.5, 3.5, axis="point", num=8) * u.mm
    y = half_width / 2

    result = na.geometry.point_in_polygon(
        x=x.to(unit),
        y=y.to(unit),
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        axis=axis,
    )

    result_expected = np.abs(x - shift) < half_width
    result_expected = na.broadcast_to(result_expected, result.shape)

    assert result.shape == dict(size=3, shift=2, point=8)
    assert np.all(result == result_expected)


@pytest.mark.parametrize(
    argnames="moving",
    argvalues=[False, True],
)
def test_point_in_polygon_does_not_copy_the_vertices_for_every_point(
    moving: bool,
):
    """
    The memory taken is in proportion to the number of points,
    not to the number of points times the number of vertices,
    when the points are in a different unit than the vertices
    and when the polygon moves along an axis of the points.
    """
    num = 100_000
    num_vertices = 101
    angle = na.linspace(0, 360, axis=axis, num=num_vertices) * u.deg
    vertices_x = 10 * u.mm * np.cos(angle)
    vertices_y = 10 * u.mm * np.sin(angle)
    x = na.linspace(-2, 2, axis="point", num=num).explicit * u.cm
    if moving:
        shift = na.linspace(0, 1, axis="wavelength", num=3) * u.mm
        vertices_x = (vertices_x + shift).explicit
        x = x.to(u.mm) + 0 * shift

    def test() -> na.AbstractExplicitArray:
        """Test the points against the polygons."""
        return na.geometry.point_in_polygon(
            x=x,
            y=0 * u.mm,
            vertices_x=vertices_x,
            vertices_y=vertices_y,
            axis=axis,
        )

    result_expected = test()

    tracemalloc.start()
    try:
        result = test()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert np.all(result == result_expected)
    assert peak < 3 * num * 8 * 8
