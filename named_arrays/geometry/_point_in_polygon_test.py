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
            na.linspace(-1, 1, axis="x", num=5) * u.mm,
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
) -> None:
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
) -> None:
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
) -> None:
    """
    The memory taken is in proportion to the number of points,
    not to the number of points times the number of vertices,
    when the points are in a different unit than the vertices
    and when the polygon moves along an axis of the points.
    """
    num = 100_000
    num_vertices = 101
    radius = 10 * u.mm
    angle = na.linspace(0, 360, axis=axis, num=num_vertices) * u.deg
    shift = 0 * u.mm
    if moving:
        shift = na.linspace(0, 1, axis="wavelength", num=3) * u.mm
    vertices_x = (radius * np.cos(angle) + shift).explicit
    vertices_y = radius * np.sin(angle)
    # none of these lands on an edge, which lie on a vertex at y = 0
    x = na.linspace(-2, 2, axis="point", num=num).explicit * u.cm
    if moving:
        x = (x.to(u.mm) + 0 * shift).explicit

    def test() -> na.AbstractExplicitArray:
        """Test the points against the polygons."""
        return na.geometry.point_in_polygon(
            x=x,
            y=0 * u.mm,
            vertices_x=vertices_x,
            vertices_y=vertices_y,
            axis=axis,
        )

    # compile the kernel before measuring
    test()

    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        result = test()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not tracing:
            tracemalloc.stop()

    assert np.all(result == (np.abs(x - shift) < radius))

    # room for four float64 copies of the points, where a copy of the
    # vertices for every point takes two times 101 of them
    num_points = result.size
    assert peak - baseline < 4 * 8 * num_points


def test_point_in_polygon_converts_every_argument_to_one_unit() -> None:
    """
    The points and vertices are compared in one unit, taken from whichever
    argument has one, and an argument without a unit counts as dimensionless.
    """
    vertices_x = 60 * u.percent * np.cos(angles)
    vertices_y = 60 * u.percent * np.sin(angles)
    x = na.linspace(0.5, 0.7, axis="x", num=2)

    result = na.geometry.point_in_polygon(
        x=x,
        y=0,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        axis=axis,
    )

    assert np.all(result == (x < 0.6))


def test_point_in_polygon_refuses_points_without_a_unit_against_lengths() -> None:
    """
    A point without a unit is dimensionless, so it cannot be compared with
    vertices which are lengths, even when another coordinate has a length.
    """
    with pytest.raises(u.UnitConversionError):
        na.geometry.point_in_polygon(
            x=0,
            y=1.5 * u.cm,
            vertices_x=circle.x,
            vertices_y=circle.y,
            axis=axis,
        )


def test_point_in_polygon_refuses_an_axis_the_vertices_lack() -> None:
    """A misspelled vertex axis raises, rather than testing every vertex alone."""
    with pytest.raises(ValueError, match="not an axis of the vertices"):
        na.geometry.point_in_polygon(
            x=0 * u.mm,
            y=0 * u.mm,
            vertices_x=circle.x,
            vertices_y=circle.y,
            axis="vertx",
        )
