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
        )
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
