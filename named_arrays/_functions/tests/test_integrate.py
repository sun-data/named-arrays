import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

from . import test_functions
from . import test_functions_vertices

__all__ = []


# All FunctionArray fixture sources -- center, vertex, and polynomial-fit --
# combined so the shape-contract test exercises every family in one place.
_fixtures = (
    test_functions._function_arrays()
    + test_functions._polynomial_function_arrays()
    + test_functions_vertices._function_arrays()
)


@pytest.mark.parametrize("array", _fixtures)
def test_integrate_shape_contract(array: na.AbstractFunctionArray):
    """``axis`` removed from outputs and inputs; remaining axes preserved."""
    array = array.explicit
    inputs = array.inputs

    # default component=None integrates the whole input via volume_cell,
    # so axis must cover every dim of the inputs.
    axis = tuple(inputs.shape)

    if not axis:
        return

    # volume_cell isn't implemented on every fixture (e.g. composite vectors);
    # if it isn't, integrate is expected to surface that error.
    try:
        inputs.volume_cell(axis)
    except (NotImplementedError, TypeError, ValueError):
        with pytest.raises((NotImplementedError, TypeError, ValueError)):
            array.integrate(axis)
        return

    result = array.integrate(axis)

    assert isinstance(result, na.FunctionArray)
    for ax in axis:
        assert ax not in result.outputs.shape
        assert ax not in result.inputs.shape
    assert set(result.outputs.shape) == set(array.outputs.shape) - set(axis)


@pytest.mark.parametrize("num", [11, 101])
def test_integrate_constant_center(num: int):
    # center axis (inputs are samples) -> trapezoidal rule.
    # constant integrand 3 over [0, 2] -> 3 * 2 = 6, exact.
    f = na.FunctionArray(
        inputs=na.ScalarLinearSpace(0, 2, axis="x", num=num) * u.nm,
        outputs=na.ScalarArray(np.full(num, 3.0), axes=("x",)) * u.ph,
    )
    result = f.integrate("x")
    assert "x" not in result.outputs.shape
    assert np.allclose(result.outputs, 6 * u.nm * u.ph)


@pytest.mark.parametrize("num", [11, 101])
def test_integrate_linear_center(num: int):
    # the trapezoidal rule is exact for a linear integrand:
    # int_0^2 x dx = 2.
    x = na.ScalarLinearSpace(0, 2, axis="x", num=num) * u.nm
    f = na.FunctionArray(inputs=x, outputs=x)
    result = f.integrate("x")
    assert np.allclose(result.outputs, 2 * u.nm ** 2)


def test_integrate_constant_vertex():
    # vertex axis (inputs are bin edges, length N+1) -> Riemann sum.
    # constant integrand 3 over edges spanning [0, 2] -> 6, exact.
    f = na.FunctionArray(
        inputs=na.ScalarLinearSpace(0, 2, axis="x", num=11) * u.nm,
        outputs=na.ScalarArray(np.full(10, 3.0), axes=("x",)) * u.ph,
    )
    result = f.integrate("x")
    assert "x" not in result.outputs.shape
    assert np.allclose(result.outputs, 6 * u.nm * u.ph)


def test_integrate_vector_center():
    # 2D center grid, constant integrand 3 over [0,2]^2 -> 12 (component=None).
    f = na.FunctionArray(
        inputs=na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=2,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=na.Cartesian2dVectorArray(51, 41),
        ),
        outputs=na.ScalarArray(np.full((51, 41), 3.0), axes=("x", "y")),
    )
    result = f.integrate(("x", "y"))
    assert not result.outputs.shape
    assert np.allclose(result.outputs, 12.0)


def test_integrate_vector_vertex():
    # 2D vertex grid (edges 51x41, cells 50x40), constant 3 over [0,2]^2 -> 12.
    f = na.FunctionArray(
        inputs=na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=2,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=na.Cartesian2dVectorArray(51, 41),
        ),
        outputs=na.ScalarArray(np.full((50, 40), 3.0), axes=("x", "y")),
    )
    result = f.integrate(("x", "y"))
    assert not result.outputs.shape
    assert np.allclose(result.outputs, 12.0)


def test_integrate_curved_grid():
    # volume_cell-based measure handles non-rectilinear grids correctly.
    # A 2x2 square rotated 30 degrees still has area 4.
    v = na.Cartesian2dVectorLinearSpace(
        start=0,
        stop=2,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=na.Cartesian2dVectorArray(51, 51),
    ).explicit
    theta = np.deg2rad(30)
    rotated = na.Cartesian2dVectorArray(
        x=v.x * np.cos(theta) - v.y * np.sin(theta),
        y=v.x * np.sin(theta) + v.y * np.cos(theta),
    ).broadcasted.explicit
    f = na.FunctionArray(
        inputs=rotated,
        outputs=na.ScalarArray(np.full((51, 51), 3.0), axes=("x", "y")),
    )
    # cells may be signed depending on traversal direction; compare magnitude.
    assert np.isclose(abs(float(f.integrate(("x", "y")).outputs.ndarray)), 12.0)


def test_integrate_component_str():
    # component='x' integrates only the x sub-coordinate of a vector input,
    # keeping the other dimension.
    f = na.FunctionArray(
        inputs=na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=2,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=na.Cartesian2dVectorArray(101, 51),
        ),
        outputs=na.ScalarArray(np.full((101, 51), 3.0), axes=("x", "y")),
    )
    result = f.integrate("x", component="x")
    assert "x" not in result.outputs.shape
    assert "y" in result.outputs.shape
    assert isinstance(result.inputs, na.Cartesian2dVectorArray)
    assert np.allclose(result.outputs, 6.0)


def test_integrate_mixed_axes():
    # mix vertex (x: inputs N+1=11, outputs N=10) and center (y: inputs N=11, outputs N=11)
    # in one integrate call. Constant integrand 3 over [0,2]^2 -> 12.
    f = na.FunctionArray(
        inputs=na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=2,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=na.Cartesian2dVectorArray(11, 11),
        ),
        outputs=na.ScalarArray(np.full((10, 11), 3.0), axes=("x", "y")),
    )
    result = f.integrate(("x", "y"))
    assert not result.outputs.shape
    assert np.isclose(float(result.outputs.ndarray), 12.0)


def test_integrate_invalid():
    f_sca = na.FunctionArray(
        inputs=na.ScalarLinearSpace(0, 1, axis="y", num=5),
        outputs=na.ScalarArray(np.ones(5), axes=("y",)),
    )

    with pytest.raises(ValueError, match="not in"):
        f_sca.integrate("z")
