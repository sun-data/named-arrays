"""
Tests of resampling a function array at new inputs which are not a grid with
the same axes as its own, such as scattered points.

Multilinear interpolation reproduces any affine function exactly, even on a
curvilinear grid, so each function here is affine in its inputs and the
expected outputs are the same function evaluated at the new inputs.
"""

from typing import Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

_x = na.linspace(-1, 1, axis="x", num=11)
_y = na.linspace(-1, 1, axis="y", num=13)
_z = na.linspace(-1, 1, axis="z", num=5)
_channel = na.ScalarArray(np.array([1.0, 2.0, 3.0]), axes="channel")
_wavelength = na.linspace(500, 600, axis="wavelength", num=4)


def _affine(inputs: na.AbstractArray) -> na.AbstractScalar:
    """
    An affine function of a scalar, or of the components of a vector.

    Parameters
    ----------
    inputs
        The point at which to evaluate the function.
    """
    if isinstance(inputs, na.AbstractVectorArray):
        components = inputs.cartesian_nd.components
        return sum(
            (i + 2) * components[c]
            for i, c in enumerate(components)
        ) + 0.5
    return 2 * inputs + 0.5


def _points(
    shape: dict[str, int],
    components: int,
    seed: int = 0,
) -> na.AbstractArray:
    """
    Random points inside the square from -1 to 1.

    Parameters
    ----------
    shape
        The axes along which the points are listed.
    components
        The number of components of each point, or zero for a scalar.
    seed
        The seed of the random number generator.
    """
    def coordinate(i: int) -> na.AbstractScalar:
        return na.random.uniform(-1, 1, shape_random=shape, seed=seed + i)

    if components == 0:
        return coordinate(0)
    if components == 2:
        return na.Cartesian2dVectorArray(coordinate(0), coordinate(1))
    return na.Cartesian3dVectorArray(coordinate(0), coordinate(1), coordinate(2))


_rectilinear = na.Cartesian2dVectorArray(_x, _y)

_angle = 0.3
_curvilinear = na.Cartesian2dVectorArray(
    x=1.3 * (_x * np.cos(_angle) - _y * np.sin(_angle)) + 0.05 * _y * _y,
    y=1.3 * (_x * np.sin(_angle) + _y * np.cos(_angle)) + 0.05 * _x * _x,
)

_spectral = na.SpectralPositionalVectorArray(_wavelength, _rectilinear)


@pytest.mark.parametrize(
    argnames="inputs,points,axis",
    argvalues=[
        (_x, _points(dict(ray=7), components=0), "x"),
        (_x, _points(dict(ray=7), components=0), None),
        (_rectilinear, _points(dict(ray=7), components=2), ("x", "y")),
        (_rectilinear, _points(dict(ray=7), components=2), None),
        (_curvilinear, _points(dict(ray=7), components=2), ("x", "y")),
        (_rectilinear, _points(dict(ray_x=3, ray_y=4), components=2), ("x", "y")),
        (_rectilinear, _points(dict(), components=2), ("x", "y")),
        (
            na.Cartesian3dVectorArray(_x, _y, _z),
            _points(dict(ray=7), components=3),
            ("x", "y", "z"),
        ),
    ],
)
def test_regrid_scattered(
    inputs: na.AbstractArray,
    points: na.AbstractArray,
    axis: None | str | tuple[str, ...],
) -> None:
    """
    A function sampled on a grid can be evaluated at points listed along
    any number of axes which the grid doesn't have, including none.
    """
    f = na.FunctionArray(
        inputs=inputs,
        outputs=_affine(inputs).broadcast_to(na.shape(inputs)),
    )

    result = f(points, axis=axis)

    assert np.all(result.inputs == points)
    assert np.allclose(result.outputs, _affine(points))
    assert result.outputs.shape == na.shape(points)


def test_regrid_scattered_batch_function() -> None:
    """
    An axis which only the outputs of the function have is broadcast to the
    points, which do not have it.
    """
    f = na.FunctionArray(
        inputs=_rectilinear,
        outputs=_channel * _affine(_rectilinear),
    )
    points = _points(dict(ray=7), components=2)

    result = f(points, axis=("x", "y"))

    assert np.all(result.inputs == points)
    assert np.allclose(result.outputs, _channel * _affine(points))
    assert result.outputs.shape == dict(channel=3, ray=7)


def test_regrid_scattered_batch_shared() -> None:
    """
    An axis which both the function and the points have gives a separate
    set of points for each of its elements.
    """
    f = na.FunctionArray(
        inputs=_rectilinear,
        outputs=_channel * _affine(_rectilinear),
    )
    points = 0.5 * _points(dict(ray=7), components=2) + 0.1 * _channel - 0.2

    result = f(points, axis=("x", "y"))

    assert np.all(result.inputs == points)
    assert np.allclose(result.outputs, _channel * _affine(points))
    assert result.outputs.shape == dict(channel=3, ray=7)


@pytest.mark.parametrize(
    argnames="points",
    argvalues=[
        _points(dict(ray=7), components=2),
        0.5 * _points(dict(ray=7), components=2) + 0.1 * _channel - 0.2,
    ],
)
def test_regrid_scattered_batch_grids(
    points: na.AbstractVectorArray,
) -> None:
    """
    A function sampled on a different grid for each element of an axis can
    be evaluated at points which share the axis or don't.
    """
    inputs = na.Cartesian2dVectorArray(_channel * _x, _y)
    f = na.FunctionArray(
        inputs=inputs,
        outputs=_affine(inputs),
    )

    result = f(points, axis=("x", "y"))

    assert np.all(result.inputs == points)
    assert np.allclose(result.outputs, _affine(points))
    assert result.outputs.shape == dict(ray=7, channel=3)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[None, _wavelength],
)
def test_regrid_scattered_unresampled(
    wavelength: None | na.AbstractScalar,
) -> None:
    """
    A component which is not resampled can be left out using `None`, or
    given as the same component of the inputs of the function, and the
    result keeps it.
    """
    f = na.FunctionArray(
        inputs=_spectral,
        outputs=_affine(_spectral),
    )
    points = _points(dict(ray=7), components=2)

    result = f(
        na.SpectralPositionalVectorArray(wavelength, points),
        axis=("x", "y"),
    )

    inputs_expected = na.SpectralPositionalVectorArray(_wavelength, points)
    assert np.all(result.inputs == inputs_expected)
    assert np.allclose(result.outputs, _affine(inputs_expected))
    assert result.outputs.shape == dict(wavelength=4, ray=7)


def test_regrid_scattered_units() -> None:
    """
    The units of the inputs and outputs carry through.
    """
    f = na.FunctionArray(
        inputs=_rectilinear * u.mm,
        outputs=_affine(_rectilinear) * u.nm,
    )
    points = _points(dict(ray=7), components=2)

    result = f(points * u.mm, axis=("x", "y"))

    assert np.all(result.inputs == points * u.mm)
    assert np.allclose(result.outputs, _affine(points) * u.nm)


def test_regrid_scattered_weights() -> None:
    """
    Weights computed in advance give the same result as computing them on
    the fly.
    """
    f = na.FunctionArray(
        inputs=_curvilinear,
        outputs=_affine(_curvilinear),
    )
    points = _points(dict(ray=7), components=2)

    weights = f.weights(points, axis=("x", "y"))
    result = f(points, axis=("x", "y"), weights=weights)

    assert np.all(result.outputs == f(points, axis=("x", "y")).outputs)
    assert np.allclose(result.outputs, _affine(points))


_outside = na.Cartesian2dVectorArray(
    x=na.ScalarArray(np.array([0.5, 1.5, -2.0]), axes="ray"),
    y=na.ScalarArray(np.array([0.25, 0.0, 0.5]), axes="ray"),
)


@pytest.mark.parametrize(
    argnames="bounds,result_expected",
    argvalues=[
        ("extrapolate", _affine(_outside)),
        ("nan", np.where(np.abs(_outside.x) <= 1, _affine(_outside), np.nan)),
    ],
)
def test_regrid_scattered_bounds(
    bounds: str,
    result_expected: na.AbstractScalar,
) -> None:
    """
    Points outside of the grid are extrapolated or set to NaN.
    """
    f = na.FunctionArray(
        inputs=_rectilinear,
        outputs=_affine(_rectilinear),
    )

    result = f(_outside, axis=("x", "y"), bounds=bounds)

    assert np.allclose(result.outputs, result_expected, equal_nan=True)


def test_regrid_scattered_bounds_raise() -> None:
    """
    Points outside of the grid raise an error if asked to.
    """
    f = na.FunctionArray(
        inputs=_rectilinear,
        outputs=_affine(_rectilinear),
    )

    with pytest.raises(ValueError, match="2 of the output points fall outside"):
        f(_outside, axis=("x", "y"), bounds="raise")


@pytest.mark.parametrize(
    argnames="call,match",
    argvalues=[
        (
            lambda: na.FunctionArray(_rectilinear, _affine(_rectilinear))(
                _points(dict(ray=7), components=2),
                axis=("x", "z"),
            ),
            r"do not vary along \('z',\)",
        ),
        (
            lambda: na.FunctionArray(_rectilinear, _affine(_rectilinear))(
                na.Cartesian2dVectorArray(_points(dict(ray=7), components=0), None),
                axis=("x", "y"),
            ),
            "marked separable using `None`",
        ),
        (
            lambda: na.FunctionArray(
                inputs=na.Cartesian2dVectorArray(_channel, _y),
                outputs=_channel * _y,
            )(
                _points(dict(ray=7), components=2),
                axis="y",
            ),
            r"The component 'x' of the new inputs varies along \('ray',\)",
        ),
        (
            lambda: na.FunctionArray(_spectral, _affine(_spectral))(
                na.SpectralPositionalVectorArray(
                    _wavelength + 10,
                    _points(dict(ray=7), components=2),
                ),
                axis=("x", "y"),
            ),
            "The component 'wavelength' of the new inputs is not resampled",
        ),
        (
            lambda: na.FunctionArray(_spectral, _affine(_spectral))(
                na.SpectralPositionalVectorArray(
                    550,
                    _points(dict(ray=7), components=2),
                ),
                axis=("x", "y"),
            ),
            "The component 'wavelength' of the new inputs is not resampled",
        ),
        (
            lambda: na.FunctionArray(
                inputs=na.Cartesian2dVectorArray(_channel * _x, _y),
                outputs=_affine(na.Cartesian2dVectorArray(_channel * _x, _y)),
            )(
                _points(dict(ray=7), components=2),
            ),
            "which vary along `axis`, 2, should be equal to the number of "
            "axes in `axis`",
        ),
        (
            lambda: na.FunctionArray(_curvilinear, _affine(_curvilinear))(
                _points(dict(ray=7), components=2),
                axis="x",
            ),
            r"which vary along `axis`, 2, should be equal to the number of "
            r"axes in `axis`, \('x',\)",
        ),
    ],
)
def test_regrid_errors(
    call: Callable[[], na.AbstractFunctionArray],
    match: str,
) -> None:
    """
    An axis which the inputs don't vary along can't be resampled, a
    component which is resampled can't be left out, a component which isn't
    resampled can't vary from one point to the next or differ from the
    function's own, and there must be as many components to resample as
    axes in `axis`.
    """
    with pytest.raises(ValueError, match=match):
        call()
