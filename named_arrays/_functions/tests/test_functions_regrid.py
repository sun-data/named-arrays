"""
Tests of resampling a function array at new inputs which are not a grid with
the same axes as its own, such as scattered points.

Multilinear interpolation reproduces any affine function exactly, even on a
curvilinear grid, so each function here is affine in its inputs and the
expected outputs are the same function evaluated at the new inputs.
"""

import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na

__all__ = []

_x = na.linspace(-1, 1, axis="x", num=11)
_y = na.linspace(-1, 1, axis="y", num=13)
_z = na.linspace(-1, 1, axis="z", num=5)
_channel = na.ScalarArray(np.array([1.0, 2.0, 3.0]), axes="channel")
_wavelength = na.linspace(500, 600, axis="wavelength", num=4)
_time = na.ScalarArray(
    ndarray=astropy.time.Time(["2019-09-30T18:00:00", "2019-09-30T18:00:10"]),
    axes="time",
)


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

_per_channel = na.Cartesian2dVectorArray(_channel * _x, _y)

_outside = na.Cartesian2dVectorArray(
    x=na.ScalarArray(np.array([0.5, 1.5, -2.0]), axes="ray"),
    y=na.ScalarArray(np.array([0.25, 0.0, 0.5]), axes="ray"),
)

# An uncertain offset with hand-picked samples, and a shift along x by it.
_uncertain = na.UncertainScalarArray(
    nominal=na.ScalarArray(0.0),
    distribution=na.ScalarArray(np.array([-0.1, 0.0, 0.2]), axes="_distribution"),
)
_shift = na.Cartesian2dVectorArray(_uncertain, 0)


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


@pytest.mark.parametrize(
    argnames="unit",
    argvalues=[1, u.mm],
)
def test_regrid_scattered_number(
    unit: int | u.UnitBase,
) -> None:
    """
    A single point can be a plain number or quantity, if the inputs of the
    function are scalars.
    """
    f = na.FunctionArray(
        inputs=_x * unit,
        outputs=_affine(_x),
    )
    point = 0.3 * unit

    result = f(point)

    assert np.all(result.inputs == na.ScalarArray(point))
    assert np.allclose(result.outputs, _affine(na.ScalarArray(0.3)))
    assert result.outputs.shape == dict()


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
    f = na.FunctionArray(
        inputs=_per_channel,
        outputs=_affine(_per_channel),
    )

    result = f(points, axis=("x", "y"))

    assert np.all(result.inputs == points)
    assert np.allclose(result.outputs, _affine(points))
    assert result.outputs.shape == dict(ray=7, channel=3)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        None,
        _wavelength,
        _wavelength.broadcast_to(dict(wavelength=4, ray=7)),
        _wavelength + 0 * _x,
    ],
)
def test_regrid_scattered_unresampled(
    wavelength: None | na.AbstractScalar,
) -> None:
    """
    A component which is not resampled can be left out using `None`, or
    given equal to the same component of the inputs of the function, even if
    it is broadcast along other axes, and the result keeps the component of
    the function.
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
    assert na.shape(result.inputs) == dict(wavelength=4, ray=7)
    assert np.allclose(result.outputs, _affine(inputs_expected))
    assert result.outputs.shape == dict(wavelength=4, ray=7)


def test_regrid_scattered_unresampled_nan() -> None:
    """
    A component which is not resampled is equal to the same component of the
    inputs of the function where both are NaN.
    """
    wavelength = na.ScalarArray(
        ndarray=np.array([500.0, np.nan, 550.0, 600.0]),
        axes="wavelength",
    )
    inputs = na.SpectralPositionalVectorArray(wavelength, _rectilinear)
    f = na.FunctionArray(
        inputs=inputs,
        outputs=_affine(inputs),
    )
    points = _points(dict(ray=7), components=2)

    result = f(
        na.SpectralPositionalVectorArray(wavelength + 0, points),
        axis=("x", "y"),
    )

    inputs_expected = na.SpectralPositionalVectorArray(wavelength, points)
    assert np.allclose(result.inputs.wavelength, wavelength, equal_nan=True)
    assert np.all(result.inputs.position == points)
    assert np.allclose(result.outputs, _affine(inputs_expected), equal_nan=True)
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


@pytest.mark.parametrize(
    argnames="inputs,outputs,points,axis,result_expected",
    argvalues=[
        (
            _x,
            _affine(_x),
            _points(dict(ray=7), components=0) + _uncertain,
            "x",
            _affine(_points(dict(ray=7), components=0) + _uncertain),
        ),
        (
            _curvilinear,
            _affine(_curvilinear),
            _points(dict(ray=7), components=2) + _shift,
            ("x", "y"),
            _affine(_points(dict(ray=7), components=2) + _shift),
        ),
        (
            _rectilinear + _shift,
            _affine(_rectilinear),
            _points(dict(ray=7), components=2),
            ("x", "y"),
            _affine(_points(dict(ray=7), components=2) - _shift),
        ),
        (
            _rectilinear,
            _affine(_rectilinear) + _uncertain,
            _points(dict(ray=7), components=2),
            ("x", "y"),
            _affine(_points(dict(ray=7), components=2)) + _uncertain,
        ),
        (
            _rectilinear + _shift,
            _affine(_rectilinear) + _uncertain,
            _points(dict(ray=7), components=2) + _shift,
            ("x", "y"),
            _affine(_points(dict(ray=7), components=2)) + _uncertain,
        ),
    ],
)
def test_regrid_uncertain(
    inputs: na.AbstractArray,
    outputs: na.AbstractScalar,
    points: na.AbstractArray,
    axis: str | tuple[str, ...],
    result_expected: na.AbstractScalar,
) -> None:
    """
    If the new inputs, the inputs of the function, or its outputs are
    uncertain, each sample is resampled with the samples of the others which
    go with it.
    A grid which is shifted by an uncertain offset, with outputs which are
    not, is the function shifted by the same offset.
    """
    f = na.FunctionArray(
        inputs=inputs,
        outputs=outputs,
    )

    result = f(points, axis=axis)

    assert np.all(result.inputs == points)
    assert isinstance(result.outputs, na.UncertainScalarArray)
    assert np.allclose(result.outputs, result_expected)


def test_regrid_uncertain_bounds() -> None:
    """
    Each sample of uncertain points outside of the grid is set to NaN if
    asked to.
    """
    f = na.FunctionArray(
        inputs=_rectilinear,
        outputs=_affine(_rectilinear),
    )
    points = _outside + _shift

    result = f(points, axis=("x", "y"), bounds="nan")

    nan = na.ScalarArray(np.array([1, np.nan, np.nan]), axes="ray")
    result_expected = _affine(points) * nan
    assert isinstance(result.outputs, na.UncertainScalarArray)
    assert np.allclose(result.outputs, result_expected, equal_nan=True)


def test_regrid_uncertain_conservative() -> None:
    """
    The conservative method resamples each sample of uncertain vertices on
    its own.
    """
    vertices = na.Cartesian2dVectorArray(
        x=na.linspace(-1, 1, axis="x", num=12),
        y=na.linspace(-1, 1, axis="y", num=14),
    )
    outputs = _affine(vertices).cell_centers()
    f = na.FunctionArray(
        inputs=vertices + _shift,
        outputs=outputs,
    )
    vertices_new = 0.5 * vertices

    result = f(vertices_new, axis=("x", "y"), method="conservative")

    def resample(shift: float | na.AbstractScalar) -> na.AbstractScalar:
        f_shifted = na.FunctionArray(vertices + na.Cartesian2dVectorArray(shift, 0), outputs)
        return f_shifted(vertices_new, axis=("x", "y"), method="conservative").outputs

    samples = _uncertain.distribution
    result_expected = na.UncertainScalarArray(
        nominal=resample(0),
        distribution=na.stack(
            [resample(samples[dict(_distribution=i)]) for i in range(samples.shape["_distribution"])],
            axis="_distribution",
        ),
    )
    assert np.allclose(result.outputs, result_expected)
    assert not np.allclose(result_expected.distribution, result_expected.nominal)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        None,
        _wavelength + _uncertain,
    ],
)
def test_regrid_uncertain_unresampled(
    wavelength: None | na.AbstractScalar,
) -> None:
    """
    An uncertain component which is not resampled is kept, and the outputs
    which depend on it are uncertain.
    """
    inputs = na.SpectralPositionalVectorArray(_wavelength + _uncertain, _rectilinear)
    f = na.FunctionArray(
        inputs=inputs,
        outputs=_affine(inputs),
    )
    points = _points(dict(ray=7), components=2)

    result = f(
        na.SpectralPositionalVectorArray(wavelength, points),
        axis=("x", "y"),
    )

    inputs_expected = na.SpectralPositionalVectorArray(_wavelength + _uncertain, points)
    assert np.all(result.inputs == inputs_expected)
    assert np.allclose(result.outputs, _affine(inputs_expected))
    assert result.outputs.shape == dict(wavelength=4, ray=7)


@pytest.mark.parametrize(
    argnames="inputs,points",
    argvalues=[
        (_rectilinear + _shift, _points(dict(ray=7), components=2)),
        (_rectilinear, _points(dict(ray=7), components=2) + _shift),
    ],
)
def test_regrid_uncertain_weights_raise(
    inputs: na.AbstractVectorArray,
    points: na.AbstractVectorArray,
) -> None:
    """
    Weights can't be computed in advance for uncertain inputs, or used to
    resample them, since each of their samples has weights of its own.
    """
    f = na.FunctionArray(
        inputs=inputs,
        outputs=_affine(_rectilinear),
    )
    weights = na.FunctionArray(_rectilinear, f.outputs).weights(
        _points(dict(ray=7), components=2),
        axis=("x", "y"),
    )

    with pytest.raises(ValueError, match="cannot be computed in advance for uncertain"):
        f.weights(points, axis=("x", "y"))
    with pytest.raises(ValueError, match="cannot be used to resample uncertain"):
        f(points, axis=("x", "y"), weights=weights)


@pytest.mark.parametrize(
    argnames="inputs,outputs,points,bounds,result_expected",
    argvalues=[
        (
            _curvilinear,
            _affine(_curvilinear),
            _points(dict(ray=7), components=2),
            "extrapolate",
            _affine(_points(dict(ray=7), components=2)),
        ),
        (
            _rectilinear,
            _channel * _affine(_rectilinear),
            0.5 * _points(dict(ray=7), components=2) + 0.1 * _channel - 0.2,
            "extrapolate",
            _channel * _affine(0.5 * _points(dict(ray=7), components=2) + 0.1 * _channel - 0.2),
        ),
        (
            _rectilinear,
            _affine(_rectilinear),
            _outside,
            "nan",
            np.where(np.abs(_outside.x) <= 1, _affine(_outside), np.nan),
        ),
        (
            _rectilinear,
            _affine(_rectilinear) + _uncertain,
            _points(dict(ray=7), components=2),
            "extrapolate",
            _affine(_points(dict(ray=7), components=2)) + _uncertain,
        ),
    ],
)
def test_regrid_scattered_weights(
    inputs: na.AbstractArray,
    outputs: na.AbstractScalar,
    points: na.AbstractArray,
    bounds: str,
    result_expected: na.AbstractScalar,
) -> None:
    """
    Weights computed in advance give the same result as computing them on
    the fly.
    """
    f = na.FunctionArray(
        inputs=inputs,
        outputs=outputs,
    )

    weights = f.weights(points, axis=("x", "y"), bounds=bounds)
    result = f(points, axis=("x", "y"), weights=weights)

    result_direct = f(points, axis=("x", "y"), bounds=bounds)
    assert np.allclose(result.outputs, result_direct.outputs, equal_nan=True)
    assert np.allclose(result.outputs, result_expected, equal_nan=True)


def test_regrid_conservative_batch() -> None:
    """
    The conservative method can resample onto a new grid which varies along
    an axis which only the outputs of the function have, and the result is
    the same as resampling each element of that axis on its own.
    """
    vertices = na.Cartesian2dVectorArray(
        x=na.linspace(-1, 1, axis="x", num=12),
        y=na.linspace(-1, 1, axis="y", num=14),
    )
    f = na.FunctionArray(
        inputs=vertices,
        outputs=_channel * _affine(vertices).cell_centers(),
    )
    vertices_new = 0.5 * vertices + 0.01 * _channel

    result = f(vertices_new, axis=("x", "y"), method="conservative")

    assert np.all(result.inputs == vertices_new)
    assert result.outputs.shape == dict(channel=3, x=11, y=13)
    for i in range(_channel.shape["channel"]):
        index = dict(channel=i)
        f_i = na.FunctionArray(f.inputs, f.outputs[index])
        result_i = f_i(vertices_new[index], axis=("x", "y"), method="conservative")
        assert np.allclose(result.outputs[index], result_i.outputs)


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
    argnames="inputs,outputs,inputs_new,axis,match",
    argvalues=[
        (
            _rectilinear,
            _affine(_rectilinear),
            _points(dict(ray=7), components=2),
            ("x", "z"),
            r"do not vary along \('z',\)",
        ),
        (
            _rectilinear,
            _affine(_rectilinear),
            na.Cartesian2dVectorArray(_points(dict(ray=7), components=0), None),
            ("x", "y"),
            "marked separable using `None`",
        ),
        (
            na.Cartesian2dVectorArray(_channel, _y),
            _channel * _y,
            _points(dict(ray=7), components=2),
            "y",
            r"The component 'x' of the new inputs varies along \('ray',\)",
        ),
        (
            _spectral,
            _affine(_spectral),
            na.SpectralPositionalVectorArray(
                _wavelength + 10,
                _points(dict(ray=7), components=2),
            ),
            ("x", "y"),
            "The component 'wavelength' of the new inputs is not resampled",
        ),
        (
            _spectral,
            _affine(_spectral),
            na.SpectralPositionalVectorArray(
                550,
                _points(dict(ray=7), components=2),
            ),
            ("x", "y"),
            "The component 'wavelength' of the new inputs is not resampled",
        ),
        (
            na.SpectralPositionalVectorArray(_wavelength * u.nm, _rectilinear),
            _affine(_spectral),
            na.SpectralPositionalVectorArray(
                550,
                _points(dict(ray=7), components=2),
            ),
            ("x", "y"),
            "The component 'wavelength' of the new inputs is not resampled",
        ),
        (
            na.TemporalPositionalVectorArray(_time, _rectilinear),
            _affine(_rectilinear) * na.ScalarArray(np.ones(2), axes="time"),
            na.TemporalPositionalVectorArray(
                na.ScalarArray(
                    ndarray=astropy.time.Time(["2019-09-30T18:00:00", "2019-09-30T18:00:20"]),
                    axes="time",
                ),
                _points(dict(ray=7), components=2),
            ),
            ("x", "y"),
            "The component 'time' of the new inputs is not resampled",
        ),
        (
            _per_channel,
            _affine(_per_channel),
            _points(dict(ray=7), components=2),
            None,
            "which vary along `axis`, 2, should be equal to the number of "
            "axes in `axis`",
        ),
        (
            _curvilinear,
            _affine(_curvilinear),
            _points(dict(ray=7), components=2),
            "x",
            r"which vary along `axis`, 2, should be equal to the number of "
            r"axes in `axis`, \('x',\)",
        ),
    ],
)
def test_regrid_errors(
    inputs: na.AbstractArray,
    outputs: na.AbstractScalar,
    inputs_new: na.AbstractArray,
    axis: None | str | tuple[str, ...],
    match: str,
) -> None:
    """
    An axis which the inputs don't vary along can't be resampled, a
    component which is resampled can't be left out, a component which isn't
    resampled must be `None` or equal to the function's own, and there must
    be as many components to resample as axes in `axis`.
    """
    f = na.FunctionArray(inputs, outputs)
    with pytest.raises(ValueError, match=match):
        f(inputs_new, axis=axis)
