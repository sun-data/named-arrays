from typing import Any
import math
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ndoptimize

_axis = "velocity"

_unit_x = u.km / u.s

_velocity = na.linspace(-100, 100, axis=_axis, num=41) * _unit_x
"""The independent variable of the test spectra."""

_correlation = [1, 0.19, -0.05]
"""The correlation of the noise of neighboring points of the test spectra."""

_sigma = 0.3 * u.DN
"""The standard deviation of the noise of the test spectra."""

_names = ndoptimize.models.gaussian_linear.parameters

_truth = na.CartesianNdVectorArray(
    dict(
        amplitude=na.ScalarArray(np.array([[4.0, 6, 8], [5, 7, 9]]), ("x", "y")) * u.DN,
        center=na.ScalarArray(np.array([-10.0, 0, 5]), "y") * _unit_x,
        width=na.ScalarArray(np.array([12.0, 18]), "x") * _unit_x,
        slope=0.001 * u.DN / _unit_x,
        intercept=1 * u.DN,
    )
)
"""The true parameters of the test spectra, which vary along `x` and `y`."""

_guess = na.CartesianNdVectorArray(
    dict(
        amplitude=5 * u.DN,
        center=0 * _unit_x,
        width=15 * _unit_x,
        slope=0 * u.DN / _unit_x,
        intercept=1 * u.DN,
    )
)
"""The initial guess for the parameters, the same for every spectrum."""


def _gaussian_linear(
    velocity: na.AbstractScalar,
    parameters: na.AbstractVectorArray,
) -> na.AbstractScalar:
    """The model of :func:`ndoptimize.models.gaussian_linear` in named arrays."""
    p = parameters
    z = (velocity - p.center) / p.width
    return p.amplitude * np.exp(-z * z / 2) + p.slope * velocity + p.intercept


_data = na.random.normal(_gaussian_linear(_velocity, _truth), _sigma, seed=0)
"""The noisy test spectra."""


def _guess_with(**components: Any) -> na.CartesianNdVectorArray:
    """:data:`_guess` with some components replaced or added."""
    return na.CartesianNdVectorArray(_guess.components | components)


def _assert_equal(
    result: na.optimize.LeastSquaresResult,
    expected: ndoptimize.LeastSquaresResult,
    axes: tuple[str, ...],
    units: dict[str, None | u.UnitBase],
) -> None:
    """
    Check that the result of :func:`named_arrays.optimize.least_squares` is
    that of :func:`ndoptimize.least_squares` along the given batch axes, with
    the given units of the parameters.
    """
    assert tuple(result.parameters.components) == tuple(units)
    for k, name in enumerate(_names):
        p = result.parameters.components[name]
        assert na.unit(p) == units[name]
        assert np.array_equal(
            na.value(p).ndarray_aligned(axes),
            expected.parameters[..., k],
        )
        for q, column in enumerate(_names):
            c = result.covariance.components[name].components[column]
            unit = [unit for unit in (units[name], units[column]) if unit is not None]
            assert na.unit(c) == (math.prod(unit[1:], start=unit[0]) if unit else None)
            assert np.array_equal(
                na.value(c).ndarray_aligned(axes),
                expected.covariance[..., k, q],
                equal_nan=True,
            )
    for name in ("chi2", "num_points", "iterations", "converged"):
        assert np.array_equal(
            getattr(result, name).ndarray_aligned(axes),
            getattr(expected, name),
        )


def test_least_squares() -> None:
    """
    Fitting named arrays with units, partial bounds and priors, and
    correlated noise gives the same fits as :func:`ndoptimize.least_squares`
    on the equivalent plain arrays.
    """
    data = _data.copy()
    data[dict(x=0, y=1, velocity=slice(10, 13))] = np.nan * u.DN

    result = na.optimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=data,
        variance=np.square(_sigma),
        guess=_guess,
        axis=_axis,
        lower=na.CartesianNdVectorArray(dict(width=3 * _unit_x)),
        prior_mean=na.CartesianNdVectorArray(dict(center=0 * _unit_x)),
        prior_sigma=na.CartesianNdVectorArray(dict(center=10 * _unit_x)),
        correlation=_correlation,
    )

    axes = ("x", "y")
    infinity = np.full(5, np.inf)
    expected = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity.ndarray.to_value(_unit_x)[:, np.newaxis],
        data=data.ndarray_aligned(axes + (_axis,)).to_value(u.DN),
        variance=_sigma.to_value(u.DN) ** 2,
        guess=[5, 0, 15, 0, 1],
        lower=[-np.inf, -np.inf, 3, -np.inf, -np.inf],
        upper=infinity,
        prior_mean=0,
        prior_sigma=[np.inf, 10, np.inf, np.inf, np.inf],
        correlation=_correlation,
    )

    assert np.all(expected.converged)
    assert expected.num_points[0, 1] == _velocity.shape[_axis] - 3
    _assert_equal(
        result=result,
        expected=expected,
        axes=axes,
        units={name: _guess.components[name].unit for name in _names},
    )

    # the fits find the truth
    error = (result.parameters.center - _truth.center) / np.sqrt(
        result.covariance.center.center
    )
    assert np.all(np.abs(error) < 4)


def test_least_squares_units() -> None:
    """
    Bounds and priors in other units than the guess give the same fits, and
    a variance in a unit which is not that of the data squared raises an
    error.
    """
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=_data,
        variance=np.square(_sigma),
        guess=_guess,
        axis=_axis,
    )
    result = na.optimize.least_squares(
        prior_mean=na.CartesianNdVectorArray(dict(width=15 * _unit_x)),
        prior_sigma=na.CartesianNdVectorArray(dict(width=5 * _unit_x)),
        **kwargs,
    )
    result_converted = na.optimize.least_squares(
        prior_mean=na.CartesianNdVectorArray(dict(width=15000 * u.m / u.s)),
        prior_sigma=na.CartesianNdVectorArray(dict(width=5000 * u.m / u.s)),
        **kwargs,
    )
    assert np.allclose(result.parameters, result_converted.parameters)

    kwargs["variance"] = 1 * u.km**2
    with pytest.raises(u.UnitConversionError):
        na.optimize.least_squares(**kwargs)


def test_least_squares_dimensionless() -> None:
    """Fits of arrays without units give parameters without units."""
    guess = na.CartesianNdVectorArray(
        {name: na.value(c) for name, c in _guess.components.items()}
    )
    result = na.optimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=na.value(_velocity),
        data=na.value(_data),
        variance=_sigma.value**2,
        guess=guess,
        axis=_axis,
    )
    result_units = na.optimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=_data,
        variance=np.square(_sigma),
        guess=_guess,
        axis=_axis,
    )
    for name in _names:
        assert na.unit(result.parameters.components[name]) is None
        assert np.array_equal(
            result.parameters.components[name],
            na.value(result_units.parameters.components[name]),
        )
    assert na.unit(result.covariance.center.width) is None

    # a variance without a unit is dimensionless, like the data in percent
    result_percent = na.optimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=na.value(_velocity),
        data=(na.value(_data) << u.dimensionless_unscaled).to(u.percent),
        variance=_sigma.value**2,
        guess=na.CartesianNdVectorArray(
            dict(
                amplitude=500 * u.percent,
                center=0,
                width=15,
                slope=0 * u.percent,
                intercept=100 * u.percent,
            )
        ),
        axis=_axis,
    )
    assert np.allclose(result_percent.parameters.amplitude, result.parameters.amplitude)
    assert np.allclose(result_percent.parameters.center, result.parameters.center)


def test_least_squares_batch_parameters() -> None:
    """
    A batch axis which only the guess varies along gives a fit of the same
    data for each guess.
    """
    data = _data[dict(x=0, y=0)]
    guess = _guess_with(center=na.ScalarArray(np.array([-20.0, 20]), "trial") * _unit_x)

    result = na.optimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=data,
        variance=np.square(_sigma),
        guess=guess,
        axis=_axis,
    )

    assert result.parameters.center.shape == dict(trial=2)
    for i in range(2):
        result_i = na.optimize.least_squares(
            model=ndoptimize.models.gaussian_linear,
            x=_velocity,
            data=data,
            variance=np.square(_sigma),
            guess=guess[dict(trial=i)],
            axis=_axis,
        )
        for name in _names:
            assert np.array_equal(
                result.parameters.components[name][dict(trial=i)],
                result_i.parameters.components[name],
            )


def test_evaluate() -> None:
    """
    Evaluating a model for a vector of parameters with batch axes gives the
    model written in named arrays, in the unit asked for.
    """
    result = na.optimize.evaluate(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        parameters=_truth,
        axis=_axis,
        unit=u.DN,
    )
    expected = _gaussian_linear(_velocity, _truth)
    assert result.shape == expected.shape
    assert na.unit(result) == u.DN
    assert np.allclose(result, expected)

    result_plain = na.optimize.evaluate(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        parameters=_truth,
        axis=_axis,
    )
    assert na.unit(result_plain) is None
    assert np.array_equal(result_plain, na.value(result))


def test_least_squares_evaluate() -> None:
    """
    The best-fit model of a fit is the model of the data, with the
    chi-squared of the fit, and can be evaluated on a new grid.
    """
    variance = np.square(_sigma)
    result = na.optimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=_data,
        variance=variance,
        guess=_guess,
        axis=_axis,
    )

    f = result.evaluate()
    assert f.shape == _data.shape
    assert na.unit(f) == u.DN
    assert np.array_equal(
        f,
        na.optimize.evaluate(
            model=ndoptimize.models.gaussian_linear,
            x=_velocity,
            parameters=result.parameters,
            axis=_axis,
            unit=u.DN,
        ),
    )
    chi2 = (np.square(_data - f) / variance).sum(axis=_axis)
    assert np.allclose(chi2, result.chi2)

    velocity = na.linspace(-100, 100, axis="velocity_fine", num=101) * _unit_x
    f_fine = result.evaluate(x=velocity, axis="velocity_fine")
    assert f_fine.shape == dict(result.chi2.shape, velocity_fine=101)


_samples = na.random.normal(
    _gaussian_linear(_velocity, _truth),
    _sigma,
    dict(_distribution=3),
    seed=1,
)
"""Three more noisy realizations of the test spectra."""


def test_least_squares_uncertain() -> None:
    """
    Fitting uncertain data fits the nominal data and each of its samples,
    and the best-fit model is uncertain too.
    """
    data = na.UncertainScalarArray(_data, _samples)
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        variance=np.square(_sigma),
        guess=_guess,
        axis=_axis,
        prior_mean=na.CartesianNdVectorArray(dict(center=0 * _unit_x)),
        prior_sigma=na.CartesianNdVectorArray(dict(center=10 * _unit_x)),
    )

    result = na.optimize.least_squares(data=data, **kwargs)
    nominal = na.optimize.least_squares(data=data.nominal, **kwargs)
    distribution = na.optimize.least_squares(data=data.distribution, **kwargs)

    for name in _names:
        p = result.parameters.components[name]
        assert isinstance(p, na.UncertainScalarArray)
        assert np.array_equal(p.nominal, nominal.parameters.components[name])
        assert np.array_equal(p.distribution, distribution.parameters.components[name])
    c = result.covariance.center.width
    assert isinstance(c, na.UncertainScalarArray)
    assert np.array_equal(c.nominal, nominal.covariance.center.width)
    assert np.array_equal(c.distribution, distribution.covariance.center.width)
    for name in ("chi2", "num_points", "iterations", "converged"):
        r = getattr(result, name)
        assert isinstance(r, na.UncertainScalarArray)
        assert np.array_equal(r.nominal, getattr(nominal, name))
        assert np.array_equal(r.distribution, getattr(distribution, name))

    f = result.evaluate()
    assert isinstance(f, na.UncertainScalarArray)
    assert np.array_equal(f.nominal, nominal.evaluate())
    assert np.array_equal(f.distribution, distribution.evaluate())


def test_least_squares_uncertain_guess() -> None:
    """
    An uncertain component of a vector, which the dispatcher does not see,
    is fit like uncertain data: one fit for the nominal guess and one for each
    of its samples.
    """
    center = na.UncertainScalarArray(
        nominal=0 * _unit_x,
        distribution=na.ScalarArray(np.array([-20.0, 20]), "_distribution") * _unit_x,
    )
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=_data,
        variance=np.square(_sigma),
        axis=_axis,
    )

    result = na.optimize.least_squares(guess=_guess_with(center=center), **kwargs)

    nominal = na.optimize.least_squares(guess=_guess, **kwargs)
    distribution = na.optimize.least_squares(
        guess=_guess_with(center=center.distribution),
        **kwargs,
    )
    p = result.parameters.center
    assert isinstance(p, na.UncertainScalarArray)
    assert np.array_equal(p.nominal, nominal.parameters.center)
    assert np.array_equal(p.distribution, distribution.parameters.center)


def test_evaluate_uncertain() -> None:
    """Evaluating a model for uncertain parameters evaluates it for the
    nominal parameters and for each of their samples."""
    amplitude = na.UncertainScalarArray(
        nominal=6 * u.DN,
        distribution=na.ScalarArray(np.array([5.0, 6.5, 7]), "_distribution") * u.DN,
    )
    parameters = na.CartesianNdVectorArray(_truth.components | dict(amplitude=amplitude))
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        axis=_axis,
        unit=u.DN,
    )

    result = na.optimize.evaluate(parameters=parameters, **kwargs)

    assert isinstance(result, na.UncertainScalarArray)
    assert np.array_equal(
        result.nominal,
        na.optimize.evaluate(parameters=na.nominal(parameters), **kwargs),
    )
    assert np.allclose(
        result.distribution,
        _gaussian_linear(
            _velocity,
            na.CartesianNdVectorArray(
                _truth.components | dict(amplitude=amplitude.distribution)
            ),
        ),
    )


@pytest.mark.parametrize(
    argnames="kwargs,error",
    argvalues=[
        (dict(parameters=na.CartesianNdVectorArray(dict(amplitude=5 * u.DN))), ValueError),
        (
            dict(parameters=na.CartesianNdVectorArray(_truth.components | dict(center=_velocity))),
            ValueError,
        ),
        (
            dict(
                parameters=na.CartesianNdVectorArray(
                    _truth.components
                    | dict(center=na.Cartesian2dVectorArray(0 * _unit_x, 0 * _unit_x))
                )
            ),
            TypeError,
        ),
        (dict(axis="wavelength"), ValueError),
    ],
)
def test_evaluate_invalid(kwargs: dict[str, Any], error: type[Exception]) -> None:
    arguments: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        parameters=_truth,
        axis=_axis,
    )
    with pytest.raises(error):
        na.optimize.evaluate(**(arguments | kwargs))


@ndoptimize.model(parameters=("amplitude", "x0", "y0", "width"))
def _gaussian_2d(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    """A circular two-dimensional Gaussian of the two inputs."""
    amplitude, x0, y0, width = parameters[0], parameters[1], parameters[2], parameters[3]
    for i in range(x.shape[0]):
        dx = (x[i, 0] - x0) / width
        dy = (x[i, 1] - y0) / width
        e = math.exp(-(dx * dx + dy * dy) / 2)
        f[i] = amplitude * e
        jacobian[i, 0] = e
        jacobian[i, 1] = amplitude * e * dx / width
        jacobian[i, 2] = amplitude * e * dy / width
        jacobian[i, 3] = amplitude * e * (dx * dx + dy * dy) / width


def test_least_squares_vector() -> None:
    """
    A vector independent variable gives the model its components as inputs,
    and the points of a fit may vary along more than one axis.
    """
    position = na.Cartesian2dVectorArray(
        x=na.linspace(-5, 5, axis="px", num=11) * u.arcsec,
        y=na.linspace(-4, 4, axis="py", num=9) * u.arcsec,
    )
    truth = na.CartesianNdVectorArray(
        dict(
            amplitude=10 * u.DN,
            x0=na.ScalarArray(np.array([-1.0, 0.5, 2]), "frame") * u.arcsec,
            y0=0.3 * u.arcsec,
            width=1.5 * u.arcsec,
        )
    )
    r2 = np.square(position.x - truth.x0) + np.square(position.y - truth.y0)
    f = truth.amplitude * np.exp(-r2 / (2 * np.square(truth.width)))
    data = na.random.normal(f, 0.5 * u.DN, seed=1)

    f_model = na.optimize.evaluate(
        model=_gaussian_2d,
        x=position,
        parameters=truth,
        axis=("px", "py"),
        unit=u.DN,
    )
    assert np.allclose(f_model, f)

    # the compiled model is its Python source
    x_test = np.array([[0.5, -1.0], [2.0, 0.3]])
    p_test = np.array([10, 0.5, 0.3, 1.5])
    f_compiled, f_python = np.empty(2), np.empty(2)
    jacobian_compiled, jacobian_python = np.empty((2, 4)), np.empty((2, 4))
    _gaussian_2d(x_test, p_test, f_compiled, jacobian_compiled)
    _gaussian_2d.function.py_func(x_test, p_test, f_python, jacobian_python)  # type: ignore[attr-defined]
    assert np.allclose(f_compiled, f_python)
    assert np.allclose(jacobian_compiled, jacobian_python)
    guess = na.CartesianNdVectorArray(
        dict(amplitude=8 * u.DN, x0=0 * u.arcsec, y0=0 * u.arcsec, width=2 * u.arcsec)
    )

    kwargs: dict[str, Any] = dict(
        model=_gaussian_2d,
        x=position,
        data=data,
        variance=0.25 * u.DN**2,
        guess=guess,
        axis=("px", "py"),
    )
    result = na.optimize.least_squares(**kwargs)

    points = ("px", "py")
    x = np.stack(
        [
            position.x.broadcast_to(position.shape).ndarray_aligned(points).value,
            position.y.broadcast_to(position.shape).ndarray_aligned(points).value,
        ],
        axis=~0,
    ).reshape(-1, 2)
    expected = ndoptimize.least_squares(
        model=_gaussian_2d,
        x=x,
        data=data.ndarray_aligned(("frame",) + points).value.reshape(3, -1),
        variance=0.25,
        guess=[8, 0, 0, 2],
    )
    assert np.all(expected.converged)
    for k, name in enumerate(_gaussian_2d.parameters):
        assert np.array_equal(
            result.parameters.components[name].ndarray_aligned(("frame",)).value,
            expected.parameters[..., k],
        )

    with pytest.raises(ValueError, match="single axis"):
        na.optimize.least_squares(correlation=_correlation, **kwargs)


@pytest.mark.parametrize(
    argnames="kwargs,error",
    argvalues=[
        (dict(guess=na.CartesianNdVectorArray(dict(amplitude=5 * u.DN))), ValueError),
        (dict(guess=_guess_with(extra=1)), ValueError),
        (dict(lower=na.CartesianNdVectorArray(dict(sigma=3 * _unit_x))), ValueError),
        (dict(lower=3 * _unit_x), TypeError),
        (dict(guess=_guess_with(center=_velocity)), ValueError),
        (dict(axis="wavelength"), ValueError),
        (dict(data=na.Cartesian2dVectorArray(_data, _data)), TypeError),
    ],
)
def test_least_squares_invalid(kwargs: dict[str, Any], error: type[Exception]) -> None:
    arguments: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity,
        data=_data,
        variance=np.square(_sigma),
        guess=_guess,
        axis=_axis,
    )
    with pytest.raises(error):
        na.optimize.least_squares(**(arguments | kwargs))
