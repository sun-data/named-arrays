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
    velocity: float | u.Quantity | na.AbstractScalar,
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


def _fit(**kwargs: Any) -> na.ModelFitFunctionArray:
    """A fit of the test spectra, with some fields replaced or added."""
    arguments: dict[str, Any] = dict(
        inputs=_velocity,
        outputs=_data,
        model=ndoptimize.models.gaussian_linear,
        guess=_guess,
        variance=np.square(_sigma),
        axis_model=_axis,
    )
    return na.ModelFitFunctionArray(**(arguments | kwargs))


def _assert_equal(
    result: na.ModelFitFunctionArray,
    expected: ndoptimize.LeastSquaresResult,
    axes: tuple[str, ...],
    units: dict[str, None | u.UnitBase],
) -> None:
    """
    Check that the results of a fit are those of
    :func:`ndoptimize.least_squares` along the given batch axes, with the
    given units of the parameters.
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


def _assert_parameters_equal(
    a: na.ModelFitFunctionArray,
    b: na.ModelFitFunctionArray,
) -> None:
    """Check that two fits found the same parameters."""
    for name in a.model.parameters:
        assert np.array_equal(
            a.parameters.components[name],
            b.parameters.components[name],
        )


def test_fit() -> None:
    """
    Fitting named arrays with units, partial bounds and priors, and
    correlated noise gives the same fits as :func:`ndoptimize.least_squares`
    on the equivalent plain arrays.
    """
    data = _data.copy()
    data[dict(x=0, y=1, velocity=slice(10, 13))] = np.nan * u.DN

    result = _fit(
        outputs=data,
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


def test_fit_units() -> None:
    """
    Bounds and priors in other units than the guess give the same fits, and
    a variance in a unit which is not that of the outputs squared raises an
    error.
    """
    result = _fit(
        prior_mean=na.CartesianNdVectorArray(dict(width=15 * _unit_x)),
        prior_sigma=na.CartesianNdVectorArray(dict(width=5 * _unit_x)),
    )
    result_converted = _fit(
        prior_mean=na.CartesianNdVectorArray(dict(width=15000 * u.m / u.s)),
        prior_sigma=na.CartesianNdVectorArray(dict(width=5000 * u.m / u.s)),
    )
    assert np.allclose(result.parameters, result_converted.parameters)

    with pytest.raises(u.UnitConversionError):
        _fit(variance=1 * u.km**2).parameters


def test_fit_dimensionless() -> None:
    """
    Fits of arrays without units give parameters without units, and a model
    without a unit.
    """
    result = _fit(
        inputs=na.value(_velocity),
        outputs=na.value(_data),
        guess=na.CartesianNdVectorArray(
            {name: na.value(c) for name, c in _guess.components.items()}
        ),
        variance=_sigma.value**2,
    )
    result_units = _fit()
    for name in _names:
        assert na.unit(result.parameters.components[name]) is None
        assert np.array_equal(
            result.parameters.components[name],
            na.value(result_units.parameters.components[name]),
        )
    assert na.unit(result.covariance.center.width) is None
    assert na.unit(result.predictions) is None
    assert np.allclose(result.predictions, na.value(result_units.predictions))

    # a variance without a unit is dimensionless, like the outputs in percent
    result_percent = _fit(
        inputs=na.value(_velocity),
        outputs=(na.value(_data) << u.dimensionless_unscaled).to(u.percent),
        guess=na.CartesianNdVectorArray(
            dict(
                amplitude=500 * u.percent,
                center=0,
                width=15,
                slope=0 * u.percent,
                intercept=100 * u.percent,
            )
        ),
        variance=_sigma.value**2,
    )
    assert np.allclose(result_percent.parameters.amplitude, result.parameters.amplitude)
    assert np.allclose(result_percent.parameters.center, result.parameters.center)


def test_fit_batch_parameters() -> None:
    """
    A batch axis which only the guess varies along gives a fit of the same
    outputs for each guess.
    """
    data = _data[dict(x=0, y=0)]
    guess = _guess_with(center=na.ScalarArray(np.array([-20.0, 20]), "trial") * _unit_x)

    result = _fit(outputs=data, guess=guess)

    assert result.parameters.center.shape == dict(trial=2)
    for i in range(2):
        result_i = _fit(outputs=data, guess=guess[dict(trial=i)])
        for name in _names:
            assert np.array_equal(
                result.parameters.components[name][dict(trial=i)],
                result_i.parameters.components[name],
            )


def test__call__() -> None:
    """
    Calling a fit evaluates the best-fit model at new inputs, in the unit of
    the outputs, along the same axis with another number of points, or at a
    single point.
    """
    fit = _fit()

    velocity = na.linspace(-100, 100, axis=_axis, num=101) * _unit_x
    result = fit(velocity)
    assert isinstance(result, na.FunctionArray)
    assert result.inputs is velocity
    assert result.outputs.shape == dict(fit.chi2.shape, velocity=101)
    assert na.unit(result.outputs) == u.DN
    assert np.allclose(result.outputs, _gaussian_linear(velocity, fit.parameters))

    point = fit(10 * _unit_x)
    assert point.outputs.shape == fit.chi2.shape
    assert np.allclose(point.outputs, _gaussian_linear(10 * _unit_x, fit.parameters))


def test_predictions() -> None:
    """The predictions are the model of the outputs, with the chi-squared of
    the fit."""
    fit = _fit()

    f = fit.predictions
    assert f.shape == _data.shape
    assert na.unit(f) == u.DN
    assert np.array_equal(f, fit(_velocity).outputs)

    chi2 = (np.square(_data - f) / np.square(_sigma)).sum(axis=_axis)
    assert np.allclose(chi2, fit.chi2)


def test__getitem__() -> None:
    """Indexing a fit along a batch axis fits the selected outputs."""
    index = dict(x=1, y=slice(0, 2))
    result = _fit()[index]
    assert isinstance(result, na.ModelFitFunctionArray)
    for name in _names:
        assert np.allclose(
            result.parameters.components[name],
            _fit().parameters.components[name][index],
        )


def test_where_model() -> None:
    """A mask leaves the outputs it excludes out of the fit."""
    where = _velocity < 60 * _unit_x
    result = _fit(where_model=where)
    expected = _fit(outputs=np.where(where, _data, np.nan * u.DN))
    _assert_parameters_equal(result, expected)
    assert np.all(result.num_points == where.sum())


def test_axis_model_none() -> None:
    """Without `axis_model`, every axis of the outputs is a single fit."""
    data = _data[dict(x=0, y=0)]
    result = _fit(outputs=data, axis_model=None)
    _assert_parameters_equal(result, _fit(outputs=data))
    assert result.chi2.shape == dict()
    assert result.predictions.shape == data.shape


def test_components_model() -> None:
    """
    The model depends on the components of the inputs named by
    `components_model`, with dots for those of nested vectors.
    """
    expected = _fit()
    position = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([1.0, 2]), "x") * u.arcsec,
        y=na.ScalarArray(np.array([3.0, 4, 5]), "y") * u.arcsec,
    )

    result = _fit(
        inputs=na.SpectralPositionalVectorArray(wavelength=_velocity, position=position),
        components_model="wavelength",
    )
    _assert_parameters_equal(result, expected)
    assert np.array_equal(result.predictions, expected.predictions)

    result_nested = _fit(
        inputs=na.SpectralPositionalVectorArray(
            wavelength=0 * u.AA,
            position=na.Cartesian2dVectorArray(_velocity, position.y),
        ),
        components_model=("position.x",),
    )
    _assert_parameters_equal(result_nested, expected)

    with pytest.raises(ValueError, match="not components of the inputs"):
        _fit(
            inputs=na.SpectralPositionalVectorArray(_velocity, position),
            components_model="velocity",
        ).parameters

    with pytest.raises(TypeError, match="vector inputs"):
        _fit(components_model="wavelength").parameters


_samples = na.random.normal(
    _gaussian_linear(_velocity, _truth),
    _sigma,
    dict(_distribution=3),
    seed=1,
)
"""Three more noisy realizations of the test spectra."""


def test_uncertain() -> None:
    """
    Fitting uncertain outputs fits the nominal outputs and each of their
    samples, and the predictions are uncertain too.
    """
    data = na.UncertainScalarArray(_data, _samples)
    kwargs: dict[str, Any] = dict(
        prior_mean=na.CartesianNdVectorArray(dict(center=0 * _unit_x)),
        prior_sigma=na.CartesianNdVectorArray(dict(center=10 * _unit_x)),
    )

    result = _fit(outputs=data, **kwargs)
    nominal = _fit(outputs=data.nominal, **kwargs)
    distribution = _fit(outputs=data.distribution, **kwargs)

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

    f = result.predictions
    assert isinstance(f, na.UncertainScalarArray)
    assert np.array_equal(f.nominal, nominal.predictions)
    assert np.array_equal(f.distribution, distribution.predictions)


def test_uncertain_guess() -> None:
    """
    An uncertain component of a vector, which the dispatcher does not see,
    is fit like uncertain outputs: one fit for the nominal guess and one for
    each of its samples.
    """
    center = na.UncertainScalarArray(
        nominal=0 * _unit_x,
        distribution=na.ScalarArray(np.array([-20.0, 20]), "_distribution") * _unit_x,
    )

    result = _fit(guess=_guess_with(center=center))

    nominal = _fit()
    distribution = _fit(guess=_guess_with(center=center.distribution))
    p = result.parameters.center
    assert isinstance(p, na.UncertainScalarArray)
    assert np.array_equal(p.nominal, nominal.parameters.center)
    assert np.array_equal(p.distribution, distribution.parameters.center)


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


def test_vector() -> None:
    """
    A vector of inputs gives the model its components as inputs, and the
    points of a fit may vary along more than one axis.
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
    data = na.random.normal(
        truth.amplitude * np.exp(-r2 / (2 * np.square(truth.width))),
        0.5 * u.DN,
        seed=1,
    )

    # the compiled model is its Python source
    x_test = np.array([[0.5, -1.0], [2.0, 0.3]])
    p_test = np.array([10, 0.5, 0.3, 1.5])
    f_compiled, f_python = np.empty(2), np.empty(2)
    jacobian_compiled, jacobian_python = np.empty((2, 4)), np.empty((2, 4))
    _gaussian_2d(x_test, p_test, f_compiled, jacobian_compiled)
    _gaussian_2d.function.py_func(x_test, p_test, f_python, jacobian_python)  # type: ignore[attr-defined]
    assert np.allclose(f_compiled, f_python)
    assert np.allclose(jacobian_compiled, jacobian_python)

    kwargs: dict[str, Any] = dict(
        inputs=position,
        outputs=data,
        model=_gaussian_2d,
        guess=na.CartesianNdVectorArray(
            dict(amplitude=8 * u.DN, x0=0 * u.arcsec, y0=0 * u.arcsec, width=2 * u.arcsec)
        ),
        variance=0.25 * u.DN**2,
        axis_model=("px", "py"),
    )
    result = na.ModelFitFunctionArray(**kwargs)

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

    p = result.parameters
    r2 = np.square(position.x - p.x0) + np.square(position.y - p.y0)
    assert np.allclose(
        result.predictions,
        p.amplitude * np.exp(-r2 / (2 * np.square(p.width))),
    )

    with pytest.raises(ValueError, match="single axis"):
        na.ModelFitFunctionArray(correlation=_correlation, **kwargs).parameters


@pytest.mark.parametrize(
    argnames="kwargs,error",
    argvalues=[
        (dict(model=None), ValueError),
        (dict(guess=None), ValueError),
        (dict(variance=None), ValueError),
        (dict(guess=na.CartesianNdVectorArray(dict(amplitude=5 * u.DN))), ValueError),
        (dict(guess=_guess_with(extra=1)), ValueError),
        (dict(lower=na.CartesianNdVectorArray(dict(sigma=3 * _unit_x))), ValueError),
        (dict(lower=3 * _unit_x), TypeError),
        (dict(guess=_guess_with(center=_velocity)), ValueError),
        (dict(axis_model="wavelength"), ValueError),
        (dict(outputs=na.Cartesian2dVectorArray(_data, _data)), TypeError),
    ],
)
def test_fit_invalid(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        _fit(**kwargs).parameters


def test_evaluate_invalid() -> None:
    """Parameters which are not scalars cannot be evaluated."""
    with pytest.raises(TypeError):
        na.optimize._evaluate(
            model=ndoptimize.models.gaussian_linear,
            x=_velocity,
            parameters=_guess_with(
                center=na.Cartesian2dVectorArray(0 * _unit_x, 0 * _unit_x)
            ),
            axis=_axis,
            unit=None,
        )


@ndoptimize.model(parameters=("weight_lower", "weight_upper"))
def _edges_weighted(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    """A weighted sum of the lower and upper edges of each bin, which tells
    them apart."""
    for i in range(x.shape[0]):
        f[i] = parameters[0] * x[i, 0] + parameters[1] * x[i, 1]
        jacobian[i, 0] = x[i, 0]
        jacobian[i, 1] = x[i, 1]


_edges = na.linspace(0, 10, axis=_axis, num=11) * _unit_x
"""The edges of ten bins along the points axis."""

_unit_weight = u.DN / _unit_x


def _edges_data(
    edges: na.AbstractScalar,
    weight_lower: na.AbstractScalar,
    weight_upper: na.AbstractScalar,
) -> na.AbstractScalar:
    """:func:`_edges_weighted` in named arrays, for the given edges of bins."""
    lower = edges[{_axis: slice(None, -1)}]
    upper = edges[{_axis: slice(1, None)}]
    return weight_lower * lower + weight_upper * upper


def test_vertex() -> None:
    """
    Along an axis of `axis_model` along which the inputs are the edges of
    bins, the model gets the lower and upper edges of each bin, in that
    order, and evaluating it at new edges gives one fewer output.
    """
    # the compiled model is its Python source
    x_test = np.array([[1.0, 2.0], [2.0, 4.0]])
    p_test = np.array([2.0, 3.0])
    f_compiled, f_python = np.empty(2), np.empty(2)
    jacobian_compiled, jacobian_python = np.empty((2, 2)), np.empty((2, 2))
    _edges_weighted(x_test, p_test, f_compiled, jacobian_compiled)
    _edges_weighted.function.py_func(x_test, p_test, f_python, jacobian_python)  # type: ignore[attr-defined]
    assert np.array_equal(f_compiled, f_python)
    assert np.array_equal(jacobian_compiled, jacobian_python)

    weight_lower = na.ScalarArray(np.array([2.0, 4.0]), "y") * _unit_weight
    weight_upper = 3 * _unit_weight
    data = _edges_data(_edges, weight_lower, weight_upper)

    fit = na.ModelFitFunctionArray(
        inputs=_edges,
        outputs=data,
        model=_edges_weighted,
        guess=na.CartesianNdVectorArray(
            dict(weight_lower=1 * _unit_weight, weight_upper=1 * _unit_weight)
        ),
        variance=1 * u.DN**2,
        axis_model=_axis,
    )
    assert fit.axes_vertex == (_axis,)
    assert np.allclose(fit.parameters.weight_lower, weight_lower)
    assert np.allclose(fit.parameters.weight_upper, weight_upper)

    assert fit.predictions.shape == data.shape
    assert np.allclose(fit.predictions, data)

    edges = na.linspace(0, 10, axis=_axis, num=21) * _unit_x
    result = fit(edges)
    assert result.inputs is edges
    assert result.outputs.shape == dict(y=2, velocity=20)
    assert np.allclose(result.outputs, _edges_data(edges, weight_lower, weight_upper))


def test_vertex_components() -> None:
    """
    An input of the model which varies along an axis along which the inputs
    are the edges of bins, but which is not in `axis_model`, raises an error,
    which selecting the other components avoids.
    """
    weight_lower = 2 * _unit_weight
    weight_upper = 3 * _unit_weight
    data = _edges_data(_edges, weight_lower, weight_upper)
    data = data * na.ScalarArray(np.array([1.0, 1.0]), "y")
    inputs = na.Cartesian2dVectorArray(
        x=_edges,
        y=na.linspace(0, 1, axis="y", num=3) * u.arcsec,
    )
    kwargs: dict[str, Any] = dict(
        inputs=inputs,
        outputs=data,
        model=_edges_weighted,
        guess=na.CartesianNdVectorArray(
            dict(weight_lower=1 * _unit_weight, weight_upper=1 * _unit_weight)
        ),
        variance=1 * u.DN**2,
        axis_model=_axis,
    )

    with pytest.raises(ValueError, match="not in `axis_model`"):
        na.ModelFitFunctionArray(**kwargs).parameters

    fit = na.ModelFitFunctionArray(components_model="x", **kwargs)
    assert np.allclose(fit.parameters.weight_lower, weight_lower)
    assert np.allclose(fit.parameters.weight_upper, weight_upper)

    # an input which varies along no axis of edges stays a single input,
    # after the edges of the others
    kwargs["inputs"] = na.Cartesian2dVectorArray(x=_edges, y=0 * u.arcsec)
    fit = na.ModelFitFunctionArray(**kwargs)
    assert np.allclose(fit.parameters.weight_lower, weight_lower)
    assert np.allclose(fit.parameters.weight_upper, weight_upper)


def test_vertex_curvilinear() -> None:
    """An input of the model which varies along two axes along which the
    inputs are the edges of bins has no single lower and upper edge."""
    inputs = _edges + na.linspace(0, 1, axis="y", num=3) * _unit_x
    outputs = na.ScalarArray.zeros(dict(velocity=10, y=2)) * u.DN
    fit = na.ModelFitFunctionArray(
        inputs=inputs,
        outputs=outputs,
        model=_edges_weighted,
        guess=na.CartesianNdVectorArray(
            dict(weight_lower=1 * _unit_weight, weight_upper=1 * _unit_weight)
        ),
        variance=1 * u.DN**2,
    )
    with pytest.raises(ValueError, match="no single lower and upper edge"):
        fit.parameters
