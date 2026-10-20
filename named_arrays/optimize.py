"""
Minimization and root finding of arbitrary functions.

Analogous to :mod:`scipy.optimize`.
"""

from __future__ import annotations
from typing import TypeVar, Callable, Sequence, TYPE_CHECKING
import dataclasses
import math
import numpy as np
import astropy.units as u
import named_arrays as na

if TYPE_CHECKING:
    import ndoptimize

__all__ = [
    "root_secant",
    "root_newton",
    "minimum_gradient_descent",
    "minimum_brent",
]

InputT = TypeVar("InputT", bound="float | u.Quantity | na.AbstractScalarArray")
OutputT = TypeVar("OutputT", bound="float | u.Quantity | na.AbstractScalarArray")


def root_newton(
        function: Callable[[InputT], OutputT],
        guess: InputT,
        jacobian: None | Callable[[InputT], OutputT] = None,
        max_abs_error: None | OutputT = None,
        max_iterations: int = 100,
        callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
):
    """
    Find the root of a given function using
    `Newton's method <https://en.wikipedia.org/wiki/Newton%27s_method>`_

    Parameters
    ----------
    function
        a scalar-valued or vector-valued function to find the roots of
    guess
        an initial guess for the root
    jacobian
        The Jacobian of the function (optional). If :obj:`None`, :func:`named_arrays.jacobian`
        is used.
    max_abs_error
        Maximum absolute error between zero and the function value.
        If the function value at the current root guess is below this value,
        the root is considered converged.
    max_iterations
        Maxmimum number of iterations to carry out before a :class:`ValueError`
        is raised.
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current iteration,
        ``x`` is the current root guess, ``f`` is the current function value,
        and ``converged`` is an array storing the convergence state for every
        root being computed.
    """
    if jacobian is None:
        def jacobian(x: float | u.Quantity | na.AbstractArray):
            return na.jacobian(
                function=function,
                x=x,
            )

    f_guess = function(guess)
    if max_abs_error is None:
        max_abs_error = 1e-10
        if na.unit(f_guess) is not None:
            max_abs_error = max_abs_error * na.unit(f_guess)
    max_abs_error = na.asanyarray(max_abs_error, like=f_guess)

    return na._named_array_function(
        func=root_newton,
        function=function,
        guess=guess,
        jacobian=jacobian,
        max_abs_error=max_abs_error,
        max_iterations=max_iterations,
        callback=callback,
    )


def root_secant(
        function: Callable[[InputT], OutputT],
        guess: InputT,
        min_step_size: None | InputT = None,
        max_abs_error: None | OutputT = None,
        max_iterations: int = 100,
        damping: None | float = None,
        callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    """
    Find the root of a given function using the
    `secant method <https://en.wikipedia.org/wiki/Secant_method>`_.

    Parameters
    ----------
    function
        a scalar-valued or vector-valued function to find the root of.
    guess
        an initial guess for the root
    min_step_size
        Stop iterating if the step size falls below this value
    max_abs_error
        Maximum absolute error between zero and the function value.
        If the function value at the current root guess is below this value,
        the root is considered converged.
    max_iterations
        Maxmimum number of iterations to carry out before a :class:`ValueError`
        is raised.
    damping
        If this parameter is equal to ``1``, it has no effect and is equivalent
        to the default, :obj:`None` (vanilla secant method).
        If this parameter is less than ``1``, the damped secant method is
        carried out.
        The damped secant method can address some of the failure modes of the
        secant method, particularly the problem of
        `entering an infinite cycle <https://en.wikipedia.org/wiki/Newton%27s_method#Starting_point_enters_a_cycle>`_
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current iteration,
        ``x`` is the current root guess, ``f`` is the current function value,
        and ``converged`` is an array storing the convergence state for every
        root being computed.
    """

    if min_step_size is None:
        min_step_size = 1e-13
        if na.unit(guess) is not None:
            min_step_size = min_step_size * na.unit(guess)
    min_step_size = na.asanyarray(min_step_size, like=guess)

    f_guess = function(guess)
    if max_abs_error is None:
        max_abs_error = 1e-13
        if na.unit(f_guess) is not None:
            max_abs_error = max_abs_error * na.unit(f_guess)
    max_abs_error = na.asanyarray(max_abs_error, like=f_guess)

    return na._named_array_function(
        func=root_secant,
        function=function,
        guess=guess,
        min_step_size=min_step_size,
        max_abs_error=max_abs_error,
        max_iterations=max_iterations,
        damping=damping,
        callback=callback,
    )


def minimum_gradient_descent(
    function: Callable[[InputT], OutputT],
    guess: InputT,
    step_size: None | InputT = None,
    momentum: float | OutputT = 0,
    gradient: None | Callable[[InputT], InputT] = None,
    min_gradient: None | InputT = None,
    max_iterations: int = 1000,
    callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    r"""
    Find the local minimum of the given function using the
    `gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ method.

    Parameters
    ----------
    function
        The function to minimize.
    guess
        An initial guess for the local minimum.
    step_size
        The learning rate for the gradient descent algorithm.
        This should have the same units as ``x / gradient(x)``.
        If :obj:`None` (the default), this takes the value
        ``0.01 * na.unit(x / gradient(x))``.
    momentum
        The momentum constant, :math:`\beta` for the gradient descent algorithm.
        Should be a dimensionless number between zero and one.
        Defaults to zero, which equivalent to vanilla gradient descent with
        no momentum.
    gradient
        The gradient of `function`.
        If :obj:`None` (the default), the gradient is computed using
        :func:`named_arrays.jacobian`.
    min_gradient
        The convergence threshold for the local minimum.
        If the gradient is smaller than this value, this function will stop.
        This should have the same units as ``gradient(x)``.
        If :obj:`None` (the default), this takes the value
        ``1e-10 * na.unit(gradient(x))``.
    max_iterations
        The maximum number of steps to take before raising an error.
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current iteration,
        ``x`` is the current guess, ``f`` is the current function value,
        and ``converged`` is an array storing the convergence state for every
        minimum being computed.

    Notes
    -----

    This function uses the update rules described in :cite:t:`Goh2017`,

    .. math::
        :label: momentum-equation

        z_{k + 1} = \beta z_k + \nabla f(x_k)

    .. math::
        :label: gradient-descent

        x_{k + 1} = x_k - \alpha z_k,

    where :math:`x_k` is the current guess for iteration :math:`k`,
    :math:`f` is the objective function,
    :math:`\alpha` is the learning rate,
    and :math:`\beta` is the momentum constant.
    """

    x = guess
    f = function(x)

    unit_f = na.unit(f, unit_dimensionless=1)
    unit_x = na.unit(x, unit_dimensionless=1, squeeze=False)

    unit_grad = unit_f / unit_x

    if step_size is None:
        step_size = 0.01 * (unit_x / unit_grad)

    if gradient is None:
        def gradient(x: float | na.AbstractScalar | na.AbstractVectorArray):
            return na.jacobian(
                function=function,
                x=x,
            )

    if min_gradient is None:
        min_gradient = 1e-10 * unit_grad
    min_gradient = na.asanyarray(min_gradient, like=f)

    return na._named_array_function(
        func=minimum_gradient_descent,
        function=function,
        guess=guess,
        step_size=step_size,
        momentum=momentum,
        gradient=gradient,
        min_gradient=min_gradient,
        max_iterations=max_iterations,
        callback=callback,
    )


def minimum_brent(
    function: Callable[[InputT], OutputT],
    a: InputT,
    b: InputT,
    min_step_size: None | InputT = None,
    max_iterations: int = 100,
    callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    r"""
    Find the minimum of a function of one variable inside a bracket using
    `Brent's method <https://en.wikipedia.org/wiki/Brent%27s_method>`_.

    This is the bounded form of the algorithm (golden-section search with
    parabolic interpolation, as in :func:`scipy.optimize.fminbound`),
    which needs neither a derivative nor an initial guess, only a bracket
    :math:`[a, b]` containing the minimum.

    The bracket, the tolerance, and the value of `function` are broadcast
    against each other, and every element of the result converges
    independently: `function` is evaluated exactly once per iteration on the
    whole array of trial points, so a function that is expensive but
    vectorized (a ray trace, for example) costs the same for one minimum as
    for a thousand.

    Parameters
    ----------
    function
        The function to minimize.
        It must accept an array with the broadcast shape of `a` and `b`
        and return an array of the same shape (or one which broadcasts
        with it).
    a
        The lower bound of the bracket.
    b
        The upper bound of the bracket.
    min_step_size
        The absolute tolerance on the location of the minimum, in the same
        units as `a` and `b`.
        If :obj:`None` (the default), this takes the value
        ``1e-5 * na.unit(a)``.
    max_iterations
        The maximum number of iterations to carry out before a
        :class:`ValueError` is raised.
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current
        iteration, ``x`` is the current best estimate of the minimum,
        ``f`` is the function value there, and ``converged`` is an array
        storing the convergence state of every minimum being computed.

    Examples
    --------
    Find the minimum of a family of V-shaped functions, one per element of
    the array of offsets, in a single vectorized call.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na

        offset = na.linspace(-1, 1, axis="offset", num=5)

        def function(x):
            return np.abs(x - offset) + 1

        na.optimize.minimum_brent(
            function=function,
            a=-2,
            b=2,
        )
    """
    if min_step_size is None:
        min_step_size = 1e-5
        if na.unit(a) is not None:
            min_step_size = min_step_size * na.unit(a)
    min_step_size = na.asanyarray(min_step_size, like=a)

    return na._named_array_function(
        func=minimum_brent,
        function=function,
        a=a,
        b=b,
        min_step_size=min_step_size,
        max_iterations=max_iterations,
        callback=callback,
    )




@dataclasses.dataclass(frozen=True, eq=False)
class _LeastSquaresResult:
    """
    The results of :func:`_least_squares` for every fit of a batch, which
    :class:`named_arrays.ModelFitFunctionArray` exposes as properties of the
    same names.
    """

    parameters: na.AbstractVectorArray
    """The best-fit parameters."""

    covariance: na.AbstractMatrixArray
    """The covariance of the best-fit parameters."""

    chi2: na.AbstractScalar
    """The chi-squared of the data at the solution."""

    num_points: na.AbstractScalar
    """The number of data points used by each fit."""

    iterations: na.AbstractScalar
    """The number of steps taken by each fit."""

    converged: na.AbstractScalar
    """Whether each fit converged."""


def _scalar(
    a: float | u.Quantity | na.AbstractScalar,
    name: str,
) -> na.ScalarArray:
    """
    `a` as an explicit scalar array.

    Parameters
    ----------
    a
        An argument of :func:`_least_squares` or :func:`_evaluate`, or a
        component of one.
    name
        How to refer to `a` in the :class:`TypeError` raised if it is not a
        scalar array, which includes an uncertain array.
    """
    if not isinstance(a, na.AbstractArray):
        return na.ScalarArray(a)
    if not isinstance(a, na.AbstractScalarArray):
        raise TypeError(f"{name} must be a scalar array, got {type(a).__qualname__}.")
    return a.explicit


def _inputs(
    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
) -> list[na.ScalarArray]:
    """
    The inputs of a model, the components of `x` if it is a vector, with
    those of nested vectors in their place, or else `x` itself, each as the
    value in its own unit.
    """
    if isinstance(x, na.AbstractVectorArray):
        inputs = list(x.explicit.cartesian_nd.components.values())
    else:
        inputs = [x]
    inputs = [_scalar(c, "An input of the model") for c in inputs]
    return [c.to_value(na.unit_normalized(c)) for c in inputs]


def _parameters(
    vector: None | na.AbstractVectorArray,
    names: tuple[str, ...],
    name_argument: str,
    complete: bool,
) -> dict[str, na.ScalarArray]:
    """
    The components of a vector of values of the parameters of a model, by
    name.

    Parameters
    ----------
    vector
        The vector, or :obj:`None` for no components.
    names
        The names of the parameters of the model.
    name_argument
        The name of the argument the vector was given as, for errors.
    complete
        Whether the vector must have a component for every parameter, rather
        than for some of them.
    """
    if vector is None:
        components = dict()
    elif isinstance(vector, na.AbstractVectorArray):
        components = vector.explicit.components
    else:
        raise TypeError(
            f"`{name_argument}` must be a vector, got {type(vector).__qualname__}."
        )

    if complete and set(components) != set(names):
        raise ValueError(
            f"The components of `{name_argument}`, {tuple(components)}, must "
            f"be the parameters of the model, {names}."
        )
    unknown = tuple(set(components) - set(names))
    if unknown:
        raise ValueError(
            f"The components {unknown} of `{name_argument}` are not "
            f"parameters of the model, {names}."
        )

    return {
        name: _scalar(c, f"The component {name!r} of `{name_argument}`")
        for name, c in components.items()
    }


def _axes(
    arrays_points: list[na.ScalarArray],
    arrays_parameters: dict[str, dict[str, na.ScalarArray]],
    axis: str | Sequence[str],
    required: bool,
) -> tuple[tuple[str, ...], dict[str, int], dict[str, int]]:
    """
    Split the axes of the arguments of :func:`_least_squares` or
    :func:`_evaluate` into the axes along which the points of each set vary
    and the batch axes.

    Parameters
    ----------
    arrays_points
        The arrays which may vary along `axis`.
    arrays_parameters
        The values of the parameters by argument and parameter, which may not
        vary along `axis`.
    axis
        The axis or axes along which the points of each set vary.
    required
        Whether every axis in `axis` must be an axis of the arguments.
        If not, those which are not are left out.

    Returns
    -------
    axis
        The axes along which the points of each set vary, as a tuple.
    shape_batch
        The shape of the batch axes.
    shape_points
        The shape of the axes in `axis`.
    """
    arrays = arrays_points + [
        p for vector in arrays_parameters.values() for p in vector.values()
    ]
    shape = na.shape_broadcasted(*arrays)

    if isinstance(axis, str):
        axis = (axis,)
    missing = tuple(a for a in axis if a not in shape)
    if missing and required:
        raise ValueError(f"{missing=} are not axes of the arguments, {shape=}.")
    axis = tuple(a for a in axis if a in shape)

    for name_argument, vector in arrays_parameters.items():
        for name, p in vector.items():
            along = tuple(a for a in axis if a in p.shape)
            if along:
                raise ValueError(
                    f"The parameters of a set are constant, but "
                    f"`{name_argument}` varies along {along} for the parameter "
                    f"{name!r}."
                )

    shape_batch = {a: shape[a] for a in shape if a not in axis}
    shape_points = {a: shape[a] for a in axis}
    return axis, shape_batch, shape_points


def _points(
    a: na.ScalarArray,
    shape_batch: dict[str, int],
    shape_points: dict[str, int],
) -> np.ndarray:
    """
    `a` as an array whose batch axes are followed by one axis, along which
    the points of each set vary.
    """
    ndarray = a.ndarray_aligned(tuple(shape_batch) + tuple(shape_points))
    shape_ndarray_batch = ndarray.shape[:len(shape_batch)]
    ndarray = np.broadcast_to(ndarray, shape_ndarray_batch + tuple(shape_points.values()))
    return ndarray.reshape(shape_ndarray_batch + (math.prod(shape_points.values()),))


def _stack(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack arrays with the same last axes along a new last axis."""
    return np.stack(np.broadcast_arrays(*arrays), axis=~0)


def _result(
    ndarray: np.ndarray,
    axes: tuple[str, ...],
    *units: None | u.UnitBase,
) -> na.ScalarArray:
    """
    A result along the given axes, with the product of the given units, or
    without a unit if none of them is a unit.
    """
    units_result = [unit for unit in units if unit is not None]
    if units_result:
        ndarray = ndarray << math.prod(units_result[1:], start=units_result[0])
    return na.ScalarArray(ndarray, axes)


def _least_squares(
    model: ndoptimize.Model,
    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    data: float | u.Quantity | na.AbstractScalar,
    variance: float | u.Quantity | na.AbstractScalar,
    guess: na.AbstractVectorArray,
    axis: str | Sequence[str],
    lower: None | na.AbstractVectorArray = None,
    upper: None | na.AbstractVectorArray = None,
    prior_mean: None | na.AbstractVectorArray = None,
    prior_sigma: None | na.AbstractVectorArray = None,
    correlation: None | Sequence[float] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> _LeastSquaresResult:
    """
    Fit a model to many independent sets of data at once with
    :func:`ndoptimize.least_squares`, for
    :class:`named_arrays.ModelFitFunctionArray`.

    The arguments are the fields of :class:`named_arrays.ModelFitFunctionArray`
    of the same names, except that `x` is the inputs of the model (already
    selected with :attr:`~named_arrays.ModelFitFunctionArray.components_model`),
    `data` is the outputs (NaN where they are masked), and `axis` is the
    axes of :attr:`~named_arrays.ModelFitFunctionArray.axis_model`, every one
    of which must be an axis of the arguments.
    None of them may be uncertain, so
    :class:`named_arrays.ModelFitFunctionArray` fits the nominal values and
    the samples of uncertain fields separately.
    """
    import ndoptimize

    names = model.parameters

    data = _scalar(data, "`data`")
    variance = _scalar(variance, "`variance`")
    inputs = _inputs(x)
    vectors = dict(
        guess=guess,
        lower=lower,
        upper=upper,
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
    )
    parameters = {
        name_argument: _parameters(
            vector=vector,
            names=names,
            name_argument=name_argument,
            complete=name_argument == "guess",
        )
        for name_argument, vector in vectors.items()
    }

    if set(parameters["prior_mean"]) != set(parameters["prior_sigma"]):
        raise ValueError(
            f"A prior needs a mean and a standard deviation, but `prior_mean` "
            f"has the components {tuple(parameters['prior_mean'])} and "
            f"`prior_sigma` has {tuple(parameters['prior_sigma'])}."
        )

    # the model sees plain numbers: the data and the inputs in their own
    # units, and each parameter in the unit of its guess
    unit_data = na.unit_normalized(data)
    data = data.to_value(unit_data)
    variance = variance.to_value(unit_data ** 2)

    units_guess = {name: na.unit_normalized(parameters["guess"][name]) for name in names}
    default = dict(lower=-np.inf, upper=np.inf, prior_mean=0, prior_sigma=np.inf)
    values = {
        name_argument: {
            name: components[name].to_value(units_guess[name])
            if name in components else na.ScalarArray(default[name_argument])
            for name in names
        }
        for name_argument, components in parameters.items()
    }

    axis, shape_batch, shape_points = _axes(
        arrays_points=[data, variance, *inputs],
        arrays_parameters=values,
        axis=axis,
        required=True,
    )

    if correlation is not None and len(axis) != 1:
        raise ValueError(
            f"`correlation` is given by separation along a single axis, but "
            f"the points of a fit vary along {axis=}."
        )

    axes_batch = tuple(shape_batch)

    result = ndoptimize.least_squares(
        model=model,
        x=_stack([_points(c, shape_batch, shape_points) for c in inputs]),
        data=_points(data, shape_batch, shape_points),
        variance=_points(variance, shape_batch, shape_points),
        correlation=correlation,
        max_iterations=max_iterations,
        tolerance=tolerance,
        **{
            name_argument: _stack([
                vector[name].ndarray_aligned(axes_batch) for name in names
            ])
            for name_argument, vector in values.items()
        },
    )

    # the results keep the components of the guess in its order, and a
    # parameter whose guess has no unit has none
    index = {name: k for k, name in enumerate(names)}
    units_result = {name: na.unit(parameters["guess"][name]) for name in names}
    order = tuple(parameters["guess"])

    return _LeastSquaresResult(
        parameters=guess.type_explicit.from_components({
            name: _result(
                result.parameters[..., index[name]],
                axes_batch,
                units_result[name],
            )
            for name in order
        }),
        covariance=na.CartesianNdMatrixArray({
            row: na.CartesianNdVectorArray({
                column: _result(
                    result.covariance[..., index[row], index[column]],
                    axes_batch,
                    units_result[row],
                    units_result[column],
                )
                for column in order
            })
            for row in order
        }),
        chi2=na.ScalarArray(result.chi2, axes_batch),
        num_points=na.ScalarArray(result.num_points, axes_batch),
        iterations=na.ScalarArray(result.iterations, axes_batch),
        converged=na.ScalarArray(result.converged, axes_batch),
    )


def _evaluate(
    model: ndoptimize.Model,
    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    parameters: na.AbstractVectorArray,
    axis: str | Sequence[str],
    unit: None | u.UnitBase,
) -> na.ScalarArray:
    """
    Evaluate a model for many sets of parameters at once with
    :func:`ndoptimize.evaluate`, for
    :meth:`named_arrays.AbstractModelFunctionArray.__call__`.

    None of the arguments may be uncertain, so
    :meth:`named_arrays.AbstractModelFunctionArray.__call__` evaluates the
    model for the nominal values and the samples of uncertain arguments
    separately.

    Parameters
    ----------
    model
        The model to evaluate.
    x
        The inputs of the model, already selected with
        :attr:`~named_arrays.AbstractModelFunctionArray.components_model`.
    parameters
        The parameters of the model, which must not vary along `axis`.
    axis
        The axes along which the points of a single evaluation vary.
        Those which `x` does not vary along are ignored.
    unit
        The unit of the values of the model, or :obj:`None` for none.
    """
    import ndoptimize

    names = model.parameters

    inputs = _inputs(x)
    components = _parameters(
        vector=parameters,
        names=names,
        name_argument="parameters",
        complete=True,
    )

    # the model sees each parameter as the value in its own unit
    values = {
        name: components[name].to_value(na.unit_normalized(components[name]))
        for name in names
    }

    axis, shape_batch, shape_points = _axes(
        arrays_points=inputs,
        arrays_parameters=dict(parameters=values),
        axis=axis,
        required=False,
    )
    axes_batch = tuple(shape_batch)

    f = ndoptimize.evaluate(
        model=model,
        x=_stack([_points(c, shape_batch, shape_points) for c in inputs]),
        parameters=_stack([values[name].ndarray_aligned(axes_batch) for name in names]),
    )

    f = f.reshape(f.shape[:~0] + tuple(shape_points.values()))

    return _result(f, axes_batch + axis, unit)
