from __future__ import annotations
from typing import TypeVar, Callable
import functools
import named_arrays as na

__all__ = [
    "root_secant",
    "root_newton",
    "minimum_gradient_descent",
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
        jacobian = lambda x: na.jacobian(function=function, x=x)

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
    gradient: None | Callable[[InputT], InputT] = None,
    min_gradient: None | InputT = None,
    max_iterations: int = 1000,
    callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    """
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
        ``0.1 * na.unit(x / gradient(x))``.
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
    """

    x = guess
    f = function(x)

    unit_f = na.unit(f, unit_dimensionless=1)
    unit_x = na.unit(x, unit_dimensionless=1, squeeze=False)

    unit_grad = unit_f / unit_x

    if step_size is None:
        step_size = 0.1 * (unit_x / unit_grad)

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
        gradient=gradient,
        min_gradient=min_gradient,
        max_iterations=max_iterations,
        callback=callback,
    )
