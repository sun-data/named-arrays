from typing import TypeVar, Callable
import named_arrays as na

__all__ = [
    "root_secant",
]

InputT = TypeVar("InputT", bound="float | u.Quantity | na.AbstractScalarArray")
OutputT = TypeVar("OutputT", bound="float | u.Quantity | na.AbstractScalarArray")


def root_secant(
        function: Callable[[InputT], OutputT],
        guess: InputT,
        min_step_size: None | InputT = None,
        max_abs_error: None | OutputT = None,
        max_iterations: int = 100,
        damping: None | float = None,
        callback: None | Callable[[InputT, OutputT], None] = None,
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
        if guess.unit is not None:
            min_step_size = min_step_size * guess.unit
    min_step_size = na.asanyarray(min_step_size, like=guess)

    f_guess = function(guess)
    if max_abs_error is None:
        max_abs_error = 1e-13
        if f_guess.unit is not None:
            max_abs_error = max_abs_error * f_guess.unit
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
