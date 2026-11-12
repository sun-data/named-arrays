"""Numba kernel for :meth:`named_arrays.PolynomialFitFunctionArray.__call__`.

This module is imported lazily by :mod:`named_arrays._functions.functions`
since importing :mod:`numba` is slow.
"""

import numpy as np
import numba

__all__ = [
    "polynomial_numba",
]


@numba.njit(cache=True, parallel=True)
def polynomial_numba(
    inputs: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    Evaluate a polynomial at every point, forming each term at one point at
    a time instead of the whole design matrix at once.

    Parameters
    ----------
    inputs
        The independent variables, of shape ``(component, set, point)``,
        where ``set`` indexes the sets of coefficients, or has a length of
        one if every set of coefficients is evaluated at the same points.
    exponents
        The power of each component in each term of the polynomial,
        of shape ``(term, component)``.
    coefficients
        The coefficient of each term for each output component,
        of shape ``(set, term, output)``.

    Returns
    -------
        The dependent variables, of shape ``(output, set, point)``.
    """

    num_component, num_set_inputs, num_point = inputs.shape
    num_set, num_term, num_output = coefficients.shape

    result = np.empty((num_output, num_set, num_point), dtype=inputs.dtype)

    for i in numba.prange(num_set * num_point):
        s = i // num_point
        p = i % num_point
        s_inputs = s if num_set_inputs > 1 else 0

        for o in range(num_output):
            result[o, s, p] = 0

        for t in range(num_term):
            term = 1.0
            for c in range(num_component):
                for _ in range(exponents[t, c]):
                    term *= inputs[c, s_inputs, p]
            for o in range(num_output):
                result[o, s, p] += coefficients[s, t, o] * term

    return result
