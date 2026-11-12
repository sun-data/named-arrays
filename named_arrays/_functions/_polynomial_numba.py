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
    num_point_coefficient: int,
) -> np.ndarray:  # pragma: nocover
    """
    Evaluate a polynomial at every point, forming each term at one point at
    a time instead of the whole design matrix at once.

    Parameters
    ----------
    inputs
        The independent variables, of shape ``(component, point)``.
        The points are flattened in C order from a grid whose leading axes
        are those of `coefficients`.
    exponents
        The power of each component in each term of the polynomial,
        of shape ``(term, component)``.
    coefficients
        The coefficient of each term for each output component,
        of shape ``(coefficient set, term, output)``.
    num_point_coefficient
        The number of consecutive points which share a set of coefficients,
        the number of points in the axes of the grid which `coefficients`
        does not have.

    Returns
    -------
        The dependent variables, of shape ``(output, point)``.
    """

    num_component, num_point = inputs.shape
    _, num_term, num_output = coefficients.shape

    result = np.empty((num_output, num_point), dtype=inputs.dtype)

    for i in numba.prange(num_point):
        s = i // num_point_coefficient
        for o in range(num_output):
            total = 0.0
            for t in range(num_term):
                term = coefficients[s, t, o]
                for c in range(num_component):
                    for _ in range(exponents[t, c]):
                        term *= inputs[c, i]
                total += term
            result[o, i] = total

    return result
