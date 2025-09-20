from typing import Callable

import numpy as np
import named_arrays as na

__all__ = [
    'HANDLED_FUNCTIONS',
]

HANDLED_FUNCTIONS = dict()


def implements(numpy_function: Callable):
    """Register an ``__array_function__`` implementation for :class:`named_array.AbstractArray` objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.shape)
def shape(
        a: na.AbstractScalarArray,
) -> dict[str, int]:
    """
    Compute the shape of the given array.

    In :mod:`numpy`, the shape of an array is a :class:`tuple` of integers.
    For this package, each axis is characterized by a name instead of
    its position, so the shape is a :class:`dict` where the keys are
    the axis names and the values are number of elements along each axis.

    Parameters
    ----------
    a
        The array to compute the shape of.
    """
    return a.shape


@implements(np.broadcast_to)
def broadcast_to(
    array: na.AbstractArray,
    shape: dict[str, int],
) -> na.AbstractExplicitArray:
    """
    Broadcast the given array to the requested shape.

    Parameters
    ----------
    array
        The array to broadcast.
    shape
        The requested shape of the result.

    See Also
    --------
    :func:`numpy.broadcast_to`: Equivalent :mod:`numpy` function.
    """
    return na.broadcast_to(
        array=array,
        shape=shape,
    )
