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
    return a.shape
