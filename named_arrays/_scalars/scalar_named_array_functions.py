from typing import Callable, TypeVar
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "RANDOM_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
]

RANDOM_FUNCTIONS = (
    na.random.uniform,
    na.random.normal,
)
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


class _ScalarTypeError(TypeError):
    pass


def _normalize(a: float | u.Quantity | na.AbstractScalarArray) -> na.AbstractScalarArray:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalarArray):
            result = a
        else:
            raise _ScalarTypeError
    else:
        result = na.ScalarArray(a)

    return result


def random(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalarArray,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity | na.AbstractScalarArray,
) -> na.ScalarArray:

    try:
        args = tuple(_normalize(arg) for arg in args)
        kwargs = {k: _normalize(kwargs[k]) for k in kwargs}
    except _ScalarTypeError:
        return NotImplemented

    if shape_random is None:
        shape_random = dict()

    shape_base = na.shape_broadcasted(*args, *kwargs.values())
    shape = na.broadcast_shapes(shape_base, shape_random)

    args = tuple(arg.ndarray_aligned(shape) for arg in args)
    kwargs = {k: kwargs[k].ndarray_aligned(shape) for k in kwargs}

    unit = None
    for a in args + tuple(kwargs.values()):
        if isinstance(a, u.Quantity):
            unit = a.unit
            break

    if unit is not None:
        args = tuple(
            arg.value if isinstance(arg, u.Quantity)
            else (arg << u.dimensionless_unscaled).to_value(unit)
            for arg in args
        )
        kwargs = {
            k: kwargs[k].value if isinstance(kwargs[k], u.Quantity)
            else (kwargs[k] << u.dimensionless_unscaled).to_value(unit)
            for k in kwargs
        }

    if seed is None:
        func = getattr(np.random, func.__name__)
    else:
        func = getattr(np.random.default_rng(seed), func.__name__)

    value = func(
        *args,
        size=tuple(shape.values()),
        **kwargs,
    )

    if unit is not None:
        value = value << unit

    return na.ScalarArray(
        ndarray=value,
        axes=tuple(shape.keys()),
    )
