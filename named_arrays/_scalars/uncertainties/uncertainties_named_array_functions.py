from typing import Callable
import astropy.units as u
import named_arrays as na

__all__ = [
    "HANDLED_FUNCTIONS",
]

HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


class _UncertainScalarTypeError(TypeError):
    pass


def _normalize(a: float | u.Quantity | na.AbstractScalar):
    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalar):
            if isinstance(a, na.AbstractUncertainScalarArray):
                result = a
            else:
                result = na.UncertainScalarArray(a, a)
        else:
            return _UncertainScalarTypeError
    else:
        result = na.UncertainScalarArray(a, a)

    return result


@_implements(na.random.uniform)
def random_uniform(
        start: float | u.Quantity | na.AbstractScalar,
        stop: float | u.Quantity | na.AbstractScalar,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
) -> na.UncertainScalarArray:

    try:
        start = _normalize(start)
        stop = _normalize(stop)
    except _UncertainScalarTypeError:
        return NotImplemented

    return na.UncertainScalarArray(
        nominal=na.random.uniform(
            start=na.as_named_array(start.nominal),
            stop=na.as_named_array(stop.nominal),
            shape_random=shape_random,
            seed=seed,
        ),
        distribution=na.random.uniform(
            start=start.distribution,
            stop=stop.distribution,
            shape_random=shape_random,
            seed=seed + 1,
        ),
    )


@_implements(na.random.normal)
def random_normal(
        center: float | u.Quantity | na.AbstractScalar,
        width: float | u.Quantity | na.AbstractScalar,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
) -> na.UncertainScalarArray:

    try:
        center = _normalize(center)
        width = _normalize(width)
    except _UncertainScalarTypeError:
        return NotImplemented

    return na.UncertainScalarArray(
        nominal=na.random.normal(
            center=na.as_named_array(center.nominal),
            width=na.as_named_array(width.nominal),
            shape_random=shape_random,
            seed=seed,
        ),
        distribution=na.random.normal(
            center=center.distribution,
            width=width.distribution,
            shape_random=shape_random,
            seed=seed + 1,
        ),
    )
