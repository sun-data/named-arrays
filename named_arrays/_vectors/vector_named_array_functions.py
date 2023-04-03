from typing import TypeVar, Callable
import astropy.units as u
import named_arrays as na

__all__ = [
    "VectorPrototypeT",
    "HANDLED_FUNCTIONS",
]

VectorPrototypeT = TypeVar("VectorPrototypeT", bound=na.AbstractVectorArray)

HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


class _VectorTypeError(TypeError):
    pass


def _prototype(*arrays: float | u.Quantity | na.AbstractArray) -> na.AbstractVectorArray:

    for array in arrays:
        if isinstance(array, na.AbstractVectorArray):
            return array

    raise _VectorTypeError


def _normalize(
        a: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        prototype: VectorPrototypeT,
) -> VectorPrototypeT:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractVectorArray):
            if a.type_abstract == prototype.type_abstract:
                result = a
            else:
                raise _VectorTypeError
        elif isinstance(a, na.AbstractScalar):
            result = prototype.type_explicit.from_scalar(a)
        else:
            raise _VectorTypeError
    else:
        result = prototype.type_explicit.from_scalar(a)

    return result


@_implements(na.random.uniform)
def random_uniform(
        start: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        stop: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
) -> na.AbstractExplicitVectorArray:

    try:
        prototype = _prototype(start, stop)
        start = _normalize(start, prototype)
        stop = _normalize(stop, prototype)
    except _VectorTypeError:
        return NotImplemented

    components_start = start.components
    components_stop = stop.components
    components_seed = {c: seed + 100 * i for i, c in enumerate(components_start)}
    components = {
        c: na.random.uniform(
            start=components_start[c],
            stop=components_stop[c],
            shape_random=shape_random,
            seed=components_seed[c],
        )
        for c in components_start
    }

    return prototype.type_explicit.from_components(components)

