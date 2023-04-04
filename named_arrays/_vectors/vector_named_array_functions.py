from typing import TypeVar, Callable
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions

__all__ = [
    "VectorPrototypeT",
    "RANDOM_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random"
]

VectorPrototypeT = TypeVar("VectorPrototypeT", bound=na.AbstractVectorArray)

RANDOM_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.RANDOM_FUNCTIONS
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


def random(
        func: Callable,
        *args: float | u.Quantity |na.AbstractScalar | na.AbstractVectorArray,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity |na.AbstractScalar | na.AbstractVectorArray,
):
    try:
        prototype = _prototype(*args, *kwargs.values())
        args = tuple(_normalize(arg, prototype) for arg in args)
        kwargs = {k: _normalize(kwargs[k], prototype) for k in kwargs}
    except _VectorTypeError:
        return NotImplemented

    components_prototype = prototype.components

    components_args = {c: tuple(arg.components[c] for arg in args) for c in components_prototype}
    components_kwargs = {c: {k: kwargs[k].components[c] for k in kwargs} for c in components_prototype}

    components = {
        c: func(
            *components_args[c],
            shape_random=shape_random,
            seed=seed,
            **components_kwargs[c],
        )
        for c in prototype.components
    }

    return prototype.type_explicit.from_components(components)
