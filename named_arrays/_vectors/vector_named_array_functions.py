from typing import Callable
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions
from . import vectors

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
    "RANDOM_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random"
]

ASARRAY_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.ASARRAY_LIKE_FUNCTIONS
RANDOM_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.RANDOM_FUNCTIONS
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


def asarray_like(
        func: Callable,
        a: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        dtype: None | type | np.dtype = None,
        order: None | str = None,
        *,
        like: None | na.AbstractVectorArray = None,
) -> None | na.AbstractVectorArray:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractVectorArray):
            if isinstance(like, na.AbstractArray):
                if isinstance(like, na.AbstractVectorArray):
                    if a.type_explicit == like.type_explicit:
                        components_a = a.components
                        components_like = like.components
                        type_like = like.type_explicit
                    else:
                        return NotImplemented
                elif isinstance(like, na.AbstractScalar):
                    components_a = a.components
                    components_like = {c: like for c in components_a}
                    type_like = a.type_explicit
                else:
                    return NotImplemented
            else:
                components_a = a.components
                components_like = {c: like for c in components_a}
                type_like = a.type_explicit
        elif isinstance(a, na.AbstractScalar):
            if isinstance(like, na.AbstractVectorArray):
                components_like = like.components
                components_a = {c: a for c in components_like}
                type_like = like.type_explicit
            else:
                return NotImplemented
        else:
            return NotImplemented
    else:
        if isinstance(like, na.AbstractVectorArray):
            components_like = like.components
            components_a = {c: a for c in components_like}
            type_like = like.type_explicit
        else:
            return NotImplemented

    return type_like.from_components({
        c: func(
            a=components_a[c],
            dtype=dtype,
            order=order,
            like=components_like[c],
        ) for c in components_like
    })


@_implements(na.arange)
def arange(
        start: float | complex | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        stop: float | complex | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        axis: str | na.AbstractVectorArray,
        step: int | na.AbstractVectorArray = 1,
):
    prototype = vectors._prototype(start, stop, axis, step)

    start = vectors._normalize(start, prototype)
    stop = vectors._normalize(stop, prototype)
    axis = vectors._normalize(axis, prototype)
    step = vectors._normalize(step, prototype)

    components_start = start.components
    components_stop = stop.components
    components_axis = axis.components
    components_step = step.components

    components = {
        c: na.arange(
            start=components_start[c],
            stop=components_stop[c],
            axis=components_axis[c],
            step=components_step[c],
        )
        for c in components_start
    }

    return prototype.type_explicit.from_components(components)


@_implements(na.unit)
def unit(a: na.AbstractVectorArray) -> None | u.UnitBase | na.AbstractVectorArray:
    components = a.components
    components = {c: na.unit(components[c]) for c in components}
    iter_components = iter(components)
    component_0 = components[next(iter_components)]
    if all(component_0 == components[c] for c in components):
        return component_0
    else:
        components = {c: 1 if components[c] is None else 1 * components[c] for c in components}
        return a.type_explicit.from_components(components,)


@_implements(na.unit_normalized)
def unit_normalized(a: na.AbstractVectorArray) -> u.UnitBase | na.AbstractVectorArray:
    components = a.components
    components = {c: na.unit_normalized(components[c]) for c in components}
    iter_components = iter(components)
    component_0 = components[next(iter_components)]
    if all(component_0 == components[c] for c in components):
        return component_0
    else:
        return a.type_explicit.from_components(components)


def random(
        func: Callable,
        *args: float | u.Quantity |na.AbstractScalar | na.AbstractVectorArray,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity |na.AbstractScalar | na.AbstractVectorArray,
):
    try:
        prototype = vectors._prototype(*args, *kwargs.values())
        args = tuple(vectors._normalize(arg, prototype) for arg in args)
        kwargs = {k: vectors._normalize(kwargs[k], prototype) for k in kwargs}
    except na.VectorTypeError:
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
