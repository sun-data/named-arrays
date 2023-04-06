from typing import Callable
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions
from . import uncertainties

__all__ = [
    "RANDOM_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
]

RANDOM_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.RANDOM_FUNCTIONS
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


def random(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalar,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity | na.AbstractScalar,
) -> na.UncertainScalarArray:

    try:
        args = tuple(uncertainties._normalize(arg) for arg in args)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except na.UncertainScalarTypeError:
        return NotImplemented

    return na.UncertainScalarArray(
        nominal=func(
            *tuple(arg.nominal for arg in args),
            shape_random=shape_random,
            seed=seed,
            **{k: kwargs[k].nominal for k in kwargs},
        ),
        distribution=func(
            *tuple(arg.distribution for arg in args),
            shape_random=shape_random,
            seed=seed,
            **{k: kwargs[k].distribution for k in kwargs},
        )
    )
