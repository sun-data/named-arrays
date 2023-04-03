from __future__ import annotations
from typing import TypeVar
import astropy.units as u
import named_arrays as na

__all__ = [
    "RandomStartT",
    "RandomStopT",
    "uniform",
]


RandomStartT = TypeVar("RandomStartT", bound="float | complex | u.Quantity | na.AbstractArray")
RandomStopT = TypeVar("RandomStopT", bound="float | complex | u.Quantity | na.AbstractArray")


def uniform(
        start: RandomStartT,
        stop: RandomStopT,
        shape_random: dict[str, int] = None,
        seed: None | int = None
) -> RandomStartT | RandomStopT:
    """
    Draw samples from a uniform distribution

    Parameters
    ----------
    start
        Lower boundary of the output
    stop
        Upper boundary of the output
    shape_random
        Additional dimensions to be broadcast against :attr:`start` and :attr:`stop`
    seed
        Seed for the random number generator, can be provided for repeatability

    See Also
    --------
    :func:`numpy.random.uniform` : Equivalent numpy function

    :class:`named_arrays.ScalarUniformRandomSample` : Implicit array version of this function for scalars

    :class:`named_arrays.UncertainScalarUniformRandomSample` : Implicit array version of this function for uncertain
        scalars
    """

    return na._named_array_function(
        func=uniform,
        start=na.as_named_array(start),
        stop=na.as_named_array(stop),
        shape_random=shape_random,
        seed=seed,
    )
