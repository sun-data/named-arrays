from __future__ import annotations
from typing import TypeVar, Sequence
import astropy.units as u
import named_arrays as na

__all__ = [
    "RandomLowT",
    "RandomHighT",
    "uniform",
    "normal",
    "poisson",
    "binomial",
    "gamma",
]


RandomLowT = TypeVar("RandomLowT", bound="float | complex | u.Quantity | na.AbstractArray")
RandomHighT = TypeVar("RandomHighT", bound="float | complex | u.Quantity | na.AbstractArray")
RandomCenterT = TypeVar("RandomCenterT", bound="float | complex | u.Quantity | na.AbstractArray")
RandomWidthT = TypeVar("RandomWidthT", bound="float | complex | u.Quantity | na.AbstractArray")
NumTrialsT = TypeVar("NumTrialsT", bound="int | na.AbstractArray")
ProbabilityT = TypeVar("ProbabilityT", bound="float | na.AbstractArray")
ShapeT = TypeVar("ShapeT", bound="float | na.AbstractArray")
ScaleT = TypeVar("ScaleT", bound="float | u.Quantity | na.AbstractArray")
ChoicesT = TypeVar("ChoicesT", bound="int | na.AbstractArray")


def uniform(
        low: RandomLowT,
        high: RandomHighT,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None
) -> RandomLowT | RandomHighT:
    """
    Draw samples from a uniform distribution

    Parameters
    ----------
    low
        Lower boundary of the output
    high
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
        low=na.as_named_array(low),
        high=na.as_named_array(high),
        shape_random=shape_random,
        seed=seed,
    )


def normal(
        loc: RandomCenterT,
        scale: RandomWidthT,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None
) -> RandomCenterT | RandomWidthT:
    """
    Draw samples from a normal distribution

    Parameters
    ----------
    loc
        The center of the distribution.
    scale
        The width of the distribution.
    shape_random
        Additional dimensions to be broadcast against :attr:`center` and :attr:`width`
    seed
        Seed for the random number generator, can be provided for repeatability.

    See Also
    --------
    :func:`numpy.random.normal` : Equivalent numpy function

    :class:`named_arrays.ScalarNormalRandomSample` : Implicit array version of this function for scalars

    :class:`named_arrays.UncertainScalarNormalRandomSample` : Implicit array version of this function for uncertain
        scalars
    """
    return na._named_array_function(
        func=normal,
        loc=na.as_named_array(loc),
        scale=na.as_named_array(scale),
        shape_random=shape_random,
        seed=seed,
    )


def poisson(
        lam: RandomCenterT,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None
) -> RandomCenterT:
    """
    Draw samples from a Poisson distribution

    Parameters
    ----------
    lam
        Expected number of events
    shape_random
        Additional dimensions to be broadcast against :attr:`center` and :attr:`width`
    seed
        Seed for the random number generator, can be provided for repeatability.

    See Also
    --------
    :func:`numpy.random.poisson` : Equivalent numpy function

    :class:`named_arrays.ScalarPoissonlRandomSample` : Implicit array version of this function for scalars

    :class:`named_arrays.UncertainScalarPoissonRandomSample` : Implicit array version of this function for uncertain
        scalars
    """
    return na._named_array_function(
        func=poisson,
        lam=na.as_named_array(lam),
        shape_random=shape_random,
        seed=seed,
    )


def binomial(
    n: NumTrialsT,
    p: ProbabilityT,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
) -> NumTrialsT | ProbabilityT:
    """
    Draw samples from a binomial distribution.

    Parameters
    ----------
    n
        The number of independent trials.
    p
        The probability of a trial being successful.
    shape_random
        Additional dimensions to be broadcast against ``n`` and ``p``.
    seed
        Optional seed for the random number generator,
        can be provided for repeatability.

    See Also
    --------
    :func:`numpy.random.binomial` : Equivalent numpy function
    """
    return na._named_array_function(
        func=binomial,
        n=n,
        p=na.as_named_array(p),
        shape_random=shape_random,
        seed=seed,
    )


def gamma(
    shape: ShapeT,
    scale: ScaleT = 1,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
) -> ShapeT | ScaleT:
    """
    Draw samples from a gamma distribution.

    Parameters
    ----------
    shape
        The shape parameter of the distribution.
    scale
        The scale parameter of the distribution.
    shape_random
        Additional dimensions to be broadcast against `shape` and `scale`.
    seed
        Optional seed for the random number generator,
        can be provided for repeatability.

    See Also
    --------
    :func:`numpy.random.gamma` : Equivalent numpy function
    """
    return na._named_array_function(
        func=gamma,
        shape=na.as_named_array(shape),
        scale=scale,
        shape_random=shape_random,
        seed=seed,
    )


def choice(
    a: ChoicesT,
    p: ProbabilityT = None,
    axis: None | str | Sequence[str] = None,
    replace: bool = True,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
) -> ChoicesT | ProbabilityT:
    """
    Draw random samples from a given array of choices.

    Parameters
    ----------
    a
        If an array, a random sample is created from its elements.
        If an ``int``, the random sample is generated as if it were
        ``np.arange(a)``
    p
        An optional list of probabilities for each element in `a`.
    axis
        The axes of `a` and `p` to flatten into a sequence of choices.
        This is useful if the axes of `a` or `p` intersect with the axes of
        `shape_random`.
    replace
        Whether the sample is drawn with or without replacement.
        Default is :obj:`True`, meaning each value in `a` can be selected
        multiple times.
    shape_random
        Additional dimensions to be broadcast against `a` and `p`.
    seed
        Optional seed for the random number generator,
        can be provided for repeatability.

    See Also
    --------
    :func:`numpy.random.choice` : Equivalent numpy function
    """
    return na._named_array_function(
        func=choice,
        a=a,
        p=p,
        axis=axis,
        replace=replace,
        shape_random=shape_random,
        seed=seed,
    )
