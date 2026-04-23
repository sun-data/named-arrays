"""Compute properties of probability density functions."""

from typing import Sequence
import numpy as np
import named_arrays as na

__all__ = [
    "argpercentile",
    "percentile",
    "median",
    "iqr",
]


def argpercentile(
    a: "na.AbstractScalar",
    q: "float | na.AbstractScalar",
    axis: None | str | Sequence[str] =  None,
) -> "dict[str, na.AbstractScalar]":
    """
    Find the fractional, 1D index of an all-positive array, `a`,
    corresponding to the percentile `q`.

    This function finds the fractional index using linear interpolation.

    Parameters
    ----------
    a
        The all-positive array on which to compute the percentile.
        If `a` contains negatives, the result is undefined.
    q
        The percentile(s) to compute.
    axis
        The axis of `a` along which to compute the percentile.
        If :obj:`None` (the default), `a` must have only one logical dimension.
    """

    if not na.shape(a):
        raise ValueError(
            "cannot perform cumulative reduction on zero-dimensional array."
        )

    axis = na.axis_normalized(a, axis)

    if len(axis) != 1:
        raise ValueError(
            f"only one logical axis allowed, got {axis=}"
        )

    axis = axis[0]

    cs = np.cumulative_sum(a, axis=axis, include_initial=True)

    y = q * cs[{axis: ~0}] / 100

    i1 = np.argmax(cs > y, axis=axis)

    x1 = i1[axis]
    x0 = x1 - 1

    i0 = i1.copy()
    i0[axis] = x0

    y0 = cs[i0]
    y1 = cs[i1]

    x = (y - y0) / (y1 - y0) * (x1 - x0) + x0

    return {axis: x}


def percentile(
    x: "na.AbstractScalar",
    f: "na.AbstractScalar",
    q: "float | na.AbstractScalar",
    axis: None | str | Sequence[str] = None,
) -> "na.AbstractScalar":
    """
    Compute the percentile of the given probability mass function, `f`.

    Parameters
    ----------
    x
        The edges of each bin of the probability mass function.
        This should have one more element than `f` along `axis`.
    f
        The probability mass function to compute the percentile of.
        Does `not` need to be normalized.
    q
        The percentile(s) to compute.
    axis
        The logical axis corresponding to changing `x`.
    """
    axis = na.axis_normalized(f, axis)

    if len(axis) != 1:
        raise ValueError(
            f"only one logical axis allowed, got {axis=}"
        )

    axis = axis[0]

    index = argpercentile(
        a=f,
        q=q,
        axis=axis,
    )

    index = index[axis]

    result = na.interp(
        x=index,
        xp=na.arange(0, x.shape[axis], axis=axis),
        fp=x,
    )

    return result


def median(
    x: "na.AbstractScalar",
    f: "na.AbstractScalar",
    axis: None | str | Sequence[str] = None,
) -> "na.AbstractScalar":
    """
    Compute the median of the given probability mass function, `f`.

    Parameters
    ----------
    x
        The edges of each bin of the probability mass function.
        This should have one more element than `f` along `axis`.
    f
        The probability mass function to compute the percentile of.
        Does `not` need to be normalized.
    axis
        The logical axis corresponding to changing `x`.
    """
    return percentile(
        x=x,
        f=f,
        q=50,
        axis=axis,
    )


def iqr(
    x: "na.AbstractScalar",
    f: "na.AbstractScalar",
    axis: None | str | Sequence[str] = None,
) -> "na.AbstractScalar":
    """
    Compute the interquartile range of the given probability mass function, `f`.

    Parameters
    ----------
    x
        The edges of each bin of the probability mass function.
        This should have one more element than `f` along `axis`.
    f
        The probability mass function to compute the percentile of.
        Does `not` need to be normalized.
    axis
        The logical axis corresponding to changing `x`.
    """
    q1 = percentile(
        x=x,
        f=f,
        q=25,
        axis=axis,
    )
    q3 = percentile(
        x=x,
        f=f,
        q=75,
        axis=axis,
    )

    return q3 - q1
