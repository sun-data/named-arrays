"""Compute properties of probability density functions."""

from typing import Sequence
import numpy as np
import named_arrays as na

__all__ = [
    "argpercentile",
    "percentile",
    "median",
    "iqr",
    "fwhm",
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


def fwhm(
    x: "na.AbstractScalar",
    f: "na.AbstractScalar",
    axis: None | str | Sequence[str] = None,
) -> "na.AbstractScalar":
    """
    Compute the full-width at half maximum (FWHM) of the function `f`
    evaluated at coordinates `x`.

    Uses linear interpolation to locate the two half-maximum crossings
    (one on each side of the peak), then returns their separation.

    Parameters
    ----------
    x
        The coordinates at which `f` is evaluated.
        Must have the same number of elements as `f` along `axis`.
    f
        The function values for which to compute the FWHM.
        Should have a single peak; the values at the endpoints of `axis`
        should be below half the maximum.
    axis
        The logical axis along which to compute the FWHM.
        If :obj:`None` (the default), `f` must have only one logical dimension.
    """
    axis = na.axis_normalized(f, axis)

    if len(axis) != 1:
        raise ValueError(
            f"only one logical axis allowed, got {axis=}"
        )

    axis = axis[0]

    half_max = np.max(f, axis=axis) / 2

    n = f.shape[axis]

    index_left_upper = np.argmax(f >= half_max, axis=axis)

    i_left_upper = index_left_upper[axis]
    i_left_lower = i_left_upper - 1

    index_left_lower = index_left_upper | {axis: i_left_lower}

    x_left_lower = x[index_left_lower]
    x_left_upper = x[index_left_upper]
    
    f_left_upper = f[index_left_upper]
    f_left_lower = f[index_left_lower]

    x_left_ptp = x_left_upper - x_left_lower
    f_left_ptp = f_left_upper - f_left_lower

    x_left = x_left_lower + x_left_ptp * (half_max - f_left_lower) / f_left_ptp

    f_reversed = f[{axis: slice(None, None, -1)}]

    index_right_reversed = np.argmax(f_reversed >= half_max, axis=axis)

    i_right_reversed = index_right_reversed[axis]

    i_right_lower = (n - 1) - i_right_reversed
    i_right_upper = i_right_lower + 1

    index_right_lower = index_right_reversed | {axis: i_right_lower}
    index_right_upper = index_right_reversed | {axis: i_right_upper}

    x_right_upper = x[index_right_upper]
    x_right_lower = x[index_right_lower]

    f_right_upper = f[index_right_upper]
    f_right_lower = f[index_right_lower]

    x_right_ptp = x_right_upper - x_right_lower
    f_right_ptp = f_right_upper - f_right_lower

    x_right = x_right_lower + x_right_ptp * (half_max - f_right_lower) / f_right_ptp

    return x_right - x_left
