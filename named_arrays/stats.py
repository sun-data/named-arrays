"""Statistical functions."""

from typing import Sequence, TypeVar
import numpy as np
import named_arrays as na

__all__ = [
    "pearsonr",
    "spearmanr",
    "rankdata",
]

XT = TypeVar('XT', bound="na.AbstractArray")
YT = TypeVar('YT', bound="na.AbstractArray")
AT = TypeVar('AT', bound="na.AbstractScalarArray")


def _take(
    a: "na.AbstractScalarArray",
    index: "na.AbstractScalarArray",
    axis: str,
) -> "na.AbstractScalarArray":
    """Gather elements of `a` along `axis` using `index`, like :func:`numpy.take_along_axis`."""
    indices = na.indices(na.shape(a))
    indices[axis] = index
    return a[indices]

def pearsonr(
    x: XT,
    y: YT,
    axis: None | str | Sequence[str] = None,
    where: bool | XT | YT = True,
) -> XT | YT:
    """
    Computes the Pearson correlation coefficient between two arrays.

    Similar to :func:`scipy.stats.pearsonr`.

    Parameters
    ----------
    x
        The first input array.
    y
        The second input array.
    axis
        The axis or axes along which to compute the correlation coefficient.
    where
        The elements of `x` and `y` to consider when computing the correlation.
    """

    kwargs = dict(
        axis=axis,
        where=where,
    )

    mean_x = np.mean(x, **kwargs)
    mean_y = np.mean(y, **kwargs)

    std_x = np.std(x, **kwargs)
    std_y = np.std(y, **kwargs)

    cov = (x - mean_x) * (y - mean_y)

    result = np.mean(cov, **kwargs) / (std_x * std_y)

    return result


def spearmanr(
    x: XT,
    y: YT,
    axis: None | str | Sequence[str] = None,
    where: bool | XT | YT = True,
) -> XT | YT:
    """
    Computes the Spearman correlation coefficient between two arrays.

    Similar to :func:`scipy.stats.spearmanr`.

    Parameters
    ----------
    x
        The first input array.
    y
        The second input array.
    axis
        The axis or axes along which to compute the correlation coefficient.
    where
        The elements of `x` and `y` to consider when computing the correlation.
    """

    shape = na.shape_broadcasted(x, y, where)

    x = na.broadcast_to(x, shape)
    y = na.broadcast_to(y, shape)
    where = na.broadcast_to(where, shape)

    axis = na.axis_normalized(x, axis=axis)
    axis_flat = na.flatten_axes(axis)

    x = x.combine_axes(axis, axis_flat)
    y = y.combine_axes(axis, axis_flat)
    where = where.combine_axes(axis, axis_flat)

    rank_x = rankdata(x, axis=axis_flat, where=where)
    rank_y = rankdata(y, axis=axis_flat, where=where)

    result = pearsonr(rank_x, rank_y, axis=axis_flat, where=where)

    return result


def rankdata(
    a: AT,
    axis: None | str | Sequence[str] = None,
    method: str = "average",
    where: "bool | na.AbstractScalarArray" = True,
) -> AT:
    """
    Assign ranks to the data along the given axis, dealing with ties according
    to the chosen `method`.

    This is a :mod:`named_arrays` implementation of :func:`scipy.stats.rankdata`.

    Parameters
    ----------
    a
        The input array to rank.
    axis
        The axis or axes along which to compute the ranks.
        If :obj:`None` (the default), the ranks are computed over a flattened
        version of all the axes.
    method
        The convention used to assign ranks to tied values.

        - ``"average"``: The average of the ranks spanned by the tied values.
        - ``"min"``: The minimum of the ranks spanned by the tied values
          (also known as competition ranking).
        - ``"max"``: The maximum of the ranks spanned by the tied values.
        - ``"dense"``: Like ``"min"``, but the rank of the next distinct value
          is always one greater than the previous group (no gaps).
    where
        A boolean mask selecting which elements to include in the ranking.
        Excluded elements are assigned a rank of NaN.

    See Also
    --------
    :func:`scipy.stats.rankdata`: The equivalent Scipy function.
    """
    methods = ("average", "min", "max", "dense")
    if method not in methods:
        raise ValueError(f"unrecognized {method=}, must be one of {methods}")
    a = a.explicit
    a = na.value(a)

    masked = isinstance(where, na.AbstractArray)
    if masked:
        shape = na.shape_broadcasted(a, where)
        a = na.broadcast_to(a, shape)
        where = na.broadcast_to(where, shape)
        a = np.where(where, a.astype(float), np.nan)

    axis = na.axis_normalized(a, axis)
    if len(axis) == 1:
        axis_flat = axis[0]
    else:
        axis_flat = na.flatten_axes(axis)
        a = a.combine_axes(axis, axis_flat)
        if masked:
            where = where.combine_axes(axis, axis_flat)

    n = a.shape[axis_flat]

    sorter = np.argsort(a, axis=axis_flat)[axis_flat]
    inverse = np.argsort(sorter, axis=axis_flat)[axis_flat]
    a_sorted = _take(a, sorter, axis_flat)

    index = na.arange(0, n, axis=axis_flat)

    is_new_group = a_sorted[{axis_flat: slice(1, n)}] != a_sorted[{axis_flat: slice(0, n - 1)}]
    first = na.broadcast_to(
        na.ScalarArray(np.array([True]), axes=(axis_flat,)),
        {**na.shape(is_new_group), axis_flat: 1},
    )
    is_new_group = na.concatenate([first, is_new_group], axis=axis_flat)
    group = np.cumsum(is_new_group.astype(int), axis=axis_flat)

    # Scatter the one-past-the-end index of each group, which is the number of
    # elements less than or equal to the value of that group. Since the groups
    # are contiguous in sorted order, the start index of a group (the number of
    # elements strictly less than its value) is the end index of the prior group.
    shape_count = {**na.shape(a_sorted), axis_flat: n + 1}
    count = na.ScalarArray(np.zeros(tuple(shape_count.values())), axes=tuple(shape_count.keys()))
    index_count = na.indices(shape_count)
    index_count[axis_flat] = group
    count[index_count] = (index + 1).astype(float)

    count_leq = _take(count, group, axis_flat)
    count_less = _take(count, group - 1, axis_flat)

    if method == "average":
        rank_sorted = (count_less + count_leq + 1) / 2
    elif method == "min":
        rank_sorted = count_less + 1
    elif method == "max":
        rank_sorted = count_leq
    else:  # method == "dense"
        rank_sorted = group

    result = _take(rank_sorted, inverse, axis_flat)

    if masked:
        result = np.where(where, result, np.nan)

    return result

