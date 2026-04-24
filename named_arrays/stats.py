"""Statistical functions."""

from typing import Sequence, TypeVar
import numpy as np
import named_arrays as na

XT = TypeVar('XT', bound="na.AbstractArray")
YT = TypeVar('YT', bound="na.AbstractArray")

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
    """

    shape = na.shape_broadcasted(x, y)

    x = na.broadcast_to(x, shape)
    y = na.broadcast_to(y, shape)

    axis = na.axis_normalized(x, axis=axis)
    _axis = "rank"

    x = x.combine_axes(axis, _axis)
    y = y.combine_axes(axis, _axis)

    rank_x = np.argsort(np.argsort(x, axis=_axis)[_axis], axis=_axis)[_axis]
    rank_y = np.argsort(np.argsort(y, axis=_axis)[_axis], axis=_axis)[_axis]

    result = pearsonr(rank_x, rank_y, axis=_axis)

    return result

