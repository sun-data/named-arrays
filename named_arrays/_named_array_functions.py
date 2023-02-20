from __future__ import annotations
from typing import Sequence, overload, Type
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    'ndim',
    'shape',
    'broadcast_to',
    'stack',
    'concatenate',
    'add_axes',
]


def ndim(a: na.AbstractArray) -> int:
    return np.ndim(a)


def shape(a: na.ArrayLike) -> dict[str, int]:
    if not isinstance(a, na.AbstractArray):
        a = na.ScalarArray(a)
    return np.shape(a)


@overload
def broadcast_to(array: float | complex | np.ndarray | u.Quantity, shape: dict[str, int]) -> na.ScalarArray:
    ...


@overload
def broadcast_to(array: na.AbstractScalarArray, shape: dict[str, int]) -> na.ScalarArray:
    ...


def broadcast_to(array, shape):
    if not isinstance(array, na.AbstractArray):
        array = na.ScalarArray(array)
    return np.broadcast_to(array=array, shape=shape)


def stack(
        arrays: Sequence[na.ArrayLike],
        axis: str,
        out: None | na.AbstractExplicitArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: None | str = "same_kind",
) -> na.AbstractArray:
    if not any(isinstance(a, na.AbstractArray) for a in arrays):
        arrays = list(arrays)
        arrays[0] = na.ScalarArray(arrays[0])
    return np.stack(
        arrays=arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
    )


def concatenate(
        arrays: Sequence[na.ArrayLike],
        axis: str,
        out: None | na.AbstractExplicitArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: None | str = "same_kind",
) -> na.AbstractArray:
    if not any(isinstance(a, na.AbstractArray) for a in arrays):
        arrays = list(arrays)
        arrays[0] = na.ScalarArray(arrays[0])
    return np.concatenate(
        arrays=arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
    )


def add_axes(array: na.ArrayLike, axes: str | Sequence[str]):
    if not isinstance(array, na.AbstractArray):
        array = na.ScalarArray(array)
    return array.add_axes(axes)
