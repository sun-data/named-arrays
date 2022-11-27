from typing import Sequence, Callable, Type
import numpy as np
import astropy.units as u

from named_arrays.core import broadcast_shapes
from . import core as scalars

__all__ = [
    'HANDLED_FUNCTIONS'
]

DEFAULT_FUNCTIONS = [
    np.ndim,
    np.argmin,
    np.nanargmin,
    np.min,
    np.nanmin,
    np.argmax,
    np.nanargmax,
    np.max,
    np.nanmax,
    np.sum,
    np.nansum,
    np.prod,
    np.nanprod,
    np.mean,
    np.nanmean,
    np.std,
    np.nanstd,
    np.var,
    np.nanvar,
    np.median,
    np.nanmedian,
    np.percentile,
    np.nanpercentile,
    np.quantile,
    np.nanquantile,
    np.all,
    np.any,
    np.ptp,
    np.count_nonzero,
]
HANDLED_FUNCTIONS = dict()


def _axis_normalized(
        a: scalars.AbstractScalarArray,
        axis: None | str | Sequence[str],
) -> tuple[str]:
    if axis is None:
        result = a.axes
    elif isinstance(axis, str):
        result = axis,
    else:
        result = tuple(axis)
    return result


def _calc_axes_new(
        a: scalars.AbstractScalarArray,
        axis: None | str | Sequence[str],
        *,
        keepdims: None | bool = None,
) -> tuple[str, ...]:
    if keepdims is None:
        keepdims = False
    if keepdims:
        return a.axes
    else:
        axis = _axis_normalized(a, axis)
        return tuple(ax for ax in a.axes if ax not in axis)


def array_function_default(
        func: Callable,
        a: scalars.AbstractScalarArray,
        axis: None | str | Sequence[str] = None,
        dtype: None | Type = None,
        out: None | scalars.ScalarArray = None,
        keepdims: None | bool = None,
        initial: None | bool | int | float | complex | u.Quantity = None,
        where: None | scalars.AbstractScalarArray = None,
):

    if out is not None:
        shape = broadcast_shapes(out.shape, a.shape)
        a = a.broadcast_to(shape)
    else:
        shape = a.shape

    kwargs = dict()

    if axis is not None:
        kwargs['axis'] = tuple(a.axes.index(ax) for ax in _axis_normalized(a, axis=axis) if ax in a.axes)
    if dtype is not None:
        kwargs['dtype'] = dtype
    if out is not None:
        kwargs['out'] = out.ndarray
    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    if initial is not None:
        kwargs['initial'] = initial
    if where is not None:
        kwargs['where'] = where.ndarray_aligned(shape)

    return scalars.ScalarArray(
        ndarray=func(a.ndarray, **kwargs),
        axes=_calc_axes_new(a, axis=axis, keepdims=keepdims),
    )


def implements(numpy_function: Callable):
    """Register an __array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.broadcast_to)
def broadcast_to(
        array: scalars.AbstractScalarArray,
        shape: dict[str, int],
):
    return scalars.ScalarArray(
        ndarray=np.broadcast_to(array.ndarray_aligned(shape), tuple(shape.values()), subok=True),
        axes=tuple(shape.keys()),
    )


@implements(np.shape)
def shape(
        a: scalars.AbstractScalarArray,
) -> dict[str, int]:
    return a.shape


@implements(np.transpose)
def transpose(
        a: scalars.AbstractScalarArray,
        axes: None | Sequence[str] = None,
):
    axes = tuple(reversed(a.axes)) if axes is None else axes
    return scalars.ScalarArray(
        ndarray=np.transpose(
            a=a.ndarray,
            axes=tuple(a.axes.index(ax) for ax in axes),
        ),
        axes=axes,
    )


@implements(np.array_equal)
def array_equal(
        a1: scalars.AbstractScalarArray,
        a2: scalars.AbstractScalarArray,
        equal_nan: bool = False,
) -> bool:
    return np.array_equal(
        a1=a1.ndarray,
        a2=a2.ndarray_aligned(a1.shape),
        equal_nan=equal_nan,
    )

