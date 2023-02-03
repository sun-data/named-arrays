from __future__ import annotations
from typing import Sequence, Callable, Type
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    'HANDLED_FUNCTIONS',
    'PERCENTILE_LIKE_FUNCTIONS',
]

DEFAULT_FUNCTIONS = [
    np.ndim,
    np.min,
    np.nanmin,
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
    np.all,
    np.any,
    np.ptp,
    np.count_nonzero,
]
PERCENTILE_LIKE_FUNCTIONS = [
    np.percentile,
    np.nanpercentile,
    np.quantile,
    np.nanquantile,
]
ARG_REDUCE_FUNCTIONS = [
    np.argmin,
    np.nanargmin,
    np.argmax,
    np.nanargmax,
]
HANDLED_FUNCTIONS = dict()


def _axis_normalized(
        a: na.AbstractScalarArray,
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
        a: na.AbstractScalarArray,
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
        a: na.AbstractScalarArray,
        axis: None | str | Sequence[str] = None,
        dtype: None | Type = None,
        out: None | na.ScalarArray = None,
        keepdims: None | bool = None,
        initial: None | bool | int | float | complex | u.Quantity = None,
        where: None | na.AbstractScalarArray = None,
):
    axis_normalized = _axis_normalized(a, axis=axis)
    if out is not None:
        axis_out = tuple(ax for ax in a.axes if ax not in axis_normalized) if not keepdims else a.axes
        out = out.transpose(axis_out)

    kwargs = dict()

    if axis is not None:
        kwargs['axis'] = tuple(a.axes.index(ax) for ax in axis_normalized if ax in a.axes)
    if dtype is not None:
        kwargs['dtype'] = dtype
    if out is not None:
        kwargs['out'] = out.ndarray
    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    if initial is not None:
        kwargs['initial'] = initial
    if where is not None and where is not np._NoValue:
        kwargs['where'] = where.ndarray_aligned(a.shape)

    return na.ScalarArray(
        ndarray=func(a.ndarray, **kwargs),
        axes=_calc_axes_new(a, axis=axis, keepdims=keepdims),
    )


def array_function_percentile_like(
        func: Callable,
        a: na.AbstractScalarArray,
        q: float | u.Quantity | na.AbstractScalarArray,
        axis: None | str | Sequence[str] = None,
        out: None | na.ScalarArray = None,
        overwrite_input: bool = False,
        method: str = 'linear',
        keepdims: bool = False,
) -> na.ScalarArray:

    if not isinstance(q, na.AbstractArray):
        q = na.ScalarArray(q)

    axis_union = set(a.axes) & set(q.axes)
    if axis_union:
        raise ValueError(f"'q' must not have any shared axes with 'a', but axes {axis_union} are shared")

    axis_normalized = _axis_normalized(a, axis=axis)
    if out is not None:
        axis_out = tuple(ax for ax in a.axes if ax not in axis_normalized) if not keepdims else a.axes
        axis_out = q.axes + axis_out
        out = out.transpose(axis_out)

    kwargs = dict()

    if axis is not None:
        kwargs['axis'] = tuple(a.axes.index(ax) for ax in axis_normalized if ax in a.axes)
    if out is not None:
        kwargs['out'] = out.ndarray
    kwargs['overwrite_input'] = overwrite_input
    kwargs['method'] = method
    if keepdims is not None:
        kwargs['keepdims'] = keepdims

    return na.ScalarArray(
        ndarray=func(a.ndarray, q.ndarray, **kwargs),
        axes=q.axes + _calc_axes_new(a, axis=axis, keepdims=keepdims),
    )


def array_function_arg_reduce(
        func: Callable,
        a: na.AbstractScalarArray,
        axis: None | str = None,
        out: None | dict[str, na.ScalarArray] = None,
        keepdims: None | bool = None,
) -> dict[str, na.ScalarArray]:

    if axis is not None:
        if axis not in a.axes:
            raise ValueError(f"Reduction axis '{axis}' not in array with axes {a.axes}")
    else:
        if not a.shape:
            raise ValueError(f"Applying {func} to zero-dimensional arrays is not supported")

    if out is not None:
        raise NotImplementedError(f"out keyword argument is not implemented for {func}")

    if keepdims:
        axis_out = a.axes
    else:
        if axis is not None:
            axis_out = tuple(ax for ax in a.axes if not ax == axis)
        else:
            axis_out = tuple()

    kwargs = dict()

    if axis is not None:
        kwargs['axis'] = a.axes.index(axis) if axis is not None else axis
    if keepdims is not None:
        kwargs['keepdims'] = keepdims

    indices = na.ScalarArray(
        ndarray=func(a.ndarray, **kwargs),
        axes=axis_out,
    )

    if axis is None:
        result = np.unravel_index(indices=indices, shape=a.shape)
    else:
        result = a.indices
        result[axis] = indices

    return result


def implements(numpy_function: Callable):
    """Register an __array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.convolve)
def convolve(
        a: na.ScalarLike,
        v: na.ScalarLike,
        mode: str = 'full',
) -> na.AbstractScalarArray:

    if not isinstance(a, na.AbstractArray):
        a = na.ScalarArray(a)
    if not isinstance(v, na.AbstractArray):
        v = na.ScalarArray(v)

    if a.ndarray is None or v.ndarray is None:
        return na.ScalarArray(None)

    shape_broadcasted = na.shape_broadcasted(a, v)
    if len(shape_broadcasted) > 1:
        raise ValueError(f"'a' and 'v' must broadcast to a 1D shape, got {shape_broadcasted}")

    result_ndarray = np.convolve(a.ndarray, v.ndarray, mode=mode)
    if len(shape_broadcasted) == 0:
        result_ndarray = result_ndarray[0]

    return na.ScalarArray(
        ndarray=result_ndarray,
        axes=tuple(shape_broadcasted.keys()),
    )


@implements(np.broadcast_to)
def broadcast_to(
        array: na.AbstractScalarArray,
        shape: dict[str, int],
):
    return na.ScalarArray(
        ndarray=np.broadcast_to(array.ndarray_aligned(shape), tuple(shape.values()), subok=True),
        axes=tuple(shape.keys()),
    )


@implements(np.shape)
def shape(
        a: na.AbstractScalarArray,
) -> dict[str, int]:
    return a.shape


@implements(np.transpose)
def transpose(
        a: na.AbstractScalarArray,
        axes: None | Sequence[str] = None,
) -> na.ScalarArray:

    if axes is not None:
        a = a.add_axes(axes)
    axes = tuple(reversed(a.axes)) if axes is None else axes
    return na.ScalarArray(
        ndarray=np.transpose(
            a=a.ndarray,
            axes=tuple(a.axes.index(ax) for ax in axes),
        ),
        axes=axes,
    )


@implements(np.moveaxis)
def moveaxis(
        a: na.AbstractScalarArray,
        source: str | Sequence[str],
        destination: str | Sequence[str],
) -> na.ScalarArray:

    axes = list(a.axes)

    types_sequence = (list, tuple,)
    if not isinstance(source, types_sequence):
        source = (source,)
    if not isinstance(destination, types_sequence):
        destination = (destination,)

    for src, dest in zip(source, destination):
        if src not in axes:
            raise ValueError(f"source axis {src} not in array axes {a.axes}")
        axes[axes.index(src)] = dest

    return na.ScalarArray(
        ndarray=a.ndarray,
        axes=tuple(axes)
    )


@implements(np.reshape)
def reshape(a: na.AbstractScalarArray, newshape: dict[str, int]) -> na.ScalarArray:
    return na.ScalarArray(
        ndarray=np.reshape(a.ndarray, tuple(newshape.values())),
        axes=tuple(newshape.keys()),
    )


@implements(np.linalg.inv)
def linalg_inv(a: na.AbstractScalarArray,):
    raise NotImplementedError(
        "np.linalg.inv not supported, use 'named_arrays.AbstractScalarArray.matrix_inverse()' instead"
    )


@implements(np.stack)
def stack(
        arrays: Sequence[bool | int | float | complex | str | u.Quantity | na.AbstractScalarArray],
        axis: str,
        out: None | na.ScalarArray = None,
) -> na.ScalarArray:
    arrays = [na.ScalarArray(arr) if not isinstance(arr, na.AbstractArray) else arr for arr in arrays]
    for array in arrays:
        if not isinstance(array, na.AbstractScalarArray):
            return NotImplemented
    shape = na.shape_broadcasted(*arrays)
    if axis in shape:
        raise ValueError(f"axis '{axis}' already in array")
    arrays = [arr.broadcast_to(shape).ndarray for arr in arrays]

    axes_new = (axis,) + tuple(shape.keys())
    axis_ndarray = 0

    if out is not None:
        np.stack(
            arrays=arrays,
            axis=axis_ndarray,
            out=out.transpose(axes_new).ndarray
        )
        return out
    else:
        return na.ScalarArray(
            ndarray=np.stack(
                arrays=arrays,
                axis=axis_ndarray,
                out=out
            ),
            axes=axes_new,
        )


@implements(np.concatenate)
def concatenate(
        arrays: Sequence[bool | int | float | complex | str | u.Quantity | na.AbstractScalarArray],
        axis: str,
        out: None | na.ScalarArray = None,
        dtype: None | str | Type | np.dtype = None,
        casting: str = "same_kind",
) -> na.ScalarArray:

    arrays = [na.ScalarArray(arr) if not isinstance(arr, na.AbstractArray) else arr for arr in arrays]
    for array in arrays:
        if not isinstance(array, na.AbstractScalarArray):
            return NotImplemented

    shapes = []
    for array in arrays:
        shape = array.shape
        shape[axis] = 1
        shapes.append(shape)

    shape_prototype = na.broadcast_shapes(*shapes)

    ndarrays = []
    for array in arrays:
        shape = shape_prototype.copy()
        if axis in array.axes:
            shape[axis] = array.shape[axis]
        array = array.broadcast_to(shape)
        ndarrays.append(array.ndarray)

    axis_index = tuple(shape_prototype).index(axis)

    if out is not None:
        np.concatenate(
            ndarrays,
            axis=axis_index,
            out=out.transpose(tuple(shape_prototype.keys())).ndarray,
            dtype=dtype,
            casting=casting,
        )
        return out
    else:
        return na.ScalarArray(
            ndarray=np.concatenate(
                ndarrays,
                axis=axis_index,
                out=out,
                dtype=dtype,
                casting=casting
            ),
            axes=tuple(shape_prototype.keys()),
        )


@implements(np.sort)
def sort(
        a: na.AbstractScalarArray,
        axis: None | str,
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> na.ScalarArray:

    if axis is not None:
        if axis not in a.axes:
            raise ValueError(f"axis '{axis}' not in input array with axes {a.axes}")

    return na.ScalarArray(
        ndarray=np.sort(
            a=a.ndarray,
            axis=a.axes.index(axis) if axis is not None else axis,
            kind=kind,
            order=order
        ),
        axes=a.axes if axis is not None else (a.axes_flattened, )
    )


@implements(np.argsort)
def argsort(
        a: na.AbstractScalarArray,
        axis: None | str,
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> dict[str, na.ScalarArray]:
    if axis is None:
        if not a.shape:
            raise ValueError("sorting zero-dimensional arrays is not supported")

        indices = na.ScalarArray(
            ndarray=np.argsort(
                a=a.ndarray,
                axis=axis,
                kind=kind,
                order=order
            ),
            axes=(a.axes_flattened, ),
        )
        return {a.axes_flattened: indices}

    else:
        if axis not in a.axes:
            raise ValueError(f"axis {axis} not in input array with axes {a.axes}")

        indices = na.ScalarArray(
            ndarray=np.argsort(
                a=a.ndarray,
                axis=a.axes.index(axis),
                kind=kind,
                order=order,
            ),
            axes=a.axes,
        )

        result = na.indices(a.shape)
        result[axis] = indices
        return result


@implements(np.unravel_index)
def unravel_index(indices: na.AbstractScalarArray, shape: dict[str, int]) -> dict[str, na.ScalarArray]:

    result_ndarray = np.unravel_index(indices=indices.ndarray, shape=tuple(shape.values()))
    result = dict()  # type: dict[str, na.ScalarArray]
    for axis, ndarray in zip(shape, result_ndarray):
        result[axis] = na.ScalarArray(
            ndarray=ndarray,
            axes=indices.axes,
        )
    return result


@implements(np.array_equal)
def array_equal(
        a1: na.AbstractScalarArray,
        a2: na.AbstractScalarArray,
        equal_nan: bool = False,
) -> bool:
    return np.array_equal(
        a1=a1.ndarray,
        a2=a2.ndarray,
        equal_nan=equal_nan,
    )


@implements(np.nonzero)
def nonzero(a: na.AbstractScalarArray):
    result = np.nonzero(a.ndarray)
    return {a.axes[r]: na.ScalarArray(result[r], axes=(f'{a.axes_flattened}_nonzero',)) for r, _ in enumerate(result)}


@implements(np.nan_to_num)
def nan_to_num(
        x: na.AbstractScalarArray,
        copy: bool = True,
        nan: float = 0.0,
        posinf: None | float = None,
        neginf: None | float = None,
):
    result_ndarray = np.nan_to_num(x.ndarray, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
    if copy:
        return na.ScalarArray(
            ndarray=result_ndarray,
            axes=x.axes,
        )
    else:
        return x
