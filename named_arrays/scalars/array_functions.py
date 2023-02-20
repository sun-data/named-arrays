from __future__ import annotations
from typing import Sequence, Callable, Type
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    'DEFAULT_FUNCTIONS',
    'PERCENTILE_LIKE_FUNCTIONS',
    'ARG_REDUCE_FUNCTIONS',
    'FFT_LIKE_FUNCTIONS',
    'FFTN_LIKE_FUNCTIONS',
    'HANDLED_FUNCTIONS',
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
FFT_LIKE_FUNCTIONS = [
    np.fft.fft,
    np.fft.ifft,
    np.fft.rfft,
    np.fft.irfft,
]
FFTN_LIKE_FUNCTIONS = [
    np.fft.fft2,
    np.fft.ifft2,
    np.fft.rfft2,
    np.fft.irfft2,
    np.fft.fftn,
    np.fft.ifftn,
    np.fft.rfftn,
    np.fft.irfftn,
]
HANDLED_FUNCTIONS = dict()


def array_function_default(
        func: Callable,
        a: na.AbstractScalarArray,
        axis: None | str | Sequence[str] = None,
        dtype: type | np.dtype = np._NoValue,
        out: None | na.ScalarArray = None,
        keepdims: bool = False,
        initial: bool | int | float | complex | u.Quantity = np._NoValue,
        where: bool | na.AbstractScalarArray = np._NoValue,
):
    a = a.array
    axes_a = a.axes

    if axis is not None:
        if not set(axis).issubset(axes_a):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of `a.axes`, "
                f"got {axis} for `axis`, but `{a.axes} for `a.axes`"
            )

    axis_normalized = na.axis_normalized(a, axis=axis)

    if out is not None:
        if not isinstance(out, na.ScalarArray):
            raise ValueError(f"`out` should be `None` or an instance of `{a.type_array}`, got `{type(out)}`")
        axes_ndarray = out.axes
        if not keepdims:
            axes_ndarray = axes_ndarray + tuple(ax for ax in axes_a if ax not in out.axes)
    else:
        axes_ndarray = axes_a

    kwargs = dict()
    kwargs["axis"] = tuple(axes_ndarray.index(ax) for ax in axis_normalized)
    if dtype is not np._NoValue:
        kwargs["dtype"] = dtype
    if out is not None and isinstance(out.ndarray, np.ndarray):
        kwargs["out"] = out.ndarray
    else:
        kwargs["out"] = None
    kwargs["keepdims"] = keepdims
    if initial is not np._NoValue:
        kwargs["initial"] = initial
    if where is not np._NoValue:
        if isinstance(where, na.AbstractArray):
            if isinstance(where, na.AbstractScalarArray):
                kwargs["where"] = where.ndarray_aligned(axes_ndarray)
            else:
                return NotImplemented
        else:
            kwargs["where"] = where

    axes_result = tuple(ax for ax in axes_ndarray if ax not in axis_normalized) if not keepdims else axes_ndarray

    result_ndarray = func(a.ndarray_aligned(axes_ndarray), **kwargs)

    if out is None:
        result = na.ScalarArray(
            ndarray=result_ndarray,
            axes=axes_result,
        )
    else:
        out.ndarray = result_ndarray
        result = out
    return result


def array_function_percentile_like(
        func: Callable,
        a: na.AbstractScalarArray,
        q: float | u.Quantity | na.AbstractScalarArray,
        axis: None | str | Sequence[str] = None,
        out: None | na.ScalarArray = None,
        overwrite_input: bool = np._NoValue,
        method: str = np._NoValue,
        keepdims: bool = False,
) -> na.ScalarArray:

    a = a.array
    axes_a = a.axes

    if axis is not None:
        if not set(axis).issubset(axes_a):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of `a.axes`, "
                f"got {axis} for `axis`, but `{a.axes} for `a.axes`"
            )

    q = q.array if isinstance(q, na.AbstractArray) else na.ScalarArray(q)
    axes_q = q.axes

    axis_union = set(a.axes) & set(q.axes)
    if axis_union:
        raise ValueError(f"'q' must not have any shared axes with 'a', but axes {axis_union} are shared")

    axis_normalized = na.axis_normalized(a, axis=axis)

    axes_result = axes_q + (tuple(ax for ax in axes_a if ax not in axis_normalized) if not keepdims else axes_a)

    kwargs = dict()
    kwargs["axis"] = tuple(axes_a.index(ax) for ax in axis_normalized)

    if out is not None:
        if not isinstance(out, na.ScalarArray):
            raise ValueError(f"`out` should be `None` or an instance of `named_arrays.ScalarArray`, got `{type(out)}`")
        kwargs["out"] = out.ndarray_aligned(axes_result)
    else:
        kwargs["out"] = out

    if overwrite_input is not np._NoValue:
        kwargs['overwrite_input'] = overwrite_input
    if method is not np._NoValue:
        kwargs['method'] = method
    kwargs['keepdims'] = keepdims

    result_ndarray = func(a.ndarray, q.ndarray, **kwargs)

    result = na.ScalarArray(
        ndarray=result_ndarray,
        axes=axes_result,
    )

    if out is not None:
        out.ndarray = result.ndarray_aligned(out.shape)
        result = out
    return result


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


def array_function_fft_like(
        func: Callable,
        a: na.AbstractScalarArray,
        axis: tuple[str, str],
        n: None | int = None,
        norm: str = "backward"
) -> na.ScalarArray:

    return na.ScalarArray(
        ndarray=func(
            a=a.ndarray,
            n=n,
            axis=a.axes.index(axis[0]),
            norm=norm,
        ),
        axes=tuple(axis[1] if ax == axis[0] else ax for ax in a.axes),
    )


def array_function_fftn_like(
        func: Callable,
        a: na.AbstractScalarArray,
        axes: dict[str, str],
        s: None | dict[str, int] = None,
        norm: str = "backward",
) -> na.ScalarArray:

    a = a.array
    shape = a.shape

    if not all(ax in shape for ax in axes):
        raise ValueError(f"Not all transform axes {axes} are in input array shape {shape}")

    if s is None:
        s = {ax: shape[ax] for ax in axes}

    if not axes.keys() == s.keys():
        raise ValueError(f"'axes' {axes} and 's' {s} must have the same keys")

    return na.ScalarArray(
        ndarray=func(
            a=a.ndarray,
            s={ax: s[ax] for ax in axes}.values(),
            axes=tuple(a.axes.index(ax) for ax in axes),
            norm=norm,
        ),
        axes=tuple(axes[ax] if ax in axes else ax for ax in a.axes),
    )


def implements(numpy_function: Callable):
    """Register an __array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.broadcast_to)
def broadcast_to(
        array: na.AbstractScalarArray,
        shape: dict[str, int],
):
    return na.ScalarArray(
        ndarray=np.broadcast_to(array.ndarray_aligned(shape), tuple(shape.values()), subok=True),
        axes=tuple(shape.keys()),
    )


@implements(np.transpose)
def transpose(
        a: na.AbstractScalarArray,
        axes: None | Sequence[str] = None,
) -> na.ScalarArray:

    if axes is not None:
        a = a.add_axes(axes)
    axes = tuple(reversed(a.axes) if axes is None else axes)
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

    if isinstance(source, str):
        source = (source,)
    if isinstance(destination, str):
        destination = (destination,)

    set_axis_diff = set(source) - set(axes)
    if set_axis_diff:
        raise ValueError(f"source axes {tuple(set_axis_diff)} not in array axes {axes}")

    for src, dest in zip(source, destination):
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
        *,
        dtype: str | np.dtype | Type = None,
        casting: str = "same_kind",
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
            out=out.transpose(axes_new).ndarray,
            dtype=dtype,
            casting=casting
        )
        return out
    else:
        return na.ScalarArray(
            ndarray=np.stack(
                arrays=arrays,
                axis=axis_ndarray,
                out=out,
                dtype=dtype,
                casting=casting,
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
        a1=na.as_named_array(a1).ndarray,
        a2=na.as_named_array(a2).ndarray,
        equal_nan=equal_nan,
    )


@implements(np.array_equiv)
def array_equiv(
        a1: na.AbstractScalarArray,
        a2: na.AbstractScalarArray,
) -> bool:
    shape = na.shape_broadcasted(a1, a2)
    return np.array_equiv(
        a1=na.as_named_array(a1).ndarray_aligned(shape),
        a2=na.as_named_array(a2).ndarray_aligned(shape),
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
        if not isinstance(x, na.AbstractExplicitArray):
            raise ValueError("can't write to an array that is not an instance of `named_array.AbstractExplictArray`")
        return x
