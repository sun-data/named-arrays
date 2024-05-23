from __future__ import annotations
from typing import Sequence, Callable, Type
import numpy as np
import astropy.units as u
import named_arrays as na
from . import scalars

__all__ = [
    'SINGLE_ARG_FUNCTIONS',
    'ARRAY_CREATION_LIKE_FUNCTIONS',
    'DEFAULT_FUNCTIONS',
    'PERCENTILE_LIKE_FUNCTIONS',
    'ARG_REDUCE_FUNCTIONS',
    'FFT_LIKE_FUNCTIONS',
    'FFTN_LIKE_FUNCTIONS',
    "EMATH_FUNCTIONS",
    'HANDLED_FUNCTIONS',
]
SINGLE_ARG_FUNCTIONS = [
    np.real,
    np.imag,
]
ARRAY_CREATION_LIKE_FUNCTIONS = [
    np.empty_like,
    np.zeros_like,
    np.ones_like,
]
SEQUENCE_FUNCTIONS = [
    np.linspace,
    np.logspace,
    np.geomspace,
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
EMATH_FUNCTIONS = [
    np.emath.sqrt,
    np.emath.log,
    np.emath.log2,
    np.emath.logn,
    np.emath.log10,
    np.emath.power,
    np.emath.arccos,
    np.emath.arcsin,
    np.emath.arctanh,
]
HANDLED_FUNCTIONS = dict()


def array_function_single_arg(
    func: Callable,
    a: na.AbstractScalarArray,
) -> na.ScalarArray:
    return a.type_explicit(
        ndarray=func(a.ndarray),
        axes=a.axes,
    )


def array_function_array_creation_like(
        func: Callable,
        prototype: na.AbstractScalarArray,
        dtype: None | type | np.dtype = None,
        order: str = "K",
        subok: bool = True,
        shape: dict[str, int] = None,
):
    if shape is None:
        shape = prototype.shape

    return prototype.type_explicit(
        ndarray=func(
            prototype.ndarray,
            dtype=dtype,
            order=order,
            subok=subok,
            shape=tuple(shape.values()),
        ),
        axes=tuple(shape.keys()),
    )


def array_function_sequence(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalarArray,
        axis: str,
        num: int = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
        **kwargs: float | u.Quantity | na.AbstractScalarArray,
):
    try:
        args = tuple(scalars._normalize(arg) for arg in args)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:
        return NotImplemented

    shape = na.shape_broadcasted(*args, *kwargs.values())

    args = tuple(arg.ndarray_aligned(shape) for arg in args)
    kwargs = {k: kwargs[k].ndarray_aligned(shape) for k in kwargs}

    if isinstance(axis, na.AbstractArray):
        if isinstance(axis, na.AbstractScalarArray):
            axis = axis.ndarray
        else:
            return NotImplemented

    return na.ScalarArray(
        ndarray=func(
            *args,
            num=num,
            endpoint=endpoint,
            dtype=dtype,
            axis=~0,
            **kwargs,
        ),
        axes=tuple(shape) + (axis, ),
    )


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
    a = a.explicit
    shape = na.shape_broadcasted(a, where)

    axis_normalized = na.axis_normalized(a, axis=axis)

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of the broadcasted shape of `a` and `where`, "
                f"got {axis} for `axis`, but `{shape} for `shape`"
            )

    if out is not None:
        if not isinstance(out, na.ScalarArray):
            raise ValueError(f"`out` should be `None` or an instance of `{a.type_explicit}`, got `{type(out)}`")
        axes_ndarray = out.axes
        if not keepdims:
            axes_ndarray = axes_ndarray + tuple(ax for ax in shape if ax not in out.axes)
    else:
        axes_ndarray = tuple(shape)

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

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalarArray):
            a = a.explicit
        else:
            return NotImplemented
    else:
        a = na.ScalarArray(a)

    axes_a = a.axes

    axis_normalized = na.axis_normalized(a, axis=axis)

    if axis is not None:
        if not set(axis_normalized).issubset(axes_a):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of `a.axes`, "
                f"got {axis} for `axis`, but `{a.axes} for `a.axes`"
            )

    if isinstance(q, na.AbstractArray):
        if isinstance(q, na.AbstractScalarArray):
            q = q.explicit
        else:
            return NotImplemented
    else:
        q = na.ScalarArray(q)

    axes_q = q.axes

    axis_union = set(a.axes) & set(q.axes)
    if axis_union:
        raise ValueError(f"'q' must not have any shared axes with 'a', but axes {axis_union} are shared")

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
        axis: None | str | Sequence[str] = None,
) -> dict[str, na.ScalarArray]:

    a = a.explicit
    shape_a = a.shape

    if axis is not None:
        if isinstance(axis, str):
            axis = (axis, )
        if not set(axis).issubset(a.axes):
            raise ValueError(f"Reduction axes {axis} are not a subset of the array axes {a.axes}")
    else:
        if not a.shape:
            raise ValueError(f"Applying {func} to zero-dimensional arrays is not supported")

    if axis is not None:
        axis_flattened = na.flatten_axes(axis)
        a_flattened = a.combine_axes(axes=axis, axis_new=axis_flattened)
        axis_ndarray = a_flattened.axes.index(axis_flattened)

        indices = na.ScalarArray(
            ndarray=func(a_flattened.ndarray, axis=axis_ndarray),
            axes=tuple(ax for ax in a_flattened.axes if ax != axis_flattened),
        )

        result = np.unravel_index(indices, shape={ax: shape_a[ax] for ax in axis})
        for ax in shape_a:
            if ax not in axis:
                result[ax] = na.ScalarArrayRange(0, shape_a[ax], axis=ax)

    else:
        index = na.ScalarArray(func(a.ndarray, axis=axis))
        result = np.unravel_index(index, shape=shape_a)

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

    a = a.explicit
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


def array_function_emath(
    func: Callable,
    *args: na.AbstractScalarArray,
    **kwargs: na.AbstractScalarArray,
) -> na.ScalarArray:
    try:
        args = tuple(scalars._normalize(a) for a in args)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except scalars.ScalarTypeError:     # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(*args, *kwargs.values())

    args = tuple(a.ndarray_aligned(shape) for a in args)
    kwargs = {k: kwargs[k].ndarray_aligned(shape) for k in  kwargs}

    return na.ScalarArray(
        ndarray=func(*args, **kwargs),
        axes=tuple(shape),
    )

def implements(numpy_function: Callable):
    """Register an __array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.copyto)
def copyto(
        dst: na.ScalarArray,
        src: na.AbstractScalarArray,
        casting: str = "same_kind",
        where: bool | na.AbstractScalarArray = True,
):

    if not isinstance(dst, na.ScalarArray):
        return NotImplemented

    shape = dst.shape

    if isinstance(src, na.AbstractArray):
        if isinstance(src, na.AbstractScalarArray):
            src_ndarray = src.ndarray_aligned(shape)
        else:
            return NotImplemented
    else:
        src_ndarray = src

    if isinstance(where, na.AbstractArray):
        if isinstance(where, na.AbstractScalarArray):
            where_ndarray = where.ndarray_aligned(shape)
        else:
            return NotImplemented
    else:
        where_ndarray = where

    try:
        np.copyto(
            dst=dst.ndarray,
            src=src_ndarray,
            casting=casting,
            where=where_ndarray,
        )
    except TypeError:
        dst.ndarray = src.ndarray


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

    arrays = [na.ScalarArray(arr) if not isinstance(arr, na.AbstractArray) else arr.explicit for arr in arrays]
    for array in arrays:
        if not isinstance(array, na.AbstractScalarArray):
            return NotImplemented

    if any(axis not in array.shape for array in arrays):
        raise ValueError(f"axis '{axis}' must be present in all the input arrays, got {[a.axes for a in arrays]}")

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
        axis: None | str | Sequence[str],
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> na.ScalarArray:

    a = a.explicit

    if axis is not None:
        if isinstance(axis, str):
            axis = (axis, )
        else:
            axis = tuple(axis)

        if not axis:
            raise ValueError(f"if `axis` is a sequence, it must not be empty, got {axis}")

        if not set(axis).issubset(a.axes):
            raise ValueError(f"`axis`, {axis} is not a subset of `a.axes`, {a.axes}")

        axis_new = na.flatten_axes(axis)
        a = a.combine_axes(axes=axis, axis_new=axis_new)
        axis = axis_new

    else:
        if not a.shape:
            return a

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

    a = a.explicit
    shape_a = a.shape

    if axis is not None:
        if isinstance(axis, str):
            axis = (axis, )
        else:
            axis = tuple(axis)

        if not axis:
            raise ValueError(f"if `axis` is a sequence, it must not be empty, got {axis}")

        if not set(axis).issubset(a.axes):
            raise ValueError(f"`axis`, {axis} is not a subset of `a.axes`, {a.axes}")

    else:
        if not a.shape:
            return dict()
        else:
            axis = a.axes

    axis_flattened = na.flatten_axes(axis)
    a_flattened = a.combine_axes(axes=axis, axis_new=axis_flattened)

    indices_ndarray = np.argsort(
        a=a_flattened.ndarray,
        axis=a_flattened.axes.index(axis_flattened),
        kind=kind,
        order=order,
    )
    indices = na.ScalarArray(indices_ndarray, axes=a_flattened.axes)

    result = np.unravel_index(indices, shape={ax: shape_a[ax] for ax in axis})

    for ax in shape_a:
        if ax not in result:
            result[ax] = na.ScalarArrayRange(
                start=0,
                stop=shape_a[ax],
                axis=ax,
            )

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


@implements(np.allclose)
def allclose(
        a: na.ScalarLike,
        b: na.ScalarLike,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
) -> bool:

    shape = na.shape_broadcasted(a, b)

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalarArray):
            a = a.ndarray_aligned(shape)
        else:
            return NotImplemented

    if isinstance(b, na.AbstractArray):
        if isinstance(b, na.AbstractScalarArray):
            b = b.ndarray_aligned(shape)
        else:
            return NotImplemented

    return np.allclose(
        a=a,
        b=b,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


@implements(np.nonzero)
def nonzero(a: na.AbstractScalarArray):
    a = a.explicit
    if not a.shape:
        return dict()
    axes = a.axes
    axes_flattened = a.axes_flattened
    result = np.nonzero(a.ndarray)
    return {axes[r]: na.ScalarArray(result[r], axes=(axes_flattened, )) for r, _ in enumerate(result)}


@implements(np.where)
def where(
    condition: na.AbstractScalarArray,
    x: float | u.Quantity | na.AbstractScalarArray,
    y: float | u.Quantity | na.AbstractScalarArray,
) -> na.ScalarArray:
    try:
        condition = scalars._normalize(condition)
        x = scalars._normalize(x)
        y = scalars._normalize(y)
    except scalars.ScalarTypeError:
        return NotImplemented

    shape = na.shape_broadcasted(condition, x, y)

    return condition.type_explicit(
        ndarray=np.where(
            condition.ndarray_aligned(shape),
            x.ndarray_aligned(shape),
            y.ndarray_aligned(shape),
        ),
        axes=tuple(shape),
    )


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


@implements(np.repeat)
def repeat(
    a: na.AbstractScalarArray,
    repeats: int | na.AbstractScalarArray,
    axis: str,
) -> na.ScalarArray:

    if axis not in a.axes:
        raise ValueError(
            f"{axis=} must be a member of {a.axes=}"
        )

    repeats = na.as_named_array(repeats)

    return a.type_explicit(
        ndarray=np.repeat(
            a=a.ndarray,
            repeats=repeats.ndarray,
            axis=a.axes.index(axis),
        ),
        axes=a.axes,
    )


@implements(np.diff)
def diff(
    a: na.AbstractScalarArray,
    axis: str,
    n: int = 1,
    prepend: None | float | na.AbstractScalarArray = None,
    append: None | float | na.AbstractScalarArray = None,
) -> na.ScalarArray:

    shape = a.shape

    if axis not in shape:
        raise ValueError(
            f"{axis=} must be a member of {a.axes=}"
        )

    try:
        a = scalars._normalize(a)
        prepend = scalars._normalize(prepend) if prepend is not None else None
        append = scalars._normalize(append) if append is not None else None
    except scalars.ScalarTypeError: # pragma: nocover
        return NotImplemented

    shape_ends = {ax: 1 if ax == axis else shape[ax] for ax in shape}

    kwargs = dict()
    if prepend is not None:
        kwargs["prepend"] = na.broadcast_to(prepend, shape_ends).ndarray
    if append is not None:
        kwargs["append"] = na.broadcast_to(append, shape_ends).ndarray

    return a.type_explicit(
        ndarray=np.diff(
            a=a.ndarray,
            n=n,
            axis=tuple(shape).index(axis),
            **kwargs,
        ),
        axes=tuple(shape),
    )
