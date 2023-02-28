from typing import Callable, Sequence, Type
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
    'STACK_LIKE_FUNCTIONS',
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
STACK_LIKE_FUNCTIONS = [
    np.stack,
    np.concatenate,
]
HANDLED_FUNCTIONS = dict()


def array_function_default(
        func: Callable,
        a: na.AbstractVectorArray,
        axis: None | str | Sequence[str] = None,
        dtype: None | Type = None,
        out: None | na.AbstractExplicitVectorArray = None,
        keepdims: None | bool = None,
        initial: None | bool | int | float | complex | u.Quantity = None,
        where: None | na.AbstractScalarArray | na.AbstractVectorArray = None,
) -> na.AbstractExplicitVectorArray:

    if out is None:
        out = a.type_array.from_scalar(out)
    if not isinstance(where, na.AbstractVectorArray):
        where = a.type_array.from_scalar(where)

    components = a.components
    components_out = out.components
    components_where = where.components
    components_result = dict()

    kwargs_base = dict()
    if axis is not None:
        kwargs_base['axis'] = axis
    if dtype is not None:
        kwargs_base['dtype'] = dtype
    if keepdims is not None:
        kwargs_base['keepdims'] = keepdims
    if initial is not None:
        kwargs_base['initial'] = initial

    for c in components:
        kwargs = kwargs_base.copy()
        if components_out[c] is not None:
            kwargs['out'] = components_out[c]
        if components_where[c] is not None:
            kwargs['where'] = components_where[c]
        components_result[c] = func(na.as_named_array(components[c]), **kwargs)

    return a.type_array.from_components(components_result)


def array_function_percentile_like(
        func: Callable,
        a: na.AbstractVectorArray,
        q: float | u.Quantity | na.AbstractScalarArray | na.AbstractVectorArray,
        axis: None | str | Sequence[str] = None,
        out: None | na.AbstractExplicitVectorArray = None,
        overwrite_input: bool = False,
        method: str = 'linear',
        keepdims: bool = False,
) -> na.AbstractExplicitVectorArray:

    if not isinstance(q, na.AbstractVectorArray):
        q = a.type_array.from_scalar(q)
    if out is None:
        out = a.type_array.from_scalar(out)

    components = a.components
    components_q = q.components
    components_out = out.components

    kwargs_base = dict()
    if axis is not None:
        kwargs_base['axis'] = axis
    kwargs_base['overwrite_input'] = overwrite_input
    kwargs_base['method'] = method
    if keepdims is not None:
        kwargs_base['keepdims'] = keepdims

    components_result = dict()
    for c in components:
        kwargs = kwargs_base.copy()
        print('components_out[c]', components_out[c])
        if components_out[c] is not None and hasattr(components_out[c], "__getitem__"):
            kwargs['out'] = components_out[c]
        components_result[c] = func(components[c], components_q[c], **kwargs)

    return a.type_array.from_components(components_result)


def array_function_arg_reduce(
        func: Callable,
        a: na.AbstractVectorArray,
        axis: None | str = None,
        out: None = None,
        keepdims: None | bool = None,
) -> dict[str, na.AbstractVectorArray]:

    a = a.broadcasted
    components = a.components

    result = {ax: a.type_array() for ax in a.axes}

    for c in components:
        result_c = func(
            a=na.as_named_array(components[c]),
            axis=axis,
            out=out,
            keepdims=keepdims,
        )
        for ax in result_c:
            result[ax].components[c] = result_c[ax]

    return result


def array_function_fft_like(
        func: Callable,
        a: na.AbstractVectorArray,
        axis: str,
        n: None | int = None,
        norm: str = "backward"
) -> na.AbstractExplicitVectorArray:

    a = a.array
    components = a.components
    shape = a.shape

    if axis not in shape:
        raise ValueError(f"transform axis {axis} not in input array with shape {shape}")

    shape_axis = {axis[0]: shape[axis[0]]}

    components = {c: na.broadcast_to(components[c], na.shape(components[c] | shape_axis)) for c in components}

    result = a.type_array()
    for c in components:
        result.components[c] = func(
            a=components[c],
            axis=axis,
            n=n,
            norm=norm,
        )

    return result


def array_function_fftn_like(
        func: Callable,
        a: na.AbstractVectorArray,
        axes: dict[str, str],
        s: None | dict[str, int] = None,
        norm: str = "backward",
) -> na.AbstractExplicitVectorArray:

    a = a.array
    components = a.components
    shape = a.shape

    if not all(ax in shape for ax in axes):
        raise ValueError(f"Not all transform axes {axes} are in input array shape {shape}")

    shape_base = {ax: shape[ax] for ax in axes}

    components = {c: na.broadcast_to(components[c], na.shape(components[c]) | shape_base) for c in components}

    result = a.type_array()
    for c in components:
        result.components[c] = func(
            a=components[c],
            axes=axes,
            s=s,
            norm=norm,
        )

    return result


def implements(numpy_function: Callable):
    """Register an ``__array_function__`` implementation for :class:`AbstractVectorArray` objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.broadcast_to)
def broadcast_to(
        array: na.AbstractVectorArray,
        shape: dict[str, int],
) -> na.AbstractExplicitVectorArray:
    components = array.components
    components_result = {c: na.broadcast_to(array=components[c], shape=shape) for c in components}
    return array.type_array.from_components(components_result)


@implements(np.transpose)
def transpose(
        a: na.AbstractVectorArray,
        axes: None | Sequence[str] = None,
) -> na.AbstractExplicitVectorArray:

    components = a.broadcasted.components
    return a.type_array.from_components({c: np.transpose(a=components[c], axes=axes) for c in components})


@implements(np.moveaxis)
def moveaxis(
        a: na.AbstractVectorArray,
        source: str | Sequence[str],
        destination: str | Sequence[str],
) -> na.AbstractExplicitVectorArray:

    axes = a.axes

    if isinstance(source, str):
        source = (source, )
    if isinstance(destination, str):
        destination = (destination, )

    set_axis_diff = set(source) - set(axes)
    if set_axis_diff:
        raise ValueError(f"source axes {tuple(set_axis_diff)} not in array axes {axes}")

    components = a.components
    components_result = dict()
    for c in components:
        shape_c = na.shape(components[c])
        source_and_destination = tuple((src, dest) for src, dest in zip(source, destination) if src in shape_c)
        components_result[c] = np.moveaxis(
            a=components[c],
            source=tuple(src_and_dest[0] for src_and_dest in source_and_destination),
            destination=tuple(src_and_dest[1] for src_and_dest in source_and_destination),
        )
    return a.type_array.from_components(components_result)


@implements(np.reshape)
def reshape(a: na.AbstractVectorArray, newshape: dict[str, int]) -> na.AbstractExplicitVectorArray:
    components = a.broadcasted.components
    return a.type_array.from_components({c: np.reshape(a=components[c], newshape=newshape) for c in components})


@implements(np.linalg.inv)
def linalg_inv(a: na.AbstractVectorArray,):
    raise NotImplementedError(
        "np.linalg.inv not supported for instances of 'named_arrays.AbstractVectorArray'"
    )


def array_function_stack_like(
        func: Callable,
        arrays: Sequence[bool | int | float | complex | str | u.Quantity | na.AbstractScalar | na.AbstractVectorArray],
        axis: str,
        out: None | na.AbstractExplicitVectorArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: str = "same_kind",
) -> na.AbstractExplicitVectorArray:

    for array in arrays:
        if isinstance(array, na.AbstractVectorArray):
            vector_prototype = array
            break

    arrays = [vector_prototype.type_array.from_scalar(a) if not isinstance(a, na.AbstractVectorArray) else a
              for a in arrays]
    shape = na.shape_broadcasted(*arrays)
    arrays = [a.broadcast_to(shape) for a in arrays]

    components_arrays = [a.components for a in arrays]

    if out is None:
        components_out = vector_prototype.type_array.from_scalar(out).components
    else:
        components_out = out.components

    components_result = dict()
    for c in components_arrays[0]:
        components_result[c] = func(
            arrays=[components[c] for components in components_arrays],
            axis=axis,
            out=components_out[c],
            dtype=dtype,
            casting=casting,
        )

    if out is None:
        return vector_prototype.type_array.from_components(components_result)
    else:
        return out


@implements(np.sort)
def sort(
        a: na.AbstractVectorArray,
        axis: None | str,
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> na.AbstractExplicitVectorArray:

    components = a.components
    components_result = dict()

    for c in components:

        if not na.shape(components[c]):
            components_result[c] = components[c]
        else:
            components_result[c] = np.sort(
                a=components[c],
                axis=axis,
                kind=kind,
                order=order,
            )

    return a.type_array.from_components(components_result)


@implements(np.argsort)
def argsort(
        a: na.AbstractVectorArray,
        axis: None | str,
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> dict[str, na.AbstractExplicitVectorArray]:

    a = a.broadcasted
    components = a.components

    if axis is None:
        result = {a.axes_flattened: dict()}
        for c in components:
            result[a.axes_flattened][c] = np.argsort(
                a=components[c],
                axis=axis,
                kind=kind,
                order=order,
            )[a.axes_flattened]

    else:
        result = {ax: dict() for ax in a.axes}
        for c in components:
            result_c = np.argsort(
                a=components[c],
                axis=axis,
                kind=kind,
                order=order,
            )
            for ax in result_c:
                result[ax][c] = result_c[ax]

    result = {ax: a.type_array.from_components(result[ax]) for ax in result}
    return result


@implements(np.unravel_index)
def unravel_index(
        indices: na.AbstractVectorArray,
        shape: dict[str, int],
) -> dict[str, na.AbstractExplicitVectorArray]:

    indices = indices.array
    components = indices.components
    result = {ax: dict() for ax in indices.axes}

    for c in components:
        result_c = np.unravel_index(
            indices=components[c],
            shape=shape,
        )
        for ax in result_c:
            result[ax][c] = result_c[ax]

    result = {ax: indices.type_array.from_components(result[ax]) for ax in result}
    return result


@implements(np.array_equal)
def array_equal(
        a1: na.AbstractVectorArray,
        a2: na.AbstractVectorArray,
        equal_nan: bool = False,
) -> bool:

    if not a1.type_array_abstract == a2.type_array_abstract:
        return False

    components_a1 = a1.components
    components_a2 = a2.components

    result = True
    for c in components_a1:
        result = result and np.array_equal(
            a1=components_a1[c],
            a2=components_a2[c],
            equal_nan=equal_nan,
        )

    return result


@implements(np.array_equiv)
def array_equiv(
        a1: na.AbstractVectorArray,
        a2: na.AbstractVectorArray,
) -> bool:

    if not a1.type_array_abstract == a2.type_array_abstract:
        return False

    components_a1 = a1.components
    components_a2 = a2.components

    result = True
    for c in components_a1:
        result = result and np.array_equiv(
            a1=components_a1[c],
            a2=components_a2[c],
        )

    return result

@implements(np.nonzero)
def nonzero(a: na.AbstractVectorArray):
    a = a.broadcasted
    components = a.components

    result = {ax: dict() for ax in a.axes}
    for c in components:
        result_c = np.nonzero(components[c])
        for ax in result_c:
            result[ax][c] = result_c[ax]

    result = {ax: a.type_array.from_components(result[ax]) for ax in result}
    return result


@implements(np.nan_to_num)
def nan_to_num(
        x: na.AbstractVectorArray,
        copy: bool = True,
        nan: float = 0.0,
        posinf: None | float = None,
        neginf: None | float = None,
):
    components = x.components
    components_result = dict()

    for c in components:
        components_result[c] = np.nan_to_num(
            x=components[c],
            copy=copy,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
        )

    if copy:
        return x.type_array.from_components(components_result)
    else:
        if not isinstance(x, na.AbstractExplicitArray):
            raise ValueError("can't write to an array that is not an instance of `named_array.AbstractExplictArray`")
        return x
