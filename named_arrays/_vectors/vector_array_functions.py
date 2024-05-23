from typing import Callable, Sequence, Type
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_array_functions
from . import vectors

__all__ = [
    'SINGLE_ARG_FUNCTIONS',
    'ARRAY_CREATION_LIKE_FUNCTIONS',
    'SEQUENCE_FUNCTIONS',
    'DEFAULT_FUNCTIONS',
    'PERCENTILE_LIKE_FUNCTIONS',
    'ARG_REDUCE_FUNCTIONS',
    'FFT_LIKE_FUNCTIONS',
    'FFTN_LIKE_FUNCTIONS',
    'HANDLED_FUNCTIONS',
    'STACK_LIKE_FUNCTIONS',
]

SINGLE_ARG_FUNCTIONS = named_arrays._scalars.scalar_array_functions.SINGLE_ARG_FUNCTIONS
ARRAY_CREATION_LIKE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.ARRAY_CREATION_LIKE_FUNCTIONS
SEQUENCE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.SEQUENCE_FUNCTIONS
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
EMATH_FUNCTIONS = named_arrays._scalars.scalar_array_functions.EMATH_FUNCTIONS
STACK_LIKE_FUNCTIONS = [
    np.stack,
    np.concatenate,
]
HANDLED_FUNCTIONS = dict()


def array_functions_single_arg(
    func: Callable,
    a: na.AbstractVectorArray,
) -> na.AbstractExplicitVectorArray:
    components = a.components
    components = {c: func(components[c]) for c in components}
    return a.type_explicit.from_components(components)


def array_function_array_creation_like(
        func: Callable,
        prototype: na.AbstractVectorArray,
        dtype: None | type | np.dtype = None,
        order: str = "K",
        subok: bool = True,
        shape: dict[str, int] = None,
) -> na.AbstractVectorArray:

    components = prototype.components

    components_result = dict()
    for c in components:
        components_result[c] = func(
            na.as_named_array(components[c]),
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape,
        )

    return prototype.type_explicit.from_components(components_result)


def array_function_sequence(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        axis: str | na.AbstractVectorArray,
        num: int | na.AbstractVectorArray = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
        **kwargs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
) -> na.AbstractVectorArray:

    try:
        prototype = vectors._prototype(*args, *kwargs.values())
        args = tuple(vectors._normalize(arg, prototype) for arg in args)
        axis = vectors._normalize(axis, prototype)
        num = vectors._normalize(num, prototype)
        kwargs = {k: vectors._normalize(kwargs[k], prototype) for k in kwargs}
    except na.VectorTypeError:
        return NotImplemented

    components_args = tuple(arg.components for arg in args)
    components_axis = axis.components
    components_num = num.components
    components_kwargs = {k: kwargs[k].components for k in kwargs}

    components = {
        c: func(
            *tuple(na.as_named_array(components_arg[c]) for components_arg in components_args),
            axis=components_axis[c],
            num=components_num[c],
            endpoint=endpoint,
            dtype=dtype,
            **{k: na.as_named_array(components_kwargs[k][c]) for k in components_kwargs},
        )
        for c in prototype.components
    }

    return prototype.type_explicit.from_components(components)


def array_function_default(
        func: Callable,
        a: na.AbstractVectorArray,
        axis: None | str | Sequence[str] = None,
        dtype: None | type | np.dtype = np._NoValue,
        out: None | na.AbstractExplicitVectorArray = None,
        keepdims: bool = False,
        initial: None | bool | int | float | complex | u.Quantity = np._NoValue,
        where: na.AbstractScalarArray | na.AbstractVectorArray = np._NoValue,
) -> na.AbstractExplicitVectorArray:

    a = a.explicit
    shape = na.shape_broadcasted(a, where)

    axis_normalized = tuple(shape) if axis is None else (axis, ) if isinstance(axis, str) else axis

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument, {axis}, must be `None` or a subset of the broadcasted shape of `a` and "
                f"`where`, {shape} "
            )

    shape_base = {ax: shape[ax] for ax in axis_normalized}

    components = a.components
    components_out = out.components if isinstance(out, na.AbstractVectorArray) else {c: out for c in components}
    components_where = where.components if isinstance(where, na.AbstractVectorArray) else {c: where for c in components}

    kwargs_base = dict(
        axis=axis,
        keepdims=keepdims,
    )

    if dtype is not np._NoValue:
        kwargs_base["dtype"] = dtype
    if initial is not np._NoValue:
        kwargs_base["initial"] = initial

    result = a.prototype_vector
    for c in components:
        component = na.as_named_array(components[c])
        where_c = components_where[c]
        shape_c = na.broadcast_shapes(component.shape, na.shape(where_c), shape_base)
        kwargs = dict()
        if where_c is not np._NoValue:
            kwargs["where"] = where_c.broadcast_to(shape_c) if isinstance(where_c, na.AbstractArray) else where_c
        result.components[c] = func(
            component.broadcast_to(shape_c),
            out=components_out[c],
            **kwargs_base,
            **kwargs,
        )

    if out is not None:
        result = out

    return result


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

    a = a.explicit
    shape = a.shape

    axis_normalized = na.axis_normalized(a, axis)

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument, {axis}, must be `None` or a subset of the shape of `a`, {shape}"
            )

    shape_base = {ax: shape[ax] for ax in axis_normalized}

    components = a.components
    components_q = q.components if isinstance(q, na.AbstractVectorArray) else {c: q for c in components}
    components_out = out.components if isinstance(out, na.AbstractVectorArray) else {c: out for c in components}

    kwargs_base = dict(
        axis=axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )

    result = a.prototype_vector
    for c in components:
        component = na.as_named_array(components[c])
        shape_c = na.broadcast_shapes(component.shape, shape_base)
        result.components[c] = func(
            component.broadcast_to(shape_c),
            q=components_q[c],
            out=components_out[c],
            **kwargs_base,
        )

    if out is not None:
        result = out

    return result


def array_function_arg_reduce(
        func: Callable,
        a: na.AbstractVectorArray,
        axis: None | str | Sequence[str] = None,
) -> dict[str, na.AbstractVectorArray]:

    a = a.explicit
    components = a.components

    if axis is not None:
        if isinstance(axis, str):
            axis = (axis, )
        if not set(axis).issubset(a.axes):
            raise ValueError(f"Reduction axes {axis} are not a subset of the array axes {a.axes}")
    else:
        if not a.shape:
            raise ValueError(f"Applying {func} to zero-dimensional arrays is not supported")

    axis_normalized = na.axis_normalized(a, axis)

    components_result = dict()
    for c in components:
        component = na.as_named_array(components[c])
        shape_c = component.shape
        axis_c = tuple(ax for ax in axis_normalized if ax in shape_c)
        if axis_c:
            components_result[c] = func(
                a=component,
                axis=axis_c,
            )
        else:
            components_result[c] = dict()

    axes_result = tuple(set(ax for c in components_result for ax in components_result[c]))

    result = dict()
    for ax in axes_result:
        result[ax] = a.prototype_vector
        for c in components:
            result[ax].components[c] = components_result[c].get(ax, None)

    return result


def array_function_fft_like(
        func: Callable,
        a: na.AbstractVectorArray,
        axis: tuple[str, str],
        n: None | int = None,
        norm: str = "backward"
) -> na.AbstractExplicitVectorArray:

    a = a.explicit
    components = a.components
    shape = a.shape

    if axis[0] not in shape:
        raise ValueError(f"`axis` {axis[0]} not in array with shape {shape}")

    shape_axis = {axis[0]: shape[axis[0]]}

    components = {
        c: na.broadcast_to(components[c], na.broadcast_shapes(na.shape(components[c]), shape_axis))
        for c in components
    }

    result = a.prototype_vector
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

    a = a.explicit
    components = a.components
    shape_a = a.shape

    if not set(axes).issubset(shape_a):
        raise ValueError(f"`axes`, {tuple(axes)}, not a subset of array axes, {tuple(shape_a)}")

    shape_base = {ax: shape_a[ax] for ax in axes}

    result = a.prototype_vector
    for c in components:
        component = na.as_named_array(components[c])
        result.components[c] = func(
            a=na.broadcast_to(component, na.broadcast_shapes(component.shape, shape_base)),
            axes=axes,
            s=s,
            norm=norm,
        )

    return result


def array_function_emath(
    func: Callable,
    *args: na.AbstractScalar | na.AbstractVectorArray,
    **kwargs: na.AbstractScalar | na.AbstractVectorArray,
):
    try:
        prototype = vectors._prototype(*args, *kwargs.values())
        args = tuple(vectors._normalize(a, prototype) for a in args)
        kwargs = {k: vectors._normalize(kwargs[k], prototype) for k in kwargs}
    except vectors.VectorTypeError:     # pragma: nocover
        return NotImplemented

    args = tuple(a.components for a in args)
    kwargs = {k: kwargs[k].components for k in kwargs}

    result = {
        c: func(*[a[c] for a in args], **{k: kwargs[k][c] for k in kwargs})
        for c in prototype.components
    }

    return prototype.type_explicit.from_components(result)


def implements(numpy_function: Callable):
    """Register an ``__array_function__`` implementation for :class:`AbstractVectorArray` objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.copyto)
def copyto(
        dst: na.AbstractExplicitVectorArray,
        src: na.AbstractScalar | na.AbstractVectorArray,
        casting: str = "same_kind",
        where: bool | na.AbstractScalar | na.AbstractVectorArray = True,
) -> None:
    if not isinstance(dst, na.AbstractExplicitVectorArray):
        return NotImplemented

    components_dst = dst.components

    if isinstance(src, na.AbstractArray):
        if isinstance(src, na.AbstractVectorArray):
            if src.type_abstract == dst.type_abstract:
                components_src = src.components
            else:
                return NotImplemented
        elif isinstance(src, na.AbstractScalar):
            components_src = {c: src for c in components_dst}
        else:
            return NotImplemented
    else:
        components_src = {c: src for c in components_dst}

    if isinstance(where, na.AbstractArray):
        if isinstance(where, na.AbstractVectorArray):
            if where.type_abstract == where.type_explicit:
                components_where = where.components
            else:
                return NotImplemented
        elif isinstance(where, na.AbstractScalar):
            components_where = {c: where for c in components_dst}
        else:
            return NotImplemented
    else:
        components_where = {c: where for c in components_dst}

    for c in components_dst:
        try:
            np.copyto(
                dst=components_dst[c],
                src=components_src[c],
                casting=casting,
                where=components_where[c],
            )
        except TypeError:
            components_dst[c] = components_src[c]


@implements(np.broadcast_to)
def broadcast_to(
        array: na.AbstractVectorArray,
        shape: dict[str, int],
) -> na.AbstractExplicitVectorArray:
    components = array.components
    components_result = {c: na.broadcast_to(array=components[c], shape=shape) for c in components}
    return array.type_explicit.from_components(components_result)


@implements(np.transpose)
def transpose(
        a: na.AbstractVectorArray,
        axes: None | Sequence[str] = None,
) -> na.AbstractExplicitVectorArray:

    components = a.broadcasted.components
    return a.type_explicit.from_components({c: np.transpose(a=components[c], axes=axes) for c in components})


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
    return a.type_explicit.from_components(components_result)


@implements(np.reshape)
def reshape(a: na.AbstractVectorArray, newshape: dict[str, int]) -> na.AbstractExplicitVectorArray:
    components = a.broadcasted.components
    return a.type_explicit.from_components({c: np.reshape(a=components[c], newshape=newshape) for c in components})


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

    arrays = [
        vector_prototype.type_explicit.from_scalar(
            scalar=a,
            like=vector_prototype,
        ) if not isinstance(a, na.AbstractVectorArray) else a
        for a in arrays
    ]
    arrays = [a.broadcasted for a in arrays]

    components_arrays = [a.components for a in arrays]

    if out is None:
        components_out = vector_prototype.type_explicit.from_scalar(out, like=vector_prototype).components
    else:
        components_out = out.components

    components_result = dict()
    for c in components_arrays[0]:
        components_result[c] = func(
            [components[c] for components in components_arrays],
            axis=axis,
            out=components_out[c],
            dtype=dtype,
            casting=casting,
        )

    if out is None:
        return vector_prototype.type_explicit.from_components(components_result)
    else:
        return out


@implements(np.sort)
def sort(
        a: na.AbstractVectorArray,
        axis: None | str | Sequence[str],
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> na.AbstractExplicitVectorArray:

    a = a.explicit
    shape_a = a.shape
    components = a.components

    if axis is None:
        axis = tuple(shape_a)
        if not axis:
            return a
    elif isinstance(axis, str):
        axis = (axis, )
    else:
        if not axis:
            raise ValueError(f"if `axis` is a sequence, it must not be empty, got {axis}")

    if not set(axis).issubset(shape_a):
        raise ValueError(f"`axis`, {axis} is not a subset of `a.axes`, {a.axes}")

    shape_base = {ax: shape_a[ax] for ax in axis}

    result = a.prototype_vector
    for c in components:
        component = na.as_named_array(components[c])
        if any(ax in axis for ax in component.axes):
            component = component.broadcast_to(na.broadcast_shapes(component.shape, shape_base))
            result.components[c] = np.sort(component, axis=axis, kind=kind, order=order)
        else:
            result.components[c] = components[c]

    return result


@implements(np.argsort)
def argsort(
        a: na.AbstractVectorArray,
        axis: None | str | Sequence[str],
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> dict[str, na.AbstractExplicitVectorArray]:

    a = a.explicit
    shape_a = a.shape
    components = a.components

    if axis is None:
        axis = tuple(shape_a)
        if not axis:
            return dict()
    elif isinstance(axis, str):
        axis = (axis, )
    else:
        if not axis:
            raise ValueError(f"if `axis` is a sequence, it must not be empty, got {axis}")

    if not set(axis).issubset(shape_a):
        raise ValueError(f"`axis`, {axis} is not a subset of `a.axes`, {a.axes}")

    shape_base = {ax: shape_a[ax] for ax in axis}

    result = {ax: a.prototype_vector for ax in shape_a}
    for c in components:
        component = na.as_named_array(components[c])
        if any(ax in axis for ax in component.axes):
            component = component.broadcast_to(na.broadcast_shapes(component.shape, shape_base))
            result_c = np.argsort(component, axis=axis, kind=kind, order=order)
        else:
            result_c = dict()
        for ax in shape_a:
            result[ax].components[c] = result_c[ax] if ax in result_c else None

    return result


@implements(np.unravel_index)
def unravel_index(
        indices: na.AbstractVectorArray,
        shape: dict[str, int],
) -> dict[str, na.AbstractExplicitVectorArray]:

    indices = indices.explicit
    components = indices.components
    result = {ax: dict() for ax in indices.axes}

    for c in components:
        result_c = np.unravel_index(
            indices=components[c],
            shape=shape,
        )
        for ax in result_c:
            result[ax][c] = result_c[ax]

    result = {ax: indices.type_explicit.from_components(result[ax]) for ax in result}
    return result


@implements(np.array_equal)
def array_equal(
        a1: na.AbstractVectorArray,
        a2: na.AbstractVectorArray,
        equal_nan: bool = False,
) -> bool:

    if not a1.type_abstract == a2.type_abstract:
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

    if not a1.type_abstract == a2.type_abstract:
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


@implements(np.allclose)
def allclose(
        a: na.ArrayLike,
        b: na.ArrayLike,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
):
    if isinstance(a, na.AbstractVectorArray):
        components_a = a.components
        if isinstance(b, na.AbstractVectorArray):
            if a.type_abstract == b.type_abstract:
                components_b = b.components
            else:
                return NotImplemented
        else:
            components_b = {c: b for c in components_a}
    else:
        if isinstance(b, na.AbstractVectorArray):
            components_b = b.components
            components_a = {c: a for c in components_b}
        else:
            return NotImplemented

    result = True
    for c in components_a:
        result = result * np.allclose(
            a=components_a[c],
            b=components_b[c],
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    return result


@implements(np.nonzero)
def nonzero(a: na.AbstractVectorArray):
    a = a.explicit
    components = a.components

    components_accumulated = True
    for c in components:
        components_accumulated = components_accumulated & components[c]

    return np.nonzero(components_accumulated)


@implements(np.where)
def where(
    condition: na.AbstractScalar | na.AbstractVectorArray,
    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    y: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
) -> na.AbstractExplicitVectorArray:
    try:
        prototype = vectors._prototype(condition, x, y)
        condition = vectors._normalize(condition, prototype)
        x = vectors._normalize(x, prototype)
        y = vectors._normalize(y, prototype)
    except vectors.VectorTypeError:
        return NotImplemented

    components_condition = condition.components
    components_x = x.components
    components_y = y.components
    components_result = dict()

    for c in components_condition:
        components_result[c] = np.where(
            components_condition[c],
            components_x[c],
            components_y[c],
        )

    return prototype.type_explicit.from_components(components_result)


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
        return x.type_explicit.from_components(components_result)
    else:
        if not isinstance(x, na.AbstractExplicitArray):
            raise ValueError("can't write to an array that is not an instance of `named_array.AbstractExplictArray`")
        return x


@implements(np.repeat)
def repeat(
    a: na.AbstractVectorArray,
    repeats: int | na.AbstractScalarArray,
    axis: str,
) -> na.AbstractExplicitVectorArray:

    components = a.broadcasted.components

    components_result = dict()
    for c in components:
        components_result[c] = np.repeat(
            a=components[c],
            repeats=repeats,
            axis=axis,
        )

    return a.type_explicit.from_components(components_result)


@implements(np.diff)
def diff(
    a: na.AbstractVectorArray,
    axis: str,
    n: int = 1,
    prepend: None | float | na.AbstractScalar | na.AbstractVectorArray = None,
    append: None | float | na.AbstractScalar | na.AbstractVectorArray = None,
) -> na.AbstractExplicitVectorArray:

    try:
        prototype = vectors._prototype(a, prepend, append)
        a = vectors._normalize(a, prototype)
        prepend = vectors._normalize(prepend, prototype)
        append = vectors._normalize(append, prototype)
    except vectors.VectorTypeError:     # pragma: nocover
        return NotImplemented

    components_a = a.broadcasted.components
    components_prepend = prepend.components
    components_append = append.components

    result = dict()
    for c in components_a:
        result[c] = np.diff(
            a=components_a[c],
            axis=axis,
            n=n,
            prepend=components_prepend[c],
            append=components_append[c],
        )

    return prototype.type_explicit.from_components(result)
