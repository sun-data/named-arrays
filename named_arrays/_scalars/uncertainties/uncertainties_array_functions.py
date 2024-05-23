from typing import Callable, Sequence
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_array_functions
from . import uncertainties

__all__ = [
    'SINGLE_ARG_FUNCTIONS',
    'ARRAY_CREATION_LIKE_FUNCTIONS',
    'SEQUENCE_FUNCTIONS',
    'DEFAULT_FUNCTIONS',
    'PERCENTILE_LIKE_FUNCTIONS',
    'ARG_REDUCE_FUNCTIONS',
    'FFT_LIKE_FUNCTIONS',
    'FFTN_LIKE_FUNCTIONS',
    'EMATH_FUNCTIONS',
    'STACK_LIKE_FUNCTIONS',
    'HANDLED_FUNCTIONS',
]

SINGLE_ARG_FUNCTIONS = named_arrays._scalars.scalar_array_functions.SINGLE_ARG_FUNCTIONS
ARRAY_CREATION_LIKE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.ARRAY_CREATION_LIKE_FUNCTIONS
SEQUENCE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.SEQUENCE_FUNCTIONS
DEFAULT_FUNCTIONS = named_arrays._scalars.scalar_array_functions.DEFAULT_FUNCTIONS
PERCENTILE_LIKE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.PERCENTILE_LIKE_FUNCTIONS
ARG_REDUCE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.ARG_REDUCE_FUNCTIONS
FFT_LIKE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.FFT_LIKE_FUNCTIONS
FFTN_LIKE_FUNCTIONS = named_arrays._scalars.scalar_array_functions.FFTN_LIKE_FUNCTIONS
EMATH_FUNCTIONS = named_arrays._scalars.scalar_array_functions.EMATH_FUNCTIONS
STACK_LIKE_FUNCTIONS = [np.stack, np.concatenate]
HANDLED_FUNCTIONS = dict()


def array_functions_single_arg(
    func: Callable,
    a: na.AbstractUncertainScalarArray,
) -> na.UncertainScalarArray:
    return a.type_explicit(
        nominal=func(a.nominal),
        distribution=func(a.distribution),
    )


def array_function_array_creation_like(
        func: Callable,
        prototype: na.AbstractUncertainScalarArray,
        dtype: None | type | np.dtype = None,
        order: str = "K",
        subok: bool = True,
        shape: dict[str, int] = None,
):
    prototype = prototype.explicit

    if shape is not None:
        shape_distribution = shape | {prototype.axis_distribution: prototype.num_distribution}
    else:
        shape_distribution = None

    return prototype.type_explicit(
        nominal=func(
            na.as_named_array(prototype.nominal),
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape,
        ),
        distribution=func(
            na.as_named_array(prototype.distribution),
            dtype=dtype,
            order=order,
            subok=subok,
            shape=shape_distribution,
        ),
    )


def array_function_sequence(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalarArray | na.AbstractUncertainScalarArray,
        axis: str,
        num: int = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
        **kwargs: float | u.Quantity | na.AbstractScalarArray,
) -> na.AbstractUncertainScalarArray:

    try:
        args = tuple(uncertainties._normalize(arg) for arg in args)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:
        return NotImplemented

    return na.UncertainScalarArray(
        nominal=func(
            *tuple(na.as_named_array(arg.nominal) for arg in args),
            axis=axis,
            num=num,
            endpoint=endpoint,
            dtype=dtype,
            **{k: na.as_named_array(kwargs[k].nominal) for k in kwargs},
        ),
        distribution=func(
            *tuple(na.as_named_array(arg.distribution) for arg in args),
            axis=axis,
            num=num,
            endpoint=endpoint,
            dtype=dtype,
            **{k: na.as_named_array(kwargs[k].distribution) for k in kwargs},
        ),
    )


def array_function_default(
        func: Callable,
        a: na.AbstractUncertainScalarArray,
        axis: None | str | Sequence[str] = None,
        dtype: type | np.dtype = np._NoValue,
        out: None | na.UncertainScalarArray = None,
        keepdims: bool = False,
        initial: bool | int | float | complex | u.Quantity = np._NoValue,
        where: na.AbstractArray = np._NoValue,
) -> na.UncertainScalarArray:

    a = a.broadcasted
    shape_a = a.shape

    kwargs = dict()
    kwargs_nominal = dict()
    kwargs_distribution = dict()

    kwargs["axis"] = tuple(shape_a.keys()) if axis is None else axis

    if dtype is not np._NoValue:
        kwargs["dtype"] = dtype

    if out is not None:
        if not isinstance(out, na.UncertainScalarArray):
            raise ValueError(f"`out` must be `None or an instance of `{a.type_explicit}`, got `{type(out)}`")
        kwargs_nominal["out"] = out.nominal
        kwargs_distribution["out"] = out.distribution
    else:
        kwargs["out"] = out

    kwargs["keepdims"] = keepdims

    if initial is not np._NoValue:
        kwargs["initial"] = initial

    if where is not np._NoValue:
        if isinstance(where, na.AbstractArray):
            if isinstance(where, na.AbstractScalar):
                if isinstance(where, na.AbstractUncertainScalarArray):
                    kwargs_nominal["where"] = where.nominal
                    kwargs_distribution["where"] = where.distribution
                else:
                    kwargs["where"] = where
            else:
                return NotImplemented
        else:
            kwargs["where"] = where

    kwargs_nominal = kwargs | kwargs_nominal
    kwargs_distribution = kwargs | kwargs_distribution

    result_nominal = func(na.as_named_array(a.nominal), **kwargs_nominal)
    result_distribution = func(a.distribution, **kwargs_distribution)

    if out is None:
        result = na.UncertainScalarArray(
            nominal=result_nominal,
            distribution=result_distribution,
        )
    else:
        out.nominal = result_nominal
        out.distribution = result_distribution
        result = out
    return result


def array_function_percentile_like(
        func: Callable,
        a: float | u.Quantity | na.AbstractScalarArray | na.AbstractUncertainScalarArray,
        q: float | u.Quantity | na.AbstractScalarArray | na.AbstractUncertainScalarArray,
        axis: None | str | Sequence[str] = None,
        out: None | na.ScalarArray = None,
        overwrite_input: bool = np._NoValue,
        method: str = np._NoValue,
        keepdims: bool = False,
):
    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalar):
            if isinstance(a, na.AbstractScalarArray):
                a = na.UncertainScalarArray(a, a)
            elif isinstance(a, na.AbstractUncertainScalarArray):
                pass
            else:
                return NotImplemented
        else:
            return NotImplemented
    else:
        a = na.UncertainScalarArray(a, a)

    if isinstance(q, na.AbstractArray):
        if isinstance(q, na.AbstractScalar):
            if isinstance(q, na.AbstractScalarArray):
                q_nominal = q_distribution = q
            elif isinstance(q, na.AbstractUncertainScalarArray):
                q_nominal = q.nominal
                q_distribution = q.distribution
            else:
                return NotImplemented
        else:
            return NotImplemented
    else:
        q_nominal = q_distribution = q

    shape_a = a.shape

    axis_normalized = na.axis_normalized(a, axis)

    if axis is not None:
        if not set(axis_normalized).issubset(shape_a):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of `a.axes`, "
                f"got {axis} for `axis`, but `{a.axes} for `a.axes`"
            )

    shape_base = {ax: shape_a[ax] for ax in axis_normalized}

    kwargs = dict()
    kwargs_nominal = dict()
    kwargs_distribution = dict()

    kwargs["axis"] = axis_normalized

    if out is not None:
        if not isinstance(out, na.UncertainScalarArray):
            raise ValueError(f"`out` must be an instance of `{a.type_explicit}`, got `{type(out)}`")
        kwargs_nominal["out"] = out.nominal
        kwargs_distribution["out"] = out.distribution
    else:
        kwargs["out"] = out

    if overwrite_input is not np._NoValue:
        kwargs["overwrite_input"] = overwrite_input
    if method is not np._NoValue:
        kwargs["method"] = method
    kwargs["keepdims"] = keepdims

    kwargs_nominal = kwargs | kwargs_nominal
    kwargs_distribution = kwargs | kwargs_distribution

    result_nominal = func(
        a=na.broadcast_to(a.nominal, na.broadcast_shapes(na.shape(a.nominal), shape_base)),
        q=q_nominal,
        **kwargs_nominal,
    )
    result_distribution = func(
        a=na.broadcast_to(a.distribution, na.broadcast_shapes(na.shape(a.distribution), shape_base)),
        q=q_distribution,
        **kwargs_distribution,
    )

    if out is None:
        result = na.UncertainScalarArray(
            nominal=result_nominal,
            distribution=result_distribution,
        )
    else:
        out.nominal = result_nominal
        out.distribution = result_distribution
        result = out

    return result


def array_function_arg_reduce(
        func: Callable,
        a: na.AbstractUncertainScalarArray,
        axis: None | str = None,
) -> dict[str, na.UncertainScalarArray]:

    a = a.broadcasted

    result_nominal = func(
        a=a.nominal,
        axis=axis,
    )

    if axis is None:
        axis_flattened = na.flatten_axes(a.shape)
        result_distribution = func(
            a=a.distribution.combine_axes(a.shape, axis_new=axis_flattened),
            axis=axis_flattened,
        )
        result_distribution_axis = result_distribution.pop(axis_flattened)
        result_distribution = np.unravel_index(result_distribution_axis, a.shape) | result_distribution

    else:
        result_distribution = func(
            a=a.distribution,
            axis=axis,
        )

    result = dict()
    for ax in result_distribution:
        result[ax] = na.UncertainScalarArray(
            nominal=result_nominal[ax] if ax in result_nominal else 0,
            distribution=result_distribution[ax].add_axes(na.UncertainScalarArray.axis_distribution),
        )

    return result


def array_function_fft_like(
        func: Callable,
        a: na.AbstractUncertainScalarArray,
        axis: tuple[str, str],
        n: None | int = None,
        norm: str = "backward"
) -> na.UncertainScalarArray:

    a = a.explicit
    shape_a = a.shape

    if axis[0] not in shape_a:
        raise ValueError(f"`axis` {axis[0]} not in array with shape {shape_a}")

    nominal = a.nominal
    distribution = a.distribution

    shape_base = {axis[0]: shape_a[axis[0]]}
    nominal = na.broadcast_to(a.nominal, na.broadcast_shapes(na.shape(nominal), shape_base))
    distribution = na.broadcast_to(distribution, na.broadcast_shapes(na.shape(distribution), shape_base))

    kwargs = dict(
        n=n,
        axis=axis,
        norm=norm,
    )

    return na.UncertainScalarArray(
        nominal=func(a=nominal, **kwargs),
        distribution=func(a=distribution, **kwargs)
    )


def array_function_fftn_like(
        func: Callable,
        a: na.AbstractUncertainScalarArray,
        axes: dict[str, str],
        s: None | dict[str, int] = None,
        norm: str = "backward",
) -> na.UncertainScalarArray:

    a = a.explicit
    shape_a = a.shape

    if not set(axes).issubset(shape_a):
        raise ValueError(f"`axes`, {tuple(axes)}, not a subset of array axes, {tuple(shape_a)}")

    nominal = a.nominal
    distribution = a.distribution

    shape_base = {ax: shape_a[ax] for ax in axes}
    nominal = na.broadcast_to(a.nominal, na.broadcast_shapes(na.shape(nominal), shape_base))
    distribution = na.broadcast_to(distribution, na.broadcast_shapes(na.shape(distribution), shape_base))

    kwargs = dict(
        axes=axes,
        s=s,
        norm=norm,
    )

    return na.UncertainScalarArray(
        nominal=func(a=nominal, **kwargs),
        distribution=func(a=distribution, **kwargs)
    )


def array_function_emath(
    func: Callable,
    *args: na.AbstractUncertainScalarArray,
    **kwargs: na.AbstractUncertainScalarArray,
) -> na.UncertainScalarArray:
    try:
        args = tuple(uncertainties._normalize(a) for a in args)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except uncertainties.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    args_nominal = tuple(a.nominal for a in args)
    args_distribution = tuple(a.distribution for a in args)

    kwargs_nominal = {k: kwargs[k].nominal for k in kwargs}
    kwargs_distribution = {k: kwargs[k].distribution for k in kwargs}

    return na.UncertainScalarArray(
        nominal=func(*args_nominal, **kwargs_nominal),
        distribution=func(*args_distribution, **kwargs_distribution)
    )


def implements(numpy_function: Callable):
    """Register an __array_function__ implementation for :class:`named_arrays.AbstractUncertainScalarArray` objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.copyto)
def copyto(
        dst: na.UncertainScalarArray,
        src: na.AbstractScalarArray | na.AbstractUncertainScalarArray,
        casting: str = "same_kind",
        where: bool | na.AbstractScalarArray | na.AbstractUncertainScalarArray = True,
):
    if not isinstance(dst, na.UncertainScalarArray):
        return NotImplemented

    if isinstance(src, na.AbstractArray):
        if isinstance(src, na.AbstractScalar):
            if isinstance(src, na.AbstractScalarArray):
                src_nominal = src_distribution = src
            elif isinstance(src, na.AbstractUncertainScalarArray):
                src_nominal = src.nominal
                src_distribution = src.distribution
            else:
                return NotImplemented
        else:
            return NotImplemented
    else:
        src_nominal = src_distribution = src

    if isinstance(where, na.AbstractArray):
        if isinstance(where, na.AbstractScalar):
            if isinstance(where, na.AbstractScalarArray):
                where_nominal = where_distribution = where
            elif isinstance(src, na.AbstractUncertainScalarArray):
                where_nominal = src.nominal
                where_distribution = src.distribution
            else:
                return NotImplemented
        else:
            return NotImplemented
    else:
        where_nominal = where_distribution = where

    try:
        np.copyto(dst=dst.nominal, src=src_nominal, casting=casting, where=where_nominal)
    except TypeError:
        dst.nominal = src_nominal

    try:
        np.copyto(dst=dst.distribution, src=src_distribution, casting=casting, where=where_distribution)
    except TypeError:
        dst.distribution = src_distribution


@implements(np.broadcast_to)
def broadcast_to(
        array: na.AbstractUncertainScalarArray,
        shape: dict[str, int]
) -> na.UncertainScalarArray:

    array = array.explicit
    shape_distribution = na.broadcast_shapes(shape, array.shape_distribution)

    return na.UncertainScalarArray(
        nominal=na.broadcast_to(array.nominal, shape=shape),
        distribution=na.broadcast_to(array.distribution, shape=shape_distribution)
    )


@implements(np.transpose)
def transpose(
        a: na.AbstractUncertainScalarArray,
        axes: None | Sequence[str] = None,
) -> na.UncertainScalarArray:

    a = a.broadcasted

    if axes is not None:
        axes_distribution = tuple(axes) + (a.axis_distribution, )
    else:
        axes_distribution = axes

    return na.UncertainScalarArray(
        nominal=np.transpose(a.nominal, axes=axes),
        distribution=np.transpose(a.distribution, axes=axes_distribution)
    )


@implements(np.moveaxis)
def moveaxis(
        a: na.AbstractUncertainScalarArray,
        source: str | Sequence[str],
        destination: str | Sequence[str],
) -> na.UncertainScalarArray:

    source = (source,) if isinstance(source, str) else source
    destination = (destination,) if isinstance(destination, str) else destination

    a = a.explicit

    set_axis_diff = set(source) - set(a.axes)
    if set_axis_diff:
        raise ValueError(f"source axes {tuple(set_axis_diff)} not in array axes {a.axes}")

    nominal = na.as_named_array(a.nominal)
    distribution = na.as_named_array(a.distribution)

    source_nominal, source_distribution = [], []
    destination_nominal, destination_distribution = [], []

    for i in range(len(source)):
        if source[i] in nominal.axes:
            source_nominal.append(source[i])
            destination_nominal.append(destination[i])
        if source[i] in distribution.axes:
            source_distribution.append(source[i])
            destination_distribution.append(destination[i])

    return na.UncertainScalarArray(
        nominal=np.moveaxis(
            a=nominal,
            source=source_nominal,
            destination=destination_nominal,
        ),
        distribution=np.moveaxis(
            a=distribution,
            source=source_distribution,
            destination=destination_distribution,
        ),
    )


@implements(np.reshape)
def reshape(
        a: na.AbstractUncertainScalarArray,
        newshape: dict[str, int],
) -> na.UncertainScalarArray:

    a = a.explicit
    a.nominal = na.broadcast_to(a.nominal, shape=a.shape)
    a.distribution = na.broadcast_to(a.distribution, shape=a.shape_distribution)
    a.distribution.change_axis_index(axis=a.axis_distribution, index=~0)

    newshape_distribution = newshape | {a.axis_distribution: a.num_distribution}

    return na.UncertainScalarArray(
        nominal=np.reshape(a.nominal, newshape=newshape),
        distribution=np.reshape(a.distribution, newshape=newshape_distribution)
    )


@implements(np.linalg.inv)
def linalg_inv(a: na.AbstractUncertainScalarArray,):
    raise NotImplementedError(
        "np.linalg.inv not supported, use 'named_arrays.AbstractScalarArray.matrix_inverse()' instead"
    )


def array_function_stack_like(
        func: Callable,
        arrays: Sequence[bool | float | complex | str | na.AbstractScalarArray | na.AbstractUncertainScalarArray],
        axis: str,
        out: None | na.AbstractUncertainScalarArray = None,
        *,
        dtype: None | type | np.dtype = None,
        casting: str = "same_kind",
) -> na.UncertainScalarArray:

    arrays_nominal = []
    arrays_distribution = []

    for array in arrays:
        array = array.broadcasted
        if isinstance(array, na.AbstractArray):
            if isinstance(array, na.AbstractScalar):
                if isinstance(array, na.AbstractUncertainScalarArray):
                    array_nominal = na.as_named_array(array.nominal)
                    array_distribution = na.as_named_array(array.distribution)
                else:
                    array_nominal = array_distribution = array
            else:
                return NotImplemented
        else:
            array_nominal = array_distribution = array
        arrays_nominal.append(array_nominal)
        arrays_distribution.append(array_distribution)

    result_nominal = func(
        arrays_nominal,
        axis=axis,
        out=out.nominal if out is not None else out,
        dtype=dtype,
        casting=casting,
    )
    result_distribution = func(
        arrays_distribution,
        axis=axis,
        out=out.distribution if out is not None else out,
        dtype=dtype,
        casting=casting,
    )

    if out is None:
        return na.UncertainScalarArray(
            nominal=result_nominal,
            distribution=result_distribution,
        )
    else:
        return out


@implements(np.sort)
def sort(
        a: na.AbstractUncertainScalarArray,
        axis: None | str | Sequence[str],
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> na.UncertainScalarArray:

    a = a.broadcasted

    indices_sorted = np.argsort(
        a=a,
        axis=axis,
        kind=kind,
        order=order,
    )

    result = na.UncertainScalarArray(
        nominal=a.nominal[indices_sorted],
        distribution=a.distribution[indices_sorted],
    )

    return result


@implements(np.argsort)
def argsort(
        a: na.AbstractUncertainScalarArray,
        axis: None | str | Sequence[str],
        kind: None | str = None,
        order: None | str | list[str] = None,
) -> dict[str, na.UncertainScalarArray]:

    a = a.broadcasted

    return np.argsort(
        a=np.mean(a.distribution, axis=a.axis_distribution),
        axis=axis,
        kind=kind,
        order=order,
    )


@implements(np.unravel_index)
def unravel_index(
        indices: na.AbstractUncertainScalarArray,
        shape: dict[str, int],
) -> dict[str, na.AbstractUncertainScalarArray]:

    if not shape:
        return dict()

    indices = indices.explicit

    result_nominal = np.unravel_index(na.as_named_array(indices.nominal), shape=shape)
    result_distribution = np.unravel_index(na.as_named_array(indices.distribution), shape=shape)

    result = {ax: na.UncertainScalarArray(result_nominal[ax], result_distribution[ax]) for ax in shape}

    return result


@implements(np.array_equal)
def array_equal(
        a1: na.AbstractUncertainScalarArray,
        a2: na.AbstractUncertainScalarArray,
        equal_nan: bool = False,
) -> bool:
    result_nominal = np.array_equal(
        a1=a1.nominal,
        a2=a2.nominal,
        equal_nan=equal_nan,
    )
    result_distribution = np.array_equal(
        a1=a1.distribution,
        a2=a2.distribution,
        equal_nan=equal_nan,
    )
    return result_nominal and result_distribution


@implements(np.array_equiv)
def array_equiv(
        a1: na.AbstractUncertainScalarArray,
        a2: na.AbstractUncertainScalarArray,
) -> bool:
    result_nominal = np.array_equiv(
        a1=a1.nominal,
        a2=a2.nominal,
    )
    result_distribution = np.array_equiv(
        a1=a1.distribution,
        a2=a2.distribution,
    )
    return result_nominal and result_distribution


@implements(np.allclose)
def allclose(
        a: na.ScalarLike,
        b: na.ScalarLike,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
):
    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalar):
            if isinstance(a, na.AbstractUncertainScalarArray):
                a_nominal = a.nominal
                a_distribution = a.distribution
            else:
                a_nominal = a_distribution = a
        else:
            return NotImplemented
    else:
        a_nominal = a_distribution = a

    if isinstance(b, na.AbstractArray):
        if isinstance(b, na.AbstractScalar):
            if isinstance(b, na.AbstractUncertainScalarArray):
                b_nominal = b.nominal
                b_distribution = b.distribution
            else:
                b_nominal = b_distribution = b
        else:
            return NotImplemented
    else:
        b_nominal = b_distribution = b

    result_nominal = np.allclose(
        a=a_nominal,
        b=b_nominal,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan
    )

    result_distribution = np.allclose(
        a=a_distribution,
        b=b_distribution,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan
    )

    return result_nominal and result_distribution


@implements(np.nonzero)
def nonzero(a: na.AbstractUncertainScalarArray) -> dict[str, na.UncertainScalarArray]:
    a = a.explicit

    result = np.nonzero(a.nominal * np.prod(a.distribution, axis=a.axis_distribution))

    return result


@implements(np.where)
def where(
    condition: na.AbstractScalar,
    x: float | u.Quantity | na.AbstractScalar,
    y: float | u.Quantity | na.AbstractScalar,
) -> na.UncertainScalarArray:
    try:
        condition = uncertainties._normalize(condition)
        x = uncertainties._normalize(x)
        y = uncertainties._normalize(y)
    except uncertainties.UncertainScalarTypeError:
        return NotImplemented

    return condition.type_explicit(
        nominal=np.where(
            condition.nominal,
            x.nominal,
            y.nominal,
        ),
        distribution=np.where(
            condition.distribution,
            x.distribution,
            y.distribution,
        ),
    )


@implements(np.nan_to_num)
def nan_to_num(
        x: na.AbstractUncertainScalarArray,
        copy: bool = True,
        nan: float = 0.0,
        posinf: None | float = None,
        neginf: None | float = None,
):
    if not copy:
        if not isinstance(x, na.UncertainScalarArray):
            raise TypeError("can't write to an array that is not an instance of `named_arrays.AbstractExplicitArray`")

    result_nominal = np.nan_to_num(x.nominal, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
    result_distribution = np.nan_to_num(x.distribution, copy=copy, nan=nan, posinf=posinf, neginf=neginf)

    if copy:
        return na.UncertainScalarArray(result_nominal, result_distribution)
    else:
        return x


@implements(np.convolve)
def convolve(
        a: na.AbstractUncertainScalarArray,
        v: na.ScalarLike,
        mode: str = 'full',
) -> na.UncertainScalarArray:
    raise ValueError("`numpy.convolve` is not supported for instances of `named_arrays.AbstractUncertainScalarArray`")


@implements(np.repeat)
def repeat(
    a: na.AbstractUncertainScalarArray,
    repeats: int | na.AbstractScalarArray,
    axis: str,
) -> na.UncertainScalarArray:

    a = a.broadcasted

    return a.type_explicit(
        nominal=np.repeat(
            a=a.nominal,
            repeats=repeats,
            axis=axis,
        ),
        distribution=np.repeat(
            a=a.distribution,
            repeats=repeats,
            axis=axis,
        )
    )


@implements(np.diff)
def diff(
    a: na.AbstractUncertainScalarArray,
    axis: str,
    n: int = 1,
    prepend: None | float | na.AbstractScalar = None,
    append: None | float | na.AbstractScalar = None,
) -> na.UncertainScalarArray:

    try:
        a = uncertainties._normalize(a)
        prepend = uncertainties._normalize(prepend)
        append = uncertainties._normalize(append)
    except uncertainties.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    a = a.broadcasted

    return a.type_explicit(
        nominal=np.diff(
            a=na.as_named_array(a.nominal),
            axis=axis,
            n=n,
            prepend=prepend.nominal,
            append=append.nominal,
        ),
        distribution=np.diff(
            a=na.as_named_array(a.distribution),
            axis=axis,
            n=n,
            prepend=prepend.distribution,
            append=append.distribution,
        ),
    )
