from typing import Callable, Sequence
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_array_functions
import named_arrays._scalars.uncertainties.uncertainties_array_functions
from named_arrays._functions import _fields

__all__ = [
    "DEFAULT_FUNCTIONS",
    "PERCENTILE_LIKE_FUNCTIONS",
    "ARG_REDUCE_FUNCTIONS",
    "STACK_LIKE_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "array_function_default",
    "array_function_percentile_like",
    "array_function_arg_reduce",
    "array_function_stack_like",
    "tranpose",
    "moveaxis",
    "reshape",
    "array_equal",
]

DEFAULT_FUNCTIONS = named_arrays._scalars.uncertainties.uncertainties_array_functions.DEFAULT_FUNCTIONS
CUMULATIVE_REDUCE_FUNCTIONS = named_arrays._scalars.uncertainties.uncertainties_array_functions.CUMULATIVE_REDUCE_FUNCTIONS
PERCENTILE_LIKE_FUNCTIONS = named_arrays._scalars.uncertainties.uncertainties_array_functions.PERCENTILE_LIKE_FUNCTIONS
ARG_REDUCE_FUNCTIONS = named_arrays._scalars.uncertainties.uncertainties_array_functions.ARG_REDUCE_FUNCTIONS
STACK_LIKE_FUNCTIONS = named_arrays._scalars.uncertainties.uncertainties_array_functions.STACK_LIKE_FUNCTIONS

HANDLED_FUNCTIONS = dict()


def array_function_default(
        func: Callable,
        a: na.AbstractFunctionArray,
        axis: None | str | Sequence[str] = None,
        dtype: None | type | np.dtype = np._NoValue,
        out: None | na.AbstractFunctionArray = None,
        keepdims: bool = True,
        initial: None | bool | int | float | complex | u.Quantity = np._NoValue,
        where: na.AbstractFunctionArray = np._NoValue,
) -> na.FunctionArray:

    func, a = named_arrays._scalars.scalar_array_functions.count_nonzero_as_sum(func, a)

    a = a.explicit
    inputs = a.inputs
    outputs = a.outputs

    if isinstance(where, na.AbstractArray):
        if isinstance(where, na.AbstractFunctionArray):
            if not np.all(where.inputs == inputs):
                raise na.InputValueError("`where.inputs` must match `a.inputs`")
            inputs_where = outputs_where = where.outputs
        elif isinstance(where, (na.AbstractScalar, na.AbstractVectorArray)):
            if where.shape:
                raise ValueError(
                    f"if `where` is an instance of {na.AbstractArray}, but not {na.AbstractFunctionArray}, "
                    f"it must have an empty shape, got {where.shape}"
                )
            inputs_where = outputs_where = where
        else:
            return NotImplemented
    else:
        inputs_where = outputs_where = where

    shape = na.shape_broadcasted(a, where)
    shape_outputs = na.shape_broadcasted(outputs, outputs_where)

    axis_normalized = tuple(shape) if axis is None else (axis,) if isinstance(axis, str) else axis

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of the broadcasted shape of `a` and `where`, "
                f"got {axis} for `axis`, but `{shape} for `shape`"
            )

    kwargs = dict(
        keepdims=keepdims,
    )

    if dtype is not np._NoValue:
        kwargs["dtype"] = dtype
    if initial is not np._NoValue:
        kwargs["initial"] = initial
    if where is not np._NoValue:
        kwargs["where"] = outputs_where

    if isinstance(out, na.AbstractFunctionArray):
        inputs_out = out.inputs
        outputs_out = out.outputs
    else:
        inputs_out = outputs_out = out

    if keepdims:
        _check_keepdims(func, a, axis_normalized)
        if inputs_out is not None:
            np.copyto(src=inputs, dst=inputs_out)
            inputs_result = inputs_out
        else:
            inputs_result = inputs
    else:
        _fields.check_axes(
            a=a,
            axis=axis_normalized,
            operation=f"np.{func.__name__} with keepdims=False removes",
            hint=" Or keep the axes with keepdims=True.",
        )
        inputs = inputs.cell_centers(axis=set(axis_normalized)-set(a.axes_center))
        shape_inputs = na.shape_broadcasted(inputs, inputs_where)
        inputs_result = np.mean(
            a=na.broadcast_to(inputs, shape_inputs),
            axis=[ax for ax in shape_inputs if ax in axis_normalized],
            out=inputs_out,
            keepdims=keepdims,
            where=inputs_where,
        )

    outputs_result = func(
        a=na.broadcast_to(outputs, shape_outputs),
        axis=[ax for ax in shape_outputs if ax in axis_normalized],
        out=outputs_out,
        **kwargs,
    )

    if out is None:
        result = a.replace(
            inputs=inputs_result,
            outputs=outputs_result,
        )
    else:
        result = _fields.out(out, _fields.values(a))

    return result


def array_function_cumulative_reduce(
    func: Callable,
    a: na.AbstractFunctionArray,
    axis: None | str | Sequence[str] = None,
    dtype: None | type | np.dtype = np._NoValue,
    out: None | na.AbstractFunctionArray = None,
    **kwargs,
) -> na.FunctionArray:

    a = a.explicit
    inputs = a.inputs
    outputs = a.outputs

    shape = a.shape

    if axis is None:
        _axis = tuple(shape)
    elif isinstance(axis, str):
        _axis = (axis, )
    else:
        _axis = axis

    if len(_axis) != 1:
        raise ValueError(f"only one axis is supported, got {_axis}.")

    _axis = _axis[0]

    if dtype is not np._NoValue:
        kwargs["dtype"] = dtype

    if isinstance(out, na.AbstractFunctionArray):
        inputs_out = out.inputs
        outputs_out = out.outputs
    else:
        inputs_out = outputs_out = out

    if inputs_out is not None:
        np.copyto(src=inputs, dst=inputs_out)
        inputs_result = inputs_out
    else:
        inputs_result = inputs

    shape_base = {_axis: shape[_axis]}

    outputs_result = func(
        na.broadcast_to(outputs, shape_base, append=True),
        axis=_axis,
        out=outputs_out,
        **kwargs,
    )

    if out is None:
        result = a.replace(
            inputs=inputs_result,
            outputs=outputs_result,
        )
    else:
        result = _fields.out(out, _fields.values(a))

    return result


def array_function_percentile_like(
        func: Callable,
        a: na.AbstractFunctionArray,
        q: float | u.Quantity | na.AbstractArray,
        axis: None | str | Sequence[str] = None,
        out: None | na.FunctionArray = None,
        overwrite_input: bool = False,
        method: str = "linear",
        keepdims: bool = False,
        *,
        weights: float | u.Quantity | na.AbstractArray = np._NoValue,
) -> na.FunctionArray:

    a = a.explicit
    inputs = a.inputs
    outputs = a.outputs

    # the weights apply to the outputs, and may have axes which they do not
    shape_outputs = na.shape_broadcasted(outputs, weights)
    shape = na.broadcast_shapes(a.shape, shape_outputs)
    shape_inputs = a.inputs.shape

    axis_normalized = na.axis_normalized(a, axis)

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument, {axis}, must be `None` or a subset of the shape of `a`, {shape}"
            )

    kwargs = dict(
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )

    if isinstance(out, na.AbstractFunctionArray):
        inputs_out = out.inputs
        outputs_out = out.outputs
    else:
        inputs_out = outputs_out = out

    if keepdims:
        _check_keepdims(func, a, axis_normalized)
        if inputs_out is not None:
            np.copyto(src=inputs, dst=inputs_out)
            inputs_result = inputs_out
        else:
            inputs_result = inputs
    else:
        _fields.check_axes(
            a=a,
            axis=axis_normalized,
            operation=f"np.{func.__name__} with keepdims=False removes",
            hint=" Or keep the axes with keepdims=True.",
        )
        inputs_result = np.mean(
            a=na.broadcast_to(inputs, shape_inputs),
            axis=[ax for ax in shape_inputs if ax in axis_normalized],
            out=inputs_out,
            keepdims=keepdims,
        )

    if weights is not np._NoValue:
        kwargs["weights"] = weights

    outputs_result = func(
        a=na.broadcast_to(outputs, shape_outputs),
        q=q,
        axis=[ax for ax in shape_outputs if ax in axis_normalized],
        out=outputs_out,
        **kwargs,
    )

    if out is None:
        result = a.replace(
            inputs=inputs_result,
            outputs=outputs_result,
        )
    else:
        result = _fields.out(out, _fields.values(a))

    return result


def _check_keepdims(
        func: Callable,
        a: na.AbstractFunctionArray,
        axis: tuple[str, ...],
) -> None:
    """
    Raise an error if a reduction which keeps its axes would reduce a field of
    `a` to a single element anyway.

    The result keeps the elements of the inputs along the reduced axes, and
    so do the fields, but the outputs and the result have a single element
    along an axis which the inputs do not have.
    """
    if not _fields.names(a):
        return
    shape_inputs = na.shape(a.inputs)
    _fields.check_axes(
        a=a,
        axis=tuple(ax for ax in axis if ax not in shape_inputs),
        operation=(
            f"np.{func.__name__} reduces to a single element, since "
            f"keepdims=True keeps only the axes of the inputs"
        ),
    )


def array_function_arg_reduce(
        func: Callable,
        a: na.AbstractFunctionArray,
        axis: None | str | Sequence[str] = None,
) -> dict[str, na.AbstractArray]:

    return func(a=a.outputs, axis=axis)


def array_function_stack_like(
        func: Callable,
        arrays: Sequence[na.AbstractFunctionArray],
        axis: str,
        out: None | na.FunctionArray = None,
        *,
        dtype: str | np.dtype | type = None,
        casting: str = "same_kind",
):

    if any(not isinstance(array, na.AbstractFunctionArray) for array in arrays):
        return NotImplemented

    if func is np.concatenate:

        if any(axis not in array.shape for array in arrays):
            raise ValueError(
                f"axis '{axis}' must be present in all the input arrays, "
                f"got {[a.axes for a in arrays]}"
            )

        if any(axis in a.axes_vertex for a in arrays):
            raise ValueError(
                f"concatenating along vertex a vertex axis '{axis}' is not supported."
            )

        arrays_broadcasted = list()
        lengths = list()
        for array in arrays:

            array = array.explicit
            shape = array.shape

            array = array.broadcast_to({axis: shape[axis]}, append=True)
            arrays_broadcasted.append(array)
            lengths.append(shape[axis])

        arrays = arrays_broadcasted

    else:
        lengths = None

    arrays_inputs = tuple(array.inputs for array in arrays)
    arrays_outputs = tuple(array.outputs for array in arrays)

    # the result takes the type of an array whose type is a subclass of the
    # types of all the others, so that it does not depend on their order
    template = arrays[0]
    for array in arrays:
        if all(isinstance(array, type(other)) for other in arrays):
            template = array
            break

    fields = _fields.stack_like(
        func=func,
        arrays=arrays,
        template=template,
        axis=axis,
        lengths=lengths,
    )

    if out is None:
        inputs_out = outputs_out = out
    else:
        inputs_out = out.inputs
        outputs_out = out.outputs

    inputs_result = func(
        arrays=arrays_inputs,
        axis=axis,
        out=inputs_out,
    )

    outputs_result = func(
        arrays=arrays_outputs,
        axis=axis,
        out=outputs_out,
        dtype=dtype,
        casting=casting,
    )

    if out is None:
        result = template.replace(
            inputs=inputs_result,
            outputs=outputs_result,
            **fields,
        )
    else:
        out.inputs = inputs_result
        out.outputs = outputs_result
        result = _fields.out(out, fields)

    return result


def _implements(numpy_function: Callable):
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@_implements(np.copyto)
def copyto(
        dst: na.FunctionArray,
        src: na.AbstractFunctionArray,
        casting: str = "same_kind",
        where: bool | na.AbstractFunctionArray = True,
) -> None:
    if not isinstance(dst, na.FunctionArray):
        return NotImplemented

    if not isinstance(src, na.AbstractFunctionArray):
        return NotImplemented

    if isinstance(where, na.AbstractArray):
        if isinstance(where, na.AbstractFunctionArray):
            if np.any(where.inputs != src.inputs):  #pragma: nocover
                raise ValueError("`where.inputs` must be equivalent to `src.inputs`")
            where_inputs = where.inputs
            where_outputs = where.outputs
        else:
            return NotImplemented
    else:
        where_inputs = where_outputs = where

    # the fields are written the way an assignment writes them, as
    # ``dst[where] = src[where]``
    if not _fields.values(dst):
        writes = []
    elif isinstance(where, na.AbstractFunctionArray):
        writes = _fields.setitem(dst, where.outputs, src[where])
    else:
        writes = _fields.setitem(dst, dict(), src)

    try:
        np.copyto(dst=dst.inputs, src=src.inputs, casting=casting, where=where_inputs)
    except TypeError:
        dst.inputs = src.inputs

    try:
        np.copyto(dst=dst.outputs, src=src.outputs, casting=casting, where=where_outputs)
    except TypeError:
        dst.outputs = src.outputs

    for write in writes:
        write()


@_implements(np.gradient)
def gradient(
    f: na.AbstractFunctionArray,
    *varargs: float | u.Quantity | na.AbstractArray,
    axis: None | str | Sequence[str] = None,
    edge_order: int = 1,
) -> na.FunctionArray | tuple[na.FunctionArray, ...]:
    """
    Differentiate the outputs of a function array against its inputs.

    Forwards to :meth:`named_arrays.AbstractFunctionArray.gradient`, which
    takes the component of the inputs to differentiate against, where this
    function always uses the inputs themselves.
    """
    if varargs:
        raise ValueError(
            f"the spacing of a function array is given by its inputs, so a "
            f"spacing argument does not apply, got {len(varargs)} of them"
        )

    if axis is None:
        raise ValueError(
            "`axis` is required, since there is no positional order to "
            "differentiate along. Name the axes to differentiate along."
        )

    axes = (axis,) if isinstance(axis, str) else tuple(axis)

    result = tuple(
        f.gradient(axis=ax, edge_order=edge_order)
        for ax in axes
    )

    if isinstance(axis, str):
        return result[0]

    return result


@_implements(np.transpose)
def tranpose(
        a: na.AbstractFunctionArray,
        axes: None | Sequence[str] = None
) -> na.FunctionArray:
    a = a.broadcasted
    shape = a.shape
    axes_normalized = tuple(reversed(shape) if axes is None else axes)

    return a.replace(
        inputs=np.transpose(
            a=a.inputs,
            axes=axes_normalized,
        ),
        outputs=np.transpose(
            a=a.outputs,
            axes=axes_normalized,
        ),
    )


@_implements(np.moveaxis)
def moveaxis(
        a: na.AbstractFunctionArray,
        source: str | Sequence[str],
        destination: str | Sequence[str],
):
    a = a.explicit
    shape = a.shape

    if isinstance(source, str):
        source = source,

    if isinstance(destination, str):
        destination = destination,

    if not set(source).issubset(shape):
        raise ValueError(f"source axes {source} not in array axes {a.axes}")

    shape_inputs = a.inputs.shape
    shape_outputs = a.outputs.shape

    source_destination_inputs = tuple((src, dest) for src, dest in zip(source, destination) if src in shape_inputs)
    source_destination_outputs = tuple((src, dest) for src, dest in zip(source, destination) if src in shape_outputs)

    source_inputs, destination_inputs = tuple(tuple(i) for i in zip(*source_destination_inputs))
    source_outputs, destination_outputs = tuple(tuple(i) for i in zip(*source_destination_outputs))

    def move(v: na.AbstractArray) -> na.AbstractArray:
        pairs = [(src, dest) for src, dest in zip(source, destination) if src in v.shape]
        return np.moveaxis(
            a=v,
            source=tuple(src for src, _ in pairs),
            destination=tuple(dest for _, dest in pairs),
        )

    return a.replace(
        **_fields.apply(a, move, axes=source),
        inputs=np.moveaxis(
            a=a.inputs,
            source=source_inputs,
            destination=destination_inputs,
        ),
        outputs=np.moveaxis(
            a=a.outputs,
            source=source_outputs,
            destination=destination_outputs,
        ),
    )


@_implements(np.reshape)
def reshape(
        a: na.AbstractFunctionArray,
        shape: dict[str, int],
) -> na.FunctionArray:

    if a.axes_vertex:
        raise ValueError(
            f"Cannot reshape an {type(a)} containing axes on cell vertices, "
            f"got {a.axes_vertex=}."
        )

    a = a.broadcasted
    shape_old = a.shape

    fields = dict()
    for name in _fields.names(a):
        v = getattr(a, name)
        if set(v.shape).isdisjoint(shape_old):
            continue
        if not set(v.shape).issubset(shape_old):
            raise ValueError(
                f"`{name}` of this {type(a).__name__} has axes "
                f"{tuple(v.shape)}, which cannot be reshaped along with the "
                f"axes of the array, {tuple(shape_old)}, since the reshape "
                f"does not account for {set(v.shape) - set(shape_old)}"
            )
        # a reshape flattens the elements in the order of the axes, so the
        # field is put in the same shape and order as the outputs first
        fields[name] = np.reshape(na.broadcast_to(v, shape_old), shape)

    return a.replace(
        inputs=np.reshape(a.inputs, shape),
        outputs=np.reshape(a.outputs, shape),
        **fields,
    )


@_implements(np.take_along_axis)
def take_along_axis(
        arr: na.AbstractFunctionArray,
        indices: na.AbstractArray,
        axis: str,
) -> na.FunctionArray:

    arr = arr.explicit
    shape = arr.shape

    if axis not in shape:
        raise ValueError(
            f"`axis`, {axis!r}, must be one of the axes in `arr`, {tuple(shape)}"
        )

    if axis in arr.axes_vertex:
        raise ValueError(
            f"`axis`, {axis!r}, describes input vertices and cannot be used in `take_along_axis`, "
            f"got vertex axes {arr.axes_vertex}."
        )

    # Broadcast only `axis` so that `inputs` and `outputs` are reordered
    # consistently even if one of them does not vary along `axis`.
    inputs = na.broadcast_to(arr.inputs, shape={axis: shape[axis]}, append=True)
    outputs = na.broadcast_to(arr.outputs, shape={axis: shape[axis]}, append=True)

    return arr.replace(
        inputs=np.take_along_axis(inputs, indices, axis=axis),
        outputs=np.take_along_axis(outputs, indices, axis=axis),
        **_fields.apply(
            arr,
            lambda v: np.take_along_axis(v, indices, axis=axis),
            axes=(axis,),
        ),
    )


@_implements(np.array_equal)
def array_equal(
        a1: na.AbstractFunctionArray,
        a2: na.AbstractFunctionArray,
        equal_nan: bool = False,
):
    inputs_equal = np.array_equal(
        a1=a1.inputs,
        a2=a2.inputs,
        equal_nan=equal_nan,
    )
    outputs_equal = np.array_equal(
        a1=a1.outputs,
        a2=a2.outputs,
        equal_nan=equal_nan
    )

    return inputs_equal and outputs_equal


@_implements(np.array_equiv)
def array_equiv(
        a1: na.AbstractFunctionArray,
        a2: na.AbstractFunctionArray,
):
    inputs_equiv = np.array_equiv(
        a1=a1.inputs,
        a2=a2.inputs,
    )
    outputs_equiv = np.array_equiv(
        a1=a1.outputs,
        a2=a2.outputs,
    )

    return inputs_equiv and outputs_equiv


@_implements(np.allclose)
def allclose(
        a: na.AbstractFunctionArray,
        b: na.AbstractFunctionArray,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
):
    close_inputs = np.allclose(
        a=a.inputs,
        b=b.inputs,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan
    )
    close_outputs = np.allclose(
        a=a.outputs,
        b=b.outputs,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    return close_inputs and close_outputs


@_implements(np.nonzero)
def nonzero(a: na.AbstractFunctionArray) -> dict[str, na.AbstractArray]:
    return np.nonzero(a.outputs)


@_implements(np.clip)
def clip(
    a: na.AbstractFunctionArray,
    a_min: None | float | na.AbstractScalarArray | na.AbstractVectorArray = np._NoValue,
    a_max: None | float | na.AbstractScalarArray | na.AbstractVectorArray = np._NoValue,
    out: None | na.FunctionArray = None,
) -> na.FunctionArray:

    a = a.explicit

    a_outputs = a.outputs

    if out is not None:
        _out = out.outputs
    else:
        _out = None

    result = np.clip(
        a=a_outputs,
        a_min=a_min,
        a_max=a_max,
        out=_out,
    )

    if out is None:
        result = a.replace(outputs=result)
    else:
        result = _fields.out(out, _fields.values(a))

    return result


@_implements(np.round)
@_implements(np.around)
def round(
    a: na.AbstractFunctionArray,
    decimals: int = 0,
    out: None | na.FunctionArray = None,
) -> na.FunctionArray:

    a = a.explicit

    if out is not None:
        _out = out.outputs
    else:
        _out = None

    result = np.round(
        a=a.outputs,
        decimals=decimals,
        out=_out,
    )

    if out is None:
        result = a.replace(outputs=result)
    else:
        result = _fields.out(out, _fields.values(a))

    return result


@_implements(np.isclose)
def isclose(
    a: na.ArrayLike,
    b: na.ArrayLike,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> na.FunctionArray:

    operands = (a, b)

    functions = [x for x in operands if isinstance(x, na.AbstractFunctionArray)]
    outputs = [x.outputs if isinstance(x, na.AbstractFunctionArray) else x for x in operands]

    inputs = functions[0].inputs
    for function in functions[1:]:
        if np.any(function.inputs != inputs):
            raise na.InputValueError("`a.inputs` must match `b.inputs`")

    return functions[0].explicit.replace(
        inputs=inputs,
        outputs=np.isclose(
            *outputs,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ),
    )


@_implements(np.repeat)
def repeat(
    a: na.AbstractFunctionArray,
    repeats: int | na.AbstractScalarArray,
    axis: str,
) -> na.FunctionArray:
    if axis in a.axes_vertex:
        raise ValueError(f"Array cannot be repeated along vertex axis {axis}.")

    a = a.broadcasted

    return a.replace(
        **_fields.apply(
            a,
            lambda v: np.repeat(a=v, repeats=repeats, axis=axis),
            axes=(axis,),
        ),
        inputs=np.repeat(
            a=a.inputs,
            repeats=repeats,
            axis=axis,
        ),
        outputs=np.repeat(
            a=a.outputs,
            repeats=repeats,
            axis=axis,
        )
    )
