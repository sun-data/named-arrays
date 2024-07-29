from typing import Callable, Sequence
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.uncertainties.uncertainties_array_functions

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
    "broadcast_to",
    "tranpose",
    "moveaxis",
    "reshape",
    "array_equal",
]

DEFAULT_FUNCTIONS = named_arrays._scalars.uncertainties.uncertainties_array_functions.DEFAULT_FUNCTIONS
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

    a = a.explicit
    inputs = a.inputs
    outputs = a.outputs

    if isinstance(where, na.AbstractArray):
        if isinstance(where, na.AbstractFunctionArray):
            if np.any(where.inputs != inputs):
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
    shape_inputs = na.shape_broadcasted(inputs, inputs_where)
    shape_outputs = na.shape_broadcasted(outputs, outputs_where)

    axis_normalized = tuple(shape) if axis is None else (axis,) if isinstance(axis, str) else axis

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument must be `None` or a subset of the broadcasted shape of `a` and `where`, "
                f"got {axis} for `axis`, but `{shape} for `shape`"
            )

    shape_base = {ax: shape[ax] for ax in axis_normalized}

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
        if inputs_out is not None:
            np.copyto(src=inputs, dst=inputs_out)
            inputs_result = inputs_out
        else:
            inputs_result = inputs
    else:
        inputs_result = np.mean(
            a=na.broadcast_to(inputs, na.broadcast_shapes(shape_inputs, shape_base)),
            axis=axis_normalized,
            out=inputs_out,
            keepdims=keepdims,
            where=inputs_where,
        )

    outputs_result = func(
        a=na.broadcast_to(outputs, na.broadcast_shapes(shape_outputs, shape_base)),
        axis=axis_normalized,
        out=outputs_out,
        **kwargs,
    )

    if out is None:
        result = a.type_explicit(
            inputs=inputs_result,
            outputs=outputs_result,
        )
    else:
        result = out

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
) -> na.FunctionArray:

    a = a.explicit
    inputs = a.inputs
    outputs = a.outputs

    shape = a.shape
    shape_inputs = a.inputs.shape
    shape_outputs = a.outputs.shape

    axis_normalized = na.axis_normalized(a, axis)

    if axis is not None:
        if not set(axis_normalized).issubset(shape):
            raise ValueError(
                f"the `axis` argument, {axis}, must be `None` or a subset of the shape of `a`, {shape}"
            )

    shape_base = {ax: shape[ax] for ax in axis_normalized}

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
        if inputs_out is not None:
            np.copyto(src=inputs, dst=inputs_out)
            inputs_result = inputs_out
        else:
            inputs_result = inputs
    else:
        inputs_result = np.mean(
            a=na.broadcast_to(inputs, na.broadcast_shapes(shape_inputs, shape_base)),
            axis=axis_normalized,
            out=inputs_out,
            keepdims=keepdims,
        )

    outputs_result = func(
        a=na.broadcast_to(outputs, na.broadcast_shapes(shape_outputs, shape_base)),
        q=q,
        axis=axis_normalized,
        out=outputs_out,
        **kwargs,
    )

    if out is None:
        result = a.type_explicit(
            inputs=inputs_result,
            outputs=outputs_result,
        )
    else:
        result = out

    return result


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
            raise ValueError(f"axis '{axis}' must be present in all the input arrays, got {[a.axes for a in arrays]}")

        arrays_broadcasted = list()
        for array in arrays:
            shape = array.shape
            shape_base = {axis: shape[axis]}
            array = na.FunctionArray(
                inputs=na.broadcast_to(array.inputs, na.broadcast_shapes(array.inputs.shape, shape_base)),
                outputs=na.broadcast_to(array.outputs, na.broadcast_shapes(array.outputs.shape, shape_base)),
            )
            arrays_broadcasted.append(array)
        arrays = arrays_broadcasted

    arrays_inputs = tuple(array.inputs for array in arrays)
    arrays_outputs = tuple(array.outputs for array in arrays)

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
        result = na.FunctionArray(inputs=inputs_result, outputs=outputs_result)
    else:
        out.inputs = inputs_result
        out.outputs = outputs_result
        result = out

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
            if np.any(where.inputs != src.inputs):
                raise ValueError(f"`where.inputs` must be equivalent to `src.inputs`")
            where_inputs = where.inputs
            where_outputs = where.outputs
        else:
            return NotImplemented
    else:
        where_inputs = where_outputs = where

    try:
        np.copyto(dst=dst.inputs, src=src.inputs, casting=casting, where=where_inputs)
    except TypeError:
        dst.inputs = src.inputs

    try:
        np.copyto(dst=dst.outputs, src=src.outputs, casting=casting, where=where_outputs)
    except TypeError:
        dst.outputs = src.outputs


@_implements(np.broadcast_to)
def broadcast_to(
        array: na.AbstractFunctionArray,
        shape: dict[str, int]
) -> na.FunctionArray:
    return array.type_explicit(
        inputs=na.broadcast_to(array.inputs, shape=shape),
        outputs=na.broadcast_to(array.outputs, shape=shape),
    )


@_implements(np.transpose)
def tranpose(
        a: na.AbstractFunctionArray,
        axes: None | Sequence[str] = None
) -> na.FunctionArray:

    a = a.explicit
    shape = a.shape

    axes_normalized = tuple(reversed(shape) if axes is None else axes)
    #
    # if not set(axes_normalized).issubset(shape):
    #     raise ValueError(f"`axes` {axes} not a subset of `a.axes`, {a.axes}")

    shape_inputs = a.inputs.shape
    shape_outputs = a.outputs.shape

    return a.type_explicit(
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

    return a.type_explicit(
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
        newshape: dict[str, int],
) -> na.FunctionArray:

    a = np.broadcast_to(a, a.shape)

    return a.type_explicit(
        inputs=np.reshape(a.inputs, newshape=newshape),
        outputs=np.reshape(a.outputs, newshape=newshape)
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


@_implements(np.repeat)
def repeat(
    a: na.AbstractFunctionArray,
    repeats: int | na.AbstractScalarArray,
    axis: str,
) -> na.FunctionArray:

    a = a.broadcasted

    return a.type_explicit(
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
