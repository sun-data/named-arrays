from typing import Callable
import numpy as np
import matplotlib
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
]

ASARRAY_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.ASARRAY_LIKE_FUNCTIONS
HANDLED_FUNCTIONS = dict()

def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


def asarray_like(
        func: Callable,
        a: None | float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray | na.AbstractFunctionArray,
        dtype: None | type | np.dtype = None,
        order: None | str = None,
        *,
        like: None | float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray | na.AbstractFunctionArray = None,
) -> None | na.AbstractFunctionArray:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractFunctionArray):
            a_inputs = a.inputs
            a_outputs = a.outputs
        elif isinstance(a, na.AbstractVectorArray):
            a_inputs = a_outputs = a
        elif isinstance(a, na.AbstractScalar):
            a_inputs = a_outputs = a
        else:
            return NotImplemented
    else:
        a_inputs = a_outputs = a

    if isinstance(like, na.AbstractArray):
        if isinstance(like, na.AbstractFunctionArray):
            like_inputs = like.inputs
            like_outputs = like.outputs
            type_like = like.type_explicit
        elif isinstance(like, na.AbstractVectorArray):
            like_inputs = like_outputs = like
            type_like = na.FunctionArray
        elif isinstance(like, na.AbstractScalar):
            like_inputs = like_outputs = like
            type_like = na.FunctionArray
        else:
            return NotImplemented
    else:
        like_inputs = like_outputs = like
        type_like = na.FunctionArray

    return type_like(
        inputs=func(
            a=a_inputs,
            dtype=dtype,
            order=order,
            like=like_inputs,
        ),
        outputs=func(
            a=a_outputs,
            dtype=dtype,
            order=order,
            like=like_outputs,
        ),
    )


@_implements(na.unit)
def unit(
        a: na.AbstractFunctionArray,
        squeeze: bool = True,
) -> None | u.UnitBase | na.AbstractArray:
    return na.unit(a.outputs, squeeze=squeeze)


@_implements(na.unit_normalized)
def unit_normalized(
        a: na.AbstractFunctionArray,
        squeeze: bool = True,
) -> u.UnitBase | na.AbstractArray:
    return na.unit_normalized(a.outputs, squeeze=squeeze)


@_implements(na.plt.pcolormesh)
def pcolormesh(
    *XY: na.AbstractArray,
    C: na.AbstractFunctionArray,
    components: None | tuple[str, str] = None,
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractArray = None,
    cmap: None | str | matplotlib.colors.Colormap = None,
    norm: None | str | matplotlib.colors.Normalize = None,
    vmin: None | na.ArrayLike = None,
    vmax: None | na.ArrayLike = None,
    **kwargs,
) -> na.ScalarArray:

    if len(XY) != 0:    # pragma: nocover
        raise ValueError(
            "if `C` is an instance of `na.AbstractFunctionArray`, "
            "`XY` must not be specified."
        )

    return na.plt.pcolormesh(
        C.inputs,
        C=C.outputs,
        components=components,
        axis_rgb=axis_rgb,
        ax=ax,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )
