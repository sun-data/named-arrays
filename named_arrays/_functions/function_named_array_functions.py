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
        unit_dimensionless: None | float | u.UnitBase = None,
        squeeze: bool = True,
) -> None | u.UnitBase | na.AbstractArray:
    return na.unit(
        a=a.outputs,
        unit_dimensionless=unit_dimensionless,
        squeeze=squeeze,
    )


@_implements(na.unit_normalized)
def unit_normalized(
        a: na.AbstractFunctionArray,
        unit_dimensionless: float | u.UnitBase,
        squeeze: bool = True,
) -> u.UnitBase | na.AbstractArray:
    return na.unit_normalized(
        a.outputs,
        unit_dimensionless=unit_dimensionless,
        squeeze=squeeze,
    )


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


@_implements(na.ndfilters.mean_filter)
def mean_filter(
    array: na.AbstractFunctionArray,
    size: dict[str, int],
    where: bool | na.AbstractFunctionArray,
) -> na.FunctionArray:

    if isinstance(array, na.AbstractFunctionArray):
        pass
    else:
        return NotImplemented   # pragma: nocover

    if isinstance(where, bool):
        where = na.FunctionArray(None, where)
    elif isinstance(where, na.AbstractFunctionArray):
        if np.all(where.inputs != array.inputs):    # pragma: nocover
            raise ValueError(
                f"if `where` is an instance of `na.AbstractFunctionArray`, "
                f"its inputs must match `array`."
            )
    else:
        return NotImplemented   # pragma: nocover

    return array.type_explicit(
        inputs=array.inputs.copy(),
        outputs=na.ndfilters.mean_filter(
            array=array.outputs,
            size=size,
            where=where.outputs,
        )
    )


@_implements(na.colorsynth.rgb)
def colorsynth_rgb(
    spd: na.AbstractFunctionArray,
    wavelength: None | na.AbstractScalarArray = None,
    axis: None | str = None,
    spd_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_norm: None | Callable = None,
) -> na.FunctionArray:
    return na.FunctionArray(
        inputs=spd.inputs.mean(axis),
        outputs=na.colorsynth.rgb(
            spd=spd.outputs,
            wavelength=wavelength,
            axis=axis,
            spd_min=spd_min,
            spd_max=spd_max,
            spd_norm=spd_norm,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            wavelength_norm=wavelength_norm,
        )
    )


@_implements(na.colorsynth.colorbar)
def colorsynth_colorbar(
    spd: na.AbstractFunctionArray,
    wavelength: None | na.AbstractScalarArray = None,
    axis: None | str = None,
    spd_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_norm: None | Callable = None,
) -> na.FunctionArray:
    return na.colorsynth.colorbar(
        spd=spd.outputs,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )
