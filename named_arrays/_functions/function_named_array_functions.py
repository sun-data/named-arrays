from typing import Callable, Literal, Sequence
import numpy as np
import matplotlib
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
]

ASARRAY_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.ASARRAY_LIKE_FUNCTIONS
NDFILTER_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.NDFILTER_FUNCTIONS
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


@_implements(na.broadcast_to)
def broadcast_to(
    array: na.AbstractFunctionArray,
    shape: dict[str, int],
    append: bool = False,
) -> na.FunctionArray:

    array = array.explicit

    axes_vertex = array.axes_vertex
    shape_inputs = {
        ax: shape[ax] + 1 if ax in axes_vertex else shape[ax]
        for ax in shape
    }

    return array.replace(
        inputs=na.broadcast_to(
            array=array.inputs,
            shape=shape_inputs,
            append=append,
        ),
        outputs=na.broadcast_to(
            array=array.outputs,
            shape=shape,
            append=append,
        ),
    )


@_implements(na.nominal)
def nominal(
    a: na.AbstractFunctionArray,
) -> na.FunctionArray:
    a = a.explicit
    return a.replace(
        inputs=na.nominal(a.inputs),
        outputs=na.nominal(a.outputs),
    )


@_implements(na.histogram)
def histogram(
    a: na.AbstractFunctionArray,
    bins: dict[str, int] | na.AbstractScalarArray,
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalarArray = None,
    max: None | na.AbstractScalarArray = None,
    density: bool = False,
    weights: None = None,
) -> na.FunctionArray[na.AbstractScalarArray, na.ScalarArray]:
    if weights is not None:  # pragma: nocover
        raise ValueError(
            "`weights` must be `None` for `AbstractFunctionArray`"
            f"inputs, got {type(weights)}."
        )

    axis_normalized = tuple(a.shape) if axis is None else (axis,) if isinstance(axis, str) else axis
    for ax in axis_normalized:
        if ax in a.axes_vertex:
            raise ValueError("Taking a histogram of a histogram doesn't work right now.")

    return na.histogram(
        a=a.inputs,
        bins=bins,
        axis=axis,
        min=min,
        max=max,
        density=density,
        weights=a.outputs,
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

    if len(C.axes_vertex) == 1:
        raise ValueError("Cannot plot single vertex axis with na.pcolormesh")

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


def ndfilter(
    func: Callable,
    array: na.AbstractFunctionArray,
    size: dict[str, int],
    where: bool | na.AbstractFunctionArray,
    **kwargs,
) -> na.FunctionArray:

    if isinstance(array, na.AbstractFunctionArray):
        pass
    else:
        return NotImplemented   # pragma: nocover

    array = array.explicit

    if isinstance(where, bool):
        where = na.FunctionArray(None, where)
    elif isinstance(where, na.AbstractFunctionArray):
        where = where.explicit
        if np.all(where.inputs != array.inputs):    # pragma: nocover
            raise ValueError(
                "if `where` is an instance of `na.AbstractFunctionArray`, "
                "its inputs must match `array`."
            )
    else:
        return NotImplemented   # pragma: nocover

    return array.replace(
        inputs=array.inputs.copy(),
        outputs=func(
            array=array.outputs,
            size=size,
            where=where.outputs,
            **kwargs,
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


@_implements(na.despike)
def despike(
    array: na.AbstractScalar | na.AbstractFunctionArray,
    axis: tuple[str, str],
    where: None | bool | na.AbstractScalar | na.AbstractFunctionArray,
    inbkg: None | na.AbstractScalar | na.AbstractFunctionArray,
    invar: None | float | na.AbstractScalar | na.AbstractFunctionArray,
    sigclip: float,
    sigfrac: float,
    objlim: float,
    gain: float,
    readnoise: float,
    satlevel: float,
    niter: int,
    sepmed: bool,
    cleantype: Literal["median", "medmask", "meanmask", "idw"],
    fsmode: Literal["median", "convolve"],
    psfmodel: Literal["gauss", "gaussx", "gaussy", "moffat"],
    psffwhm: float,
    psfsize: int,
    psfk: None | na.AbstractScalar,
    psfbeta: float,
    verbose: bool,
) -> na.ScalarArray:

    result = array.copy_shallow()

    if isinstance(array, na.AbstractFunctionArray):
        array = array.outputs
    if isinstance(where, na.AbstractFunctionArray):
        where = where.outputs
    if isinstance(inbkg, na.AbstractFunctionArray):
        inbkg = inbkg.outputs
    if isinstance(invar, na.AbstractFunctionArray):
        invar = invar.outputs

    result.outputs = na.despike(
        array=array,
        axis=axis,
        where=where,
        inbkg=inbkg,
        invar=invar,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        gain=gain,
        readnoise=readnoise,
        satlevel=satlevel,
        niter=niter,
        sepmed=sepmed,
        cleantype=cleantype,
        fsmode=fsmode,
        psfmodel=psfmodel,
        psffwhm=psffwhm,
        psfsize=psfsize,
        psfk=psfk,
        psfbeta=psfbeta,
        verbose=verbose,
    )

    return result
