from typing import Callable, Literal, Sequence, TYPE_CHECKING
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colors

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


@_implements(na.debroadcast)
def debroadcast(
    array: na.AbstractFunctionArray,
    axes: None | str | Sequence[str] = None,
) -> na.FunctionArray:
    array = array.explicit
    shape = array.shape

    if axes is None:
        axes = tuple(shape)
    elif isinstance(axes, str):
        axes = (axes,)

    inputs = array.inputs
    outputs = array.outputs
    shape_inputs = na.shape(inputs)

    # A vertex axis represents bin edges, which vary along the axis and so are
    # never constant, and cannot be sliced symmetrically with the bin centers.
    axes_vertex = array.axes_vertex

    index = dict()
    for axis in axes:
        if axis not in shape:
            continue
        if axis in axes_vertex:
            continue
        # ``outputs`` and ``inputs`` are compared separately (rather than the
        # whole function) so that differing coordinates along ``axis`` register
        # as "not constant" instead of raising.
        # Empty axes are never removed (see :func:`_debroadcast`).
        constant = shape[axis] != 0
        constant = constant and bool(np.all(outputs == outputs[{axis: slice(0, 1)}]))
        if constant and axis in shape_inputs:
            constant = bool(np.all(inputs == inputs[{axis: slice(0, 1)}]))
        if constant:
            index[axis] = 0

    return array[index]


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
    ax: "None | matplotlib.axes.Axes | na.AbstractArray" = None,
    cmap: "None | str | matplotlib.colors.Colormap" = None,
    norm: "None | str | matplotlib.colors.Normalize" = None,
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


def _offsets_cells(offset: na.AbstractScalar) -> na.AbstractScalar:
    """
    Express the offsets of a kernel as plain numbers of cells.

    Parameters
    ----------
    offset
        One component of the offsets, either dimensionless or in pixels.
    """
    unit = na.unit(offset)
    if unit is None:
        return offset
    for unit_cells in (u.pix, u.dimensionless_unscaled):
        if unit.is_equivalent(unit_cells):
            return offset.to(unit_cells).value
    raise ValueError(
        f"the offsets of the kernel must be dimensionless or in pixels, got {unit}"
    )


@_implements(na.regridding.convolve_weights)
def regridding_convolve_weights(
    weights: na.AbstractScalar,
    shape_input: dict[str, int],
    shape_output: dict[str, int],
    kernel: na.AbstractFunctionArray,
    axis_output: None | str | Sequence[str] = None,
) -> tuple[na.ScalarArray, dict[str, int], dict[str, int]]:

    import regridding

    weights = na.as_named_array(weights)
    if not isinstance(weights, na.AbstractScalarArray):  # pragma: nocover
        return NotImplemented
    weights = weights.explicit

    shape_weights = weights.shape

    # the resampled axes of the grids are the ones the weights are not an
    # array over, as `regrid_from_weights` infers them
    axis_input = tuple(a for a in shape_input if a not in shape_weights)

    if axis_output is None:
        resampled = tuple(a for a in shape_output if a not in shape_weights)
    else:
        if isinstance(axis_output, str):
            axis_output = (axis_output,)
        resampled = tuple(axis_output)
        if len(set(resampled)) != len(resampled):
            raise ValueError(f"axis_output={resampled} names an axis twice")
        for axis in resampled:
            if axis not in shape_output:
                raise ValueError(
                    f"{axis!r} in axis_output={resampled} is not an axis of the "
                    f"output grid, {shape_output}"
                )
            if axis in shape_weights:
                raise ValueError(
                    f"{axis!r} in axis_output={resampled} is an orthogonal axis "
                    f"of the weights, which are an array over {shape_weights}"
                )

    if len(resampled) != len(axis_input):
        raise ValueError(
            f"the weights resample the {len(axis_input)} input axes "
            f"{axis_input}, but axis_output={resampled} names "
            f"{len(resampled)} output axes"
        )

    kernel = kernel.explicit

    offsets = kernel.inputs
    if isinstance(offsets, na.AbstractVectorArray):
        components = dict(offsets.cartesian_nd.components)
    else:
        components = {None: offsets}
    components = {k: na.as_named_array(c) for k, c in components.items()}

    values = na.as_named_array(kernel.outputs)

    for array in (values, *components.values()):
        if not isinstance(array, na.AbstractScalarArray):
            raise TypeError(
                f"the inputs and outputs of the kernel must be scalars or "
                f"vectors of scalars without uncertainty, got {type(array)}"
            )

    if len(components) != len(resampled):
        raise ValueError(
            f"the offsets of the kernel have {len(components)} components, "
            f"but the weights resample {len(resampled)} output axes, {resampled}"
        )

    # the output axis each component of the offsets acts along: the one it is
    # named after, if every component is named after one, or else the one at
    # the same position in `axis_output`, which then has to be given
    if set(components) == set(resampled):
        component_of = {a: a for a in resampled}
    elif axis_output is None and len(components) > 1:
        raise ValueError(
            f"the components of the kernel's offsets, {tuple(components)}, are "
            f"not named after the resampled output axes, {resampled}, so "
            f"axis_output has to be given to say which axis each acts along"
        )
    else:
        component_of = dict(zip(resampled, components))

    components = {k: _offsets_cells(c) for k, c in components.items()}

    unit = na.unit(values)
    if unit is not None:
        if not unit.is_equivalent(u.dimensionless_unscaled):
            raise ValueError(f"the kernel must be dimensionless, got {unit}")
        values = values.to(u.dimensionless_unscaled).value

    # the axes of the kernel itself are the axes the offsets vary along; any
    # other axis of the offsets is broadcast like the other axes of `values`
    shape_offsets = na.broadcast_shapes(*[c.shape for c in components.values()])
    offsets_full = {
        k: np.asarray(
            c.broadcast_to(shape_offsets).ndarray_aligned(shape_offsets),
            dtype=float,
        )
        for k, c in components.items()
    }
    for offset in offsets_full.values():
        if not np.all(np.isfinite(offset)):
            raise ValueError(f"the offsets of the kernel must be finite, got {offset}")
    shape_kernel = {
        a: n
        for i, (a, n) in enumerate(shape_offsets.items())
        if any(np.any(np.diff(o, axis=i) != 0) for o in offsets_full.values())
    }
    index_kernel = tuple(
        slice(None) if a in shape_kernel else 0 for a in shape_offsets
    )

    for axis in shape_kernel:
        if axis in shape_output:
            raise ValueError(
                f"the offsets of the kernel vary along {axis!r}, which is an "
                f"axis of the output grid, {shape_output}"
            )

    offsets_cells = dict()
    for k, offset in offsets_full.items():
        offset = offset[index_kernel].reshape(-1)
        offset_rounded = np.rint(offset)
        if not np.allclose(offset, offset_rounded, rtol=0, atol=1e-6):
            raise ValueError(
                f"the offsets of the kernel must be whole numbers of cells, "
                f"got {offset}"
            )
        offsets_cells[k] = offset_rounded.astype(np.int64)

    offsets_stacked = np.stack(list(offsets_cells.values()), axis=~0)
    if len(np.unique(offsets_stacked, axis=0)) != len(offsets_stacked):
        raise ValueError(
            "the kernel lists an offset more than once; a kernel which varies "
            "along another axis should not have offsets which vary along it"
        )

    # an axis of length one carries no variation, so it is not one to add
    for axis, num in values.shape.items():
        if num == 1 and axis not in shape_kernel:
            values = values[{axis: 0}]

    shape_extra = {a: n for a, n in values.shape.items() if a not in shape_kernel}

    for axis in shape_extra:
        if axis in axis_input and axis not in resampled:
            raise ValueError(
                f"the kernel varies along {axis!r}, a resampled axis of the "
                f"input grid; a kernel which varies across the field varies "
                f"along the output axes, {resampled}, instead"
            )

    # broadcast the orthogonal axes of the weights against those of the
    # grids and of the kernel, so that the kernel may add one
    shape_orthogonal = na.broadcast_shapes(
        shape_weights,
        {a: n for a, n in shape_output.items() if a not in resampled},
        {a: n for a, n in shape_input.items() if a not in axis_input},
        {a: n for a, n in shape_extra.items() if a not in resampled},
    )
    axes_orthogonal = tuple(shape_orthogonal)
    num_orthogonal = len(axes_orthogonal)

    # lay both grids out with the orthogonal axes first and the resampled
    # axes after them, in the order the flat indices of the weights address
    # them, which is their order in the shapes the weights were built with
    axes_output = tuple(a for a in shape_output if a in resampled)

    # the kernel is built directly in the layout `regridding.convolve_weights`
    # takes: its orthogonal axes, its resampled axes, and then the stencil,
    # with one axis for each resampled axis, in the same order
    values = values.broadcast_to(shape_extra | shape_kernel)
    values = values.ndarray_aligned(axes_orthogonal + axes_output + tuple(shape_kernel))
    values = np.asarray(values, dtype=float)
    shape_lead = values.shape[: num_orthogonal + len(axes_output)]
    values = values.reshape(shape_lead + (-1,))

    # scatter the kernel into a dense stencil centered on zero offset, so the
    # offsets need be neither centered nor contiguous
    half = {
        a: int(np.abs(offsets_cells[component_of[a]]).max(initial=0))
        for a in axes_output
    }
    stencil = np.zeros(shape_lead + tuple(2 * half[a] + 1 for a in axes_output))
    index_stencil = tuple(offsets_cells[component_of[a]] + half[a] for a in axes_output)
    stencil[(Ellipsis,) + index_stencil] = values

    weights_ndarray = weights.broadcast_to(shape_orthogonal)
    weights_ndarray = weights_ndarray.ndarray_aligned(axes_orthogonal)

    result, _, _ = regridding.convolve_weights(
        weights=(
            weights_ndarray,
            tuple(shape_orthogonal.values()) + tuple(shape_input[a] for a in axis_input),
            tuple(shape_orthogonal.values()) + tuple(shape_output[a] for a in axes_output),
        ),
        kernel=stencil,
        axis_input=tuple(range(num_orthogonal, num_orthogonal + len(axis_input))),
        axis_output=tuple(range(num_orthogonal, num_orthogonal + len(axes_output))),
    )

    result = na.ScalarArray(result, axes_orthogonal)

    # the orthogonal axes come first, as `weights` returns them
    shape_input = na.broadcast_shapes(shape_orthogonal, shape_input)
    shape_output = na.broadcast_shapes(shape_orthogonal, shape_output)

    return result, shape_input, shape_output
