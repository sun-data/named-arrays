from typing import Callable, TypeVar
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import astropy.units as u
import named_arrays as na
from named_arrays._scalars import scalars
import named_arrays._scalars.scalar_named_array_functions
from . import vectors

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
    "RANDOM_FUNCTIONS",
    "PLT_PLOT_LIKE_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
    "plt_plot_like",
]

InputT = TypeVar("InputT", bound="float | u.Quantity | na.AbstractVectorArray")
OutputT = TypeVar("OutputT", bound="float | u.Quantity | na.AbstractVectorArray")

ASARRAY_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.ASARRAY_LIKE_FUNCTIONS
RANDOM_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.RANDOM_FUNCTIONS
PLT_PLOT_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


def asarray_like(
        func: Callable,
        a: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        dtype: None | type | np.dtype = None,
        order: None | str = None,
        *,
        like: None | na.AbstractVectorArray = None,
) -> None | na.AbstractVectorArray:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractVectorArray):
            if isinstance(like, na.AbstractArray):
                if isinstance(like, na.AbstractVectorArray):
                    if a.type_explicit == like.type_explicit:
                        components_a = a.components
                        components_like = like.components
                        type_like = like.type_explicit
                    else:
                        return NotImplemented
                elif isinstance(like, na.AbstractScalar):
                    components_a = a.components
                    components_like = {c: like for c in components_a}
                    type_like = a.type_explicit
                else:
                    return NotImplemented
            else:
                components_a = a.components
                components_like = {c: like for c in components_a}
                type_like = a.type_explicit
        elif isinstance(a, na.AbstractScalar):
            if isinstance(like, na.AbstractVectorArray):
                components_like = like.components
                components_a = {c: a for c in components_like}
                type_like = like.type_explicit
            else:
                return NotImplemented
        else:
            return NotImplemented
    else:
        if isinstance(like, na.AbstractVectorArray):
            components_like = like.components
            components_a = {c: a for c in components_like}
            type_like = like.type_explicit
        else:
            return NotImplemented

    return type_like.from_components({
        c: func(
            a=components_a[c],
            dtype=dtype,
            order=order,
            like=components_like[c],
        ) for c in components_like
    })


@_implements(na.arange)
def arange(
        start: float | complex | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        stop: float | complex | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        axis: str | na.AbstractVectorArray,
        step: int | na.AbstractVectorArray = 1,
):
    prototype = vectors._prototype(start, stop, axis, step)

    start = vectors._normalize(start, prototype)
    stop = vectors._normalize(stop, prototype)
    axis = vectors._normalize(axis, prototype)
    step = vectors._normalize(step, prototype)

    components_start = start.components
    components_stop = stop.components
    components_axis = axis.components
    components_step = step.components

    components = {
        c: na.arange(
            start=components_start[c],
            stop=components_stop[c],
            axis=components_axis[c],
            step=components_step[c],
        )
        for c in components_start
    }

    return prototype.type_explicit.from_components(components)


@_implements(na.unit)
def unit(
        a: na.AbstractVectorArray,
        unit_dimensionless: None | float | u.UnitBase = None,
        squeeze: bool = True,
) -> None | u.UnitBase | na.AbstractVectorArray:
    components = a.components
    components = {
        c: na.unit(
            a=components[c],
            unit_dimensionless=unit_dimensionless,
            squeeze=squeeze)
        for c in components
    }
    iter_components = iter(components)
    component_0 = components[next(iter_components)]
    if squeeze:
        if all(component_0 == components[c] for c in components):
            return component_0

    components = {c: 1 if components[c] is None else 1 * components[c] for c in components}
    return a.type_explicit.from_components(components,)


@_implements(na.unit_normalized)
def unit_normalized(
        a: na.AbstractVectorArray,
        unit_dimensionless: float | u.UnitBase = u.dimensionless_unscaled,
        squeeze: bool = True,
) -> u.UnitBase | na.AbstractVectorArray:
    components = a.components
    components = {
        c: na.unit_normalized(
            components[c],
            unit_dimensionless=unit_dimensionless,
            squeeze=squeeze
        )
        for c in components
    }
    iter_components = iter(components)
    component_0 = components[next(iter_components)]
    if squeeze:
        if all(component_0 == components[c] for c in components):
            return component_0

    return a.type_explicit.from_components(components)


@_implements(na.interp)
def interp(
        x: float | u.Quantity | na.AbstractScalar,
        xp:  na.AbstractScalar,
        fp: na.AbstractVectorArray,
        axis: None | str = None,
        left: None | float | na.AbstractScalar | na.AbstractVectorArray = None,
        right: None | float | na.AbstractScalar | na.AbstractVectorArray = None,
        period: None | float| na.AbstractScalar | na.AbstractVectorArray = None,
) -> na.AbstractUncertainScalarArray:
    try:
        left = vectors._normalize(left, prototype=fp)
        right = vectors._normalize(right, prototype=fp)
    except na.VectorTypeError:
        return NotImplemented

    fp_cartesian = fp.cartesian_nd
    components_fp = fp_cartesian.components
    components_left = left.cartesian_nd.components
    components_right = right.cartesian_nd.components

    components_result = dict()
    for c in components_fp:
        components_result[c] = na.interp(
            x=x,
            xp=xp,
            fp=components_fp[c],
            axis=axis,
            left=components_left[c],
            right=components_right[c],
            period=period,
        )

    result = type(fp_cartesian)(components_result)
    result = fp.type_explicit.from_cartesian_nd(
        array=result,
        like=fp,
    )

    return result


def random(
        func: Callable,
        *args: float | u.Quantity |na.AbstractScalar | na.AbstractVectorArray,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity |na.AbstractScalar | na.AbstractVectorArray,
):
    try:
        prototype = vectors._prototype(*args, *kwargs.values())
        args = tuple(vectors._normalize(arg, prototype) for arg in args)
        kwargs = {k: vectors._normalize(kwargs[k], prototype) for k in kwargs}
    except na.VectorTypeError:
        return NotImplemented

    components_prototype = prototype.components

    components_args = {c: tuple(arg.components[c] for arg in args) for c in components_prototype}
    components_kwargs = {c: {k: kwargs[k].components[c] for k in kwargs} for c in components_prototype}

    components = {
        c: func(
            *components_args[c],
            shape_random=shape_random,
            seed=seed,
            **components_kwargs[c],
        )
        for c in prototype.components
    }

    return prototype.type_explicit.from_components(components)


def plt_plot_like(
        func: Callable,
        *args: na.AbstractCartesian2dVectorArray,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalarArray = True,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:

    if len(args) != 1:
        return NotImplemented

    a, = args

    if not isinstance(a, na.AbstractVectorArray):
        return NotImplemented

    a = a.cartesian_nd

    components_a = a.components

    if components is None:
        components = components_a

    args = tuple(na.as_named_array(components_a[c]) for c in components)

    return func(
        *args,
        ax=ax,
        axis=axis,
        where=where,
        **kwargs,
    )


@_implements(na.plt.pcolormesh)
def pcolormesh(
    *XY: na.AbstractVectorArray,
    C: na.AbstractScalarArray,
    components: None | tuple[str, str] = None,
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractScalarArray = None,
    cmap: None | str | matplotlib.colors.Colormap = None,
    norm: None | str | matplotlib.colors.Normalize = None,
    vmin: None | float | u.Quantity | na.AbstractScalarArray = None,
    vmax: None | float | u.Quantity | na.AbstractScalarArray = None,
    **kwargs,
) -> na.ScalarArray:
    try:
        C = scalars._normalize(C)
        vmin = scalars._normalize(vmin) if vmin is not None else vmin
        vmax = scalars._normalize(vmax) if vmax is not None else vmax
    except na.ScalarTypeError:
        return NotImplemented

    try:
        prototype = vectors._prototype(*XY)
        XY = tuple(vectors._normalize(arg, prototype) for arg in XY)
    except na.VectorTypeError:  # pragma: nocover
        return NotImplemented

    if len(XY) != 1:    # pragma: nocover
        raise ValueError("if any element of `XY` is a vector, `XY` must have a length of 1")
    XY = XY[0]

    components_XY = XY.components

    if components is None:
        if len(components_XY) == 2:
            components = components_XY.values()
        else:   # pragma: nocover
            raise ValueError(
                f"if `XY` is a vector and `components` is `None`, "
                f"`XY` must have exactly 2 components, got {len(components_XY)}."
            )
    else:
        components = [components_XY[c] for c in components]

    return na.plt.pcolormesh(
        *components,
        C=C,
        axis_rgb=axis_rgb,
        ax=ax,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )


@_implements(na.jacobian)
def jacobian(
        function: Callable[[InputT], OutputT],
        x: InputT,
        dx: None | InputT = None,
        like: None | OutputT = None,
) -> na.AbstractVectorArray | na.AbstractMatrixArray:

    f = function(x)

    if like is None:
        like = f

    like = na.asanyarray(like)

    type_x = x.type_explicit

    if isinstance(x, na.AbstractVectorArray):

        components_x = x.cartesian_nd.components
        components_dx = dx.cartesian_nd.components

        if isinstance(f, na.AbstractVectorArray):

            components_f = f.cartesian_nd.components

            components_result = {c: dict() for c in components_f}

            for c_x in components_x:
                components_x0 = components_x.copy()
                components_x0[c_x] = components_x0[c_x] + components_dx[c_x]
                x0 = type_x.from_cartesian_nd(na.CartesianNdVectorArray(components_x0), like=x)
                f0 = function(x0)
                df = f0 - f
                components_df = df.cartesian_nd.components
                for c_f in components_result:
                    components_result[c_f][c_x] = components_df[c_f] / components_dx[c_x]

            components_result = {
                c: type_x.from_cartesian_nd(na.CartesianNdVectorArray(components_result[c]), like=x)
                for c in components_result
            }

            result = like.type_matrix.from_cartesian_nd(na.CartesianNdVectorArray(components_result), like=like.matrix)

        elif isinstance(na.as_named_array(f), na.AbstractScalar):

            components_result = dict()

            for c_x in components_x:
                components_x0 = components_x.copy()
                components_x0[c_x] = components_x0[c_x] + components_dx[c_x]
                x0 = type_x.from_cartesian_nd(na.CartesianNdVectorArray(components_x0), like=x)
                f0 = function(x0)
                df = f0 - f
                components_result[c_x] = df / components_dx[c_x]

            result = na.CartesianNdVectorArray(components_result)
            result = type_x.from_cartesian_nd(result, like=x)

        else:
            return NotImplemented

    elif isinstance(na.as_named_array(x), na.AbstractScalar):
        x0 = x + dx
        f0 = function(x0)
        df = f0 - f
        return df / dx

    else:
        return NotImplemented

    return result


@_implements(na.optimize.root_newton)
def optimize_root_newton(
        function: Callable[[na.ScalarLike], na.ScalarLike],
        guess: na.ScalarLike,
        jacobian: Callable[[na.ScalarLike], na.ScalarLike],
        max_abs_error: na.ScalarLike,
        max_iterations: int = 100,
        callback: None | Callable[[int, na.ScalarLike, na.ScalarLike, na.ScalarLike], None] = None,
) -> na.ScalarArray:

    x = guess
    f = function(x)

    if not isinstance(x, na.AbstractVectorArray):
        return NotImplemented

    if not isinstance(f, na.AbstractVectorArray):
        return NotImplemented

    if na.shape(max_abs_error):
        raise ValueError(f"argument `max_abs_error` should have an empty shape, got {na.shape(max_abs_error)}")

    shape = na.shape_broadcasted(f, x)

    converged = na.broadcast_to(0 * na.value(f), shape=shape).astype(bool)

    x = na.broadcast_to(x, shape).astype(float)

    for i in range(max_iterations):

        if callback is not None:
            callback(i, x, f, converged)

        converged |= np.abs(f) < max_abs_error

        if np.all(converged):
            return x

        jac = jacobian(x)

        correction = jac.inverse @ f

        x = x - correction

        f = function(x)

    raise ValueError("Max iterations exceeded")


@_implements(na.colorsynth.rgb)
def colorsynth_rgb(
    spd: na.AbstractVectorArray,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | float | u.Quantity | na.AbstractVectorArray = None,
    spd_max: None | float | u.Quantity | na.AbstractVectorArray = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | float | u.Quantity | na.AbstractScalar = None,
    wavelength_max: None | float | u.Quantity | na.AbstractScalar = None,
    wavelength_norm: None | float | u.Quantity | na.AbstractScalar = None,
) -> na.AbstractVectorArray:
    try:
        p = vectors._prototype(
            spd,
            spd_min,
            spd_max,
        )
        spd = vectors._normalize(spd, prototype=p)
        spd_min = vectors._normalize(spd_min, prototype=p) if spd_min is not None else None
        spd_max = vectors._normalize(spd_max, prototype=p) if spd_max is not None else None
    except na.VectorTypeError:  # pragma: nocover
        return NotImplemented

    spd = spd.broadcasted.components
    spd_min = spd_min.broadcasted.components if spd_min is not None else None
    spd_max = spd_max.broadcasted.components if spd_max is not None else None

    result = dict()
    for c in spd:
        result[c] = na.colorsynth.rgb(
            spd=na.as_named_array(spd[c]),
            wavelength=wavelength,
            axis=axis,
            spd_min=spd_min[c] if spd_min is not None else None,
            spd_max=spd_max[c] if spd_max is not None else None,
            spd_norm=spd_norm,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            wavelength_norm=wavelength_norm,
        )

    result = p.type_explicit.from_components(result)

    return result


@_implements(na.optimize.minimum_gradient_descent)
def optimize_minimum_gradient_descent(
    function: Callable[[na.AbstractVectorArray], na.AbstractScalar],
    guess: na.AbstractVectorArray,
    step_size: float | na.AbstractScalar,
    gradient: None | Callable[[na.AbstractVectorArray], na.AbstractScalar],
    min_gradient: na.ScalarLike,
    max_iterations: int,
    callback: (
        None
        | Callable[[int, na.AbstractVectorArray, na.ScalarLike, na.ScalarLike], None]
    ),
) -> na.ScalarArray:

    x = guess
    f = function(x)

    if not isinstance(x, na.AbstractVectorArray):   # pragma: nocover
        return NotImplemented

    if not isinstance(na.as_named_array(f), na.AbstractScalar):  # pragma: nocover
        return NotImplemented

    if na.shape(min_gradient):  # pragma: nocover
        raise ValueError(
            f"argument `min_gradient` should have an empty shape, "
            f"got {na.shape(min_gradient)}"
        )

    shape = na.shape_broadcasted(f, x)

    converged = na.broadcast_to(0 * na.value(x), shape=shape).astype(bool)

    x = na.broadcast_to(x, shape).astype(float)

    for i in range(max_iterations):

        if callback is not None:
            callback(i, x, f, converged)

        grad = gradient(x)

        converged |= np.abs(grad) < min_gradient

        if np.all(converged):
            return x

        correction = step_size * grad

        x = x - correction

        f = function(x)

    raise ValueError("Max iterations exceeded")  # pragma: nocover


@_implements(na.ndfilters.mean_filter)
def mean_filter(
    array: na.AbstractVectorArray,
    size: dict[str, int],
    where: na.AbstractVectorArray,
) -> na.AbstractExplicitVectorArray:

    try:
        prototype = vectors._prototype(array, where)
        array = vectors._normalize(array, prototype)
        where = vectors._normalize(where, prototype)
    except vectors.VectorTypeError:     # pragma: nocover
        return NotImplemented

    components_array = array.broadcasted.components
    components_where = where.broadcasted.components

    result = dict()
    for c in components_array:
        result[c] = na.ndfilters.mean_filter(
            array=components_array[c],
            size=size,
            where=components_where[c],
        )

    result = prototype.type_explicit.from_components(result)

    return result
