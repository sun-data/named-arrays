from typing import Callable, Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions
from . import uncertainties

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
    "RANDOM_FUNCTIONS",
    "PLT_PLOT_LIKE_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
]

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
        a: None | float | u.Quantity | na.AbstractScalarArray | na.AbstractUncertainScalarArray,
        dtype: None | type | np.dtype = None,
        order: None | str = None,
        *,
        like: None | na.AbstractUncertainScalarArray = None,
) -> None | na.UncertainScalarArray:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractUncertainScalarArray):
            a_nominal = a.nominal
            a_distribution = a.distribution
        elif isinstance(a, na.AbstractScalarArray):
            a_nominal = a_distribution = a
        else:
            return NotImplemented
    else:
        a_nominal = a_distribution = a

    if isinstance(like, na.AbstractArray):
        if isinstance(like, na.AbstractUncertainScalarArray):
            like_nominal = like.nominal
            like_distribution = like.distribution
            type_like = like.type_explicit
        elif isinstance(like, na.AbstractScalarArray):
            like_nominal = like_distribution = like
            type_like = na.UncertainScalarArray
        else:
            return NotImplemented
    else:
        like_nominal = like_distribution = like
        type_like = na.UncertainScalarArray

    return type_like(
        nominal=func(
            a=a_nominal,
            dtype=dtype,
            order=order,
            like=like_nominal,
        ),
        distribution=func(
            a=a_distribution,
            dtype=dtype,
            order=order,
            like=like_distribution,
        ),
    )


@_implements(na.unit)
def unit(
        a: na.AbstractUncertainScalarArray,
        unit_dimensionless: None | float | u.UnitBase = None,
        squeeze: bool = True,
) -> None | u.UnitBase:
    return na.unit(
        a=a.nominal,
        unit_dimensionless=unit_dimensionless,
    )


@_implements(na.unit_normalized)
def unit_normalized(
        a: na.AbstractUncertainScalarArray,
        unit_dimensionless: float | u.UnitBase = u.dimensionless_unscaled,
        squeeze: bool = True,
) -> u.UnitBase:
    return na.unit_normalized(
        a.nominal,
        unit_dimensionless=unit_dimensionless,
    )


@_implements(na.interp)
def interp(
        x: float | u.Quantity | na.AbstractScalar,
        xp:  na.AbstractScalar,
        fp: na.AbstractScalar,
        axis: None | str = None,
        left: None | float | u.Quantity | na.AbstractScalar = None,
        right: None | float | u.Quantity | na.AbstractScalar = None,
        period: None | float | u.Quantity | na.AbstractScalar = None,
) -> na.AbstractUncertainScalarArray:
    try:
        x = uncertainties._normalize(x)
        xp = uncertainties._normalize(xp)
        fp = uncertainties._normalize(fp)
        left = uncertainties._normalize(left)
        right = uncertainties._normalize(right)
        period = uncertainties._normalize(period)
    except na.UncertainScalarTypeError:
        return NotImplemented

    result = x.type_explicit(
        nominal=na.interp(
            x=x.nominal,
            xp=xp.nominal,
            fp=fp.nominal,
            left=left.nominal,
            right=right.nominal,
            period=period.nominal,
        ),
        distribution=na.interp(
            x=x.distribution,
            xp=xp.distribution,
            fp=fp.distribution,
            left=left.distribution,
            right=right.distribution,
            period=period.distribution,
        ),
    )

    return result


@_implements(na.histogram2d)
def histogram2d(
    x: na.AbstractScalar,
    y: na.AbstractScalar,
    bins: dict[str, int] | na.AbstractCartesian2dVectorArray,
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalar | na.AbstractCartesian2dVectorArray = None,
    max: None | na.AbstractScalar | na.AbstractCartesian2dVectorArray = None,
    density: bool = False,
    weights: None | na.AbstractScalar = None,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.UncertainScalarArray]:
    try:
        x = uncertainties._normalize(x).broadcasted
        y = uncertainties._normalize(y).broadcasted
        weights = uncertainties._normalize(weights)

        if isinstance(bins, na.AbstractCartesian2dVectorArray):
            bins = na.Cartesian2dVectorArray(
                x=uncertainties._normalize(bins.x),
                y=uncertainties._normalize(bins.y),
            )
            bins_nominal = na.Cartesian2dVectorArray(
                x=bins.x.nominal,
                y=bins.y.nominal,
            )
            bins_distribution = na.Cartesian2dVectorArray(
                x=bins.x.distribution,
                y=bins.y.distribution,
            )
        else:
            bins_nominal = bins_distribution = bins

        if min is not None:
            if not isinstance(min, na.AbstractCartesian2dVectorArray):
                min = na.Cartesian2dVectorArray.from_scalar(min)
            min = na.Cartesian2dVectorArray(
                x=uncertainties._normalize(min.x),
                y=uncertainties._normalize(min.y),
            )
            min_nominal = na.Cartesian2dVectorArray(
                x=min.x.nominal,
                y=min.y.nominal,
            )
            min_distribution = na.Cartesian2dVectorArray(
                x=min.x.distribution,
                y=min.y.distribution,
            )
        else:
            min_nominal = min_distribution = min

        if max is not None:
            if not isinstance(max, na.AbstractCartesian2dVectorArray):
                max = na.Cartesian2dVectorArray.from_scalar(max)
            max = na.Cartesian2dVectorArray(
                x=uncertainties._normalize(max.x),
                y=uncertainties._normalize(max.y),
            )
            max_nominal = na.Cartesian2dVectorArray(
                x=max.x.nominal,
                y=max.y.nominal,
            )
            max_distribution = na.Cartesian2dVectorArray(
                x=max.x.distribution,
                y=max.y.distribution,
            )
        else:
            max_nominal = max_distribution = max

    except na.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    result_nominal = na.histogram2d(
        x=x.nominal,
        y=y.nominal,
        bins=bins_nominal,
        axis=axis,
        min=min_nominal,
        max=max_nominal,
        density=density,
        weights=weights.nominal,
    )

    result_distribution = na.histogram2d(
        x=x.distribution,
        y=y.distribution,
        bins=bins_distribution,
        axis=axis,
        min=min_distribution,
        max=max_distribution,
        density=density,
        weights=weights.distribution,
    )

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=na.UncertainScalarArray(
                nominal=result_nominal.inputs.x,
                distribution=result_distribution.inputs.x,
            ),
            y=na.UncertainScalarArray(
                nominal=result_nominal.inputs.y,
                distribution=result_distribution.inputs.y,
            ),
        ),
        outputs=na.UncertainScalarArray(
            nominal=result_nominal.outputs,
            distribution=result_distribution.outputs,
        ),
    )


def random(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalar,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity | na.AbstractScalar,
) -> na.UncertainScalarArray:

    try:
        args = tuple(uncertainties._normalize(arg) for arg in args)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except na.UncertainScalarTypeError:
        return NotImplemented

    return na.UncertainScalarArray(
        nominal=func(
            *tuple(arg.nominal for arg in args),
            shape_random=shape_random,
            seed=seed,
            **{k: kwargs[k].nominal for k in kwargs},
        ),
        distribution=func(
            *tuple(arg.distribution for arg in args),
            shape_random=shape_random,
            seed=seed,
            **{k: kwargs[k].distribution for k in kwargs},
        )
    )


def plt_plot_like(
        func: Callable,
        *args: na.AbstractScalar,
        ax: None | matplotlib.axes.Axes = None,
        axis: None | str = None,
        where: bool | na.AbstractScalarArray = True,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.UncertainScalarArray[
    npt.NDArray[matplotlib.artist.Artist],
    npt.NDArray[matplotlib.artist.Artist]
]:

    if components is not None:
        raise ValueError(f"`components` should be `None` for scalars, got {components}")

    try:
        args = tuple(uncertainties._normalize(arg) for arg in args)
        where = uncertainties._normalize(where)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except na.UncertainScalarTypeError:
        return NotImplemented

    shape = na.shape_broadcasted(*args)

    if axis is None:
        if len(shape) != 1:
            raise ValueError(
                f"if `axis` is `None`, the broadcasted shape of `*args`, {shape}, should have one element"
            )
        axis = next(iter(shape))

    args = tuple(arg.broadcast_to(shape) for arg in args)

    axis_distribution = args[0].axis_distribution
    shape_distribution = na.broadcast_shapes(*[arg.shape_distribution for arg in args])

    if axis_distribution in shape_distribution:
        num_distribution = shape_distribution[axis_distribution]
    else:
        num_distribution = 1

    if num_distribution == 0:
        alpha = 1
    else:
        alpha = max(1 / num_distribution, 1/255)
    if "alpha" in kwargs:
        kwargs["alpha"] *= alpha
    else:
        kwargs["alpha"] = na.UncertainScalarArray(1, alpha)

    if "color" not in kwargs:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        shape_orthogonal = {a: shape[a] for a in shape if a != axis}
        color = na.ScalarArray.empty(shape=shape_orthogonal, dtype=object)
        for i, index in enumerate(color.ndindex()):
            color[index] = color_cycle[i % len(color_cycle)]
        kwargs["color"] = uncertainties._normalize(color)

    result = na.UncertainScalarArray(
        nominal=func(
            *tuple(na.as_named_array(arg.nominal) for arg in args),
            ax=ax,
            axis=axis,
            where=where.nominal,
            **{k: kwargs[k].nominal for k in kwargs}
        ),
        distribution=func(
            *tuple(na.as_named_array(arg.distribution) for arg in args),
            ax=ax,
            axis=axis,
            where=where.distribution,
            **{k: kwargs[k].distribution for k in kwargs}
        )
    )

    return result


@_implements(na.plt.scatter)
def plt_scatter(
        *args: na.AbstractScalar,
        s: None | na.AbstractScalar = None,
        c: None | na.AbstractScalar = None,
        ax: None | matplotlib.axes.Axes | na.ScalarArray = None,
        where: bool | na.AbstractScalar = True,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.UncertainScalarArray:

    if components is not None:
        raise ValueError(
            f"`components` should be `None` for scalars, got {components}"
        )

    try:
        args = tuple(uncertainties._normalize(arg) for arg in args)
        s = uncertainties._normalize(s)
        c = uncertainties._normalize(c)
        where = uncertainties._normalize(where)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except na.UncertainScalarTypeError:
        return NotImplemented

    if ax is None:
        ax = plt.gca()
    ax = na.as_named_array(ax)

    axis_distribution = args[0].axis_distribution
    shape_distribution = na.broadcast_shapes(*[arg.shape_distribution for arg in args])

    if axis_distribution in shape_distribution:
        num_distribution = shape_distribution[axis_distribution]
    else:
        num_distribution = 1

    if num_distribution == 0:
        alpha = 1
    else:
        alpha = max(1 / num_distribution, 1 / 255)

    if "alpha" in kwargs:
        kwargs["alpha"] = kwargs["alpha"] * alpha
    else:
        kwargs["alpha"] = na.UncertainScalarArray(1, alpha)

    result_nominal = na.plt.scatter(
        *[na.as_named_array(arg.nominal) for arg in args],
        s=s.nominal,
        c=c.nominal,
        ax=ax,
        where=where.nominal,
        components=components,
        **{k: kwargs[k].nominal for k in kwargs},
    )

    if c.distribution is None:
        c_distribution = na.ScalarArray.zeros(shape=ax.shape | dict(rgba=4))
        for index in ax.ndindex():
            facecolor = result_nominal[index].ndarray.get_facecolor()[0]
            c_distribution[index] = na.ScalarArray(
                ndarray=facecolor,
                axes="rgba",
            )
        c.distribution = c_distribution

    result_distribution = na.plt.scatter(
        *[na.as_named_array(arg.distribution) for arg in args],
        s=s.distribution,
        c=c.distribution,
        ax=ax,
        where=where.distribution,
        components=components,
        **{k: kwargs[k].distribution for k in kwargs},
    )

    result = na.UncertainScalarArray(
        nominal=result_nominal,
        distribution=result_distribution,
    )

    return result


@_implements(na.jacobian)
def jacobian(
        function: Callable[[na.AbstractScalar], na.AbstractScalar],
        x: na.AbstractScalar,
        dx: None | na.AbstractScalar = None,
        like: None | na.AbstractScalar = None,
) -> na.AbstractScalar:
    return named_arrays._scalars.scalar_named_array_functions.jacobian(
        function=function,
        x=x,
        dx=dx,
        like=like,
    )


@_implements(na.optimize.root_newton)
def optimize_root_newton(
        function: Callable[[na.ScalarLike], na.ScalarLike],
        guess: na.ScalarLike,
        jacobian: Callable[[na.ScalarLike], na.ScalarLike],
        max_abs_error: na.ScalarLike,
        max_iterations: int = 100,
        callback: None | Callable[[int, na.ScalarLike, na.ScalarLike, na.ScalarLike], None] = None,
) -> na.UncertainScalarArray:
    return named_arrays._scalars.scalar_named_array_functions.optimize_root_newton(
        function=function,
        guess=guess,
        jacobian=jacobian,
        max_abs_error=max_abs_error,
        max_iterations=max_iterations,
        callback=callback,
    )


@_implements(na.optimize.root_secant)
def optimize_root_secant(
        function: Callable[[na.ScalarLike], na.ScalarLike],
        guess: na.ScalarLike,
        min_step_size: na.ScalarLike,
        max_abs_error: na.ScalarLike,
        max_iterations: int = 100,
        damping: None | float = None,
        callback: None | Callable[[int, na.ScalarLike, na.ScalarLike, na.ScalarLike], None] = None,
) -> na.UncertainScalarArray:

    try:
        guess = uncertainties._normalize(guess).astype(float)

        min_step_size = uncertainties._normalize(min_step_size)

        x0 = guess - 10 * min_step_size
        x1 = guess

        f0 = function(x0)
        f0 = uncertainties._normalize(f0)

        max_abs_error = uncertainties._normalize(max_abs_error)

    except uncertainties.UncertainScalarTypeError:
        return NotImplemented

    if na.shape(max_abs_error):
        raise ValueError(f"argument `max_abs_error` should have an empty shape, got {na.shape(max_abs_error)}")

    shape = na.shape_broadcasted(f0, guess, min_step_size)
    shape_distribution = na.broadcast_shapes(
        f0.shape_distribution,
        guess.shape_distribution,
        min_step_size.shape_distribution,
    )

    f0 = na.broadcast_to(f0, shape)
    f0.distribution = na.broadcast_to(f0.distribution, shape_distribution)

    converged = na.broadcast_to(0 * na.value(f0), shape=shape).astype(bool)

    x1 = na.broadcast_to(x1, shape)
    x1.distribution = na.broadcast_to(x1.distribution, shape_distribution)

    for i in range(max_iterations):

        f1 = function(x1)

        if callback is not None:
            callback(i, x1, f1, converged)

        if max_abs_error is not None:
            converged |= np.abs(f1) < max_abs_error

        dx = x1 - x0

        converged |= np.abs(dx) < np.abs(min_step_size)

        if np.all(converged):
            return x1

        active = ~converged

        dx_active = na.UncertainScalarArray(
            nominal=dx.nominal[active.nominal],
            distribution=dx.distribution[active.distribution],
        )
        f0_active = na.UncertainScalarArray(
            nominal=f0.nominal[active.nominal],
            distribution=f0.distribution[active.distribution],
        )
        f1_active = na.UncertainScalarArray(
            nominal=f1.nominal[active.nominal],
            distribution=f1.distribution[active.distribution],
        )

        df_active = f1_active - f0_active
        if np.any(df_active.nominal == 0) or np.any(df_active.distribution == 0):
            raise ValueError("stationary point detected")

        jacobian = df_active / dx_active

        correction = f1_active / jacobian
        if damping is not None:
            correction = damping * correction

        x2 = x1.copy()
        x2.nominal[active.nominal] = x2.nominal[active.nominal] - correction.nominal
        x2.distribution[active.distribution] = x2.distribution[active.distribution] - correction.distribution

        x0 = x1
        x1 = x2
        f0 = f1

    raise ValueError("Max iterations exceeded")


def _colorsynth_normalize(
    spd: na.AbstractScalar,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | na.ScalarLike = None,
    spd_max: None | na.ScalarLike = None,
    wavelength_min: None | na.ScalarLike = None,
    wavelength_max: None | na.ScalarLike = None,
) -> tuple[
    na.AbstractUncertainScalarArray,
    na.AbstractUncertainScalarArray,
    str,
    na.AbstractUncertainScalarArray,
    na.AbstractUncertainScalarArray,
    na.AbstractUncertainScalarArray,
    na.AbstractUncertainScalarArray,
]:

    try:
        spd = uncertainties._normalize(spd)
    except na.UncertainScalarArray:     # pragma: nocover
        return NotImplemented

    spd = spd.broadcasted
    shape = spd.shape

    if axis is None:
        if len(shape) != 1:
            raise ValueError(
                f"If `axis` is `None`, the shape of `array` should have only"
                f"one element, got {shape=}"
            )
        else:
            axis = next(iter(shape))

    if wavelength is None:
        wavelength = na.linspace(0, 1, axis=axis, num=shape[axis])

    try:
        wavelength = uncertainties._normalize(wavelength)
    except na.UncertainScalarArray:     # pragma: nocover
        return NotImplemented

    if spd_min is None:
        spd_min = np.minimum(
            spd.nominal.min(),
            spd.distribution.min(),
        )
    if spd_max is None:
        spd_max = np.maximum(
            spd.nominal.max(),
            spd.distribution.max(),
        )
    if wavelength_min is None:
        wavelength_min = np.minimum(
            wavelength.nominal.min(),
            wavelength.distribution.min(),
        )
    if wavelength_max is None:
        wavelength_max = np.maximum(
            wavelength.nominal.max(),
            wavelength.distribution.max(),
        )

    try:
        spd_min = uncertainties._normalize(spd_min)
        spd_max = uncertainties._normalize(spd_max)
        wavelength_min = uncertainties._normalize(wavelength_min)
        wavelength_max = uncertainties._normalize(wavelength_max)
    except na.UncertainScalarArray:     # pragma: nocover
        return NotImplemented

    return (
        spd,
        wavelength,
        axis,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

@_implements(na.colorsynth.rgb)
def colorsynth_rgb(
    spd: na.AbstractScalar,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | na.ScalarLike = None,
    spd_max: None | na.ScalarLike = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | na.ScalarLike = None,
    wavelength_max: None | na.ScalarLike = None,
    wavelength_norm: None | Callable = None,
) -> na.UncertainScalarArray:

    (
        spd,
        wavelength,
        axis,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    ) = _colorsynth_normalize(
        spd,
        wavelength,
        axis,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

    return na.UncertainScalarArray(
        nominal=na.colorsynth.rgb(
            spd=spd.nominal,
            wavelength=wavelength.nominal,
            axis=axis,
            spd_min=spd_min.nominal,
            spd_max=spd_max.nominal,
            spd_norm=spd_norm,
            wavelength_min=wavelength_min.nominal,
            wavelength_max=wavelength_max.nominal,
            wavelength_norm=wavelength_norm,
        ),
        distribution=na.colorsynth.rgb(
            spd=spd.distribution,
            wavelength=wavelength.distribution,
            axis=axis,
            spd_min=spd_min.distribution,
            spd_max=spd_max.distribution,
            spd_norm=spd_norm,
            wavelength_min=wavelength_min.distribution,
            wavelength_max=wavelength_max.distribution,
            wavelength_norm=wavelength_norm,
        ),
    )


@_implements(na.colorsynth.colorbar)
def colorsynth_colorbar(
    spd: na.AbstractScalar,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | na.ScalarLike = None,
    spd_max: None | na.ScalarLike = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | na.ScalarLike = None,
    wavelength_max: None | na.ScalarLike = None,
    wavelength_norm: None | Callable = None,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.UncertainScalarArray]:

    (
        spd,
        wavelength,
        axis,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    ) = _colorsynth_normalize(
        spd,
        wavelength,
        axis,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

    result_nominal = na.colorsynth.colorbar(
        spd=na.as_named_array(spd.nominal),
        wavelength=wavelength.nominal,
        axis=axis,
        spd_min=spd_min.nominal,
        spd_max=spd_max.nominal,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min.nominal,
        wavelength_max=wavelength_max.nominal,
        wavelength_norm=wavelength_norm,
    )

    result_distribution = na.colorsynth.colorbar(
        spd=na.as_named_array(spd.distribution),
        wavelength=wavelength.distribution,
        axis=axis,
        spd_min=spd_min.distribution,
        spd_max=spd_max.distribution,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min.distribution,
        wavelength_max=wavelength_max.distribution,
        wavelength_norm=wavelength_norm,
    )

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=na.UncertainScalarArray(
                nominal=result_nominal.inputs.x,
                distribution=result_distribution.inputs.x,
            ),
            y=na.UncertainScalarArray(
                nominal=result_nominal.inputs.y,
                distribution=result_distribution.inputs.y,
            ),
        ),
        outputs=na.UncertainScalarArray(
            nominal=result_nominal.outputs,
            distribution=result_distribution.outputs,
        )
    )


@_implements(na.ndfilters.mean_filter)
def mean_filter(
    array: na.AbstractScalar,
    size: dict[str, int],
    where: na.AbstractScalar,
) -> na.UncertainScalarArray:

    try:
        array = uncertainties._normalize(array)
        where = uncertainties._normalize(where)
    except uncertainties.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    array = array.broadcasted
    where = where.broadcasted

    return array.type_explicit(
        nominal=na.ndfilters.mean_filter(
            array=array.nominal,
            size=size,
            where=where.nominal,
        ),
        distribution=na.ndfilters.mean_filter(
            array=array.distribution,
            size=size,
            where=where.distribution,
        )
    )