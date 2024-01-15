from typing import Callable
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
        squeeze: bool = True,
) -> None | u.UnitBase:
    return na.unit(a.nominal)


@_implements(na.unit_normalized)
def unit_normalized(
        a: na.AbstractUncertainScalarArray,
        squeeze: bool = True,
) -> u.UnitBase:
    return na.unit_normalized(a.nominal)


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
