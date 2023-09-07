from typing import Callable, TypeVar
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
from . import scalars

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
    "RANDOM_FUNCTIONS",
    "PLT_PLOT_LIKE_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
]

ASARRAY_LIKE_FUNCTIONS = (
    na.asarray,
    na.asanyarray,
)
RANDOM_FUNCTIONS = (
    na.random.uniform,
    na.random.normal,
    na.random.poisson,
)
PLT_PLOT_LIKE_FUNCTIONS = (
    na.plt.plot,
    na.plt.fill,
)
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


def asarray_like(
        func: Callable,
        a: None | float | u.Quantity | na.AbstractScalarArray,
        dtype: None | type | np.dtype = None,
        order: None | str = None,
        *,
        like: None | na.AbstractScalarArray = None,
) -> None | na.ScalarArray:

    func_numpy = getattr(np, func.__name__)

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalarArray):
            a_ndarray = a.ndarray
            a_axes = a.axes
        else:
            return NotImplemented
    else:
        a_ndarray = a
        a_axes = None

    if isinstance(like, na.AbstractArray):
        if isinstance(like, na.AbstractScalarArray):
            like_ndarray = like.ndarray
            type_like = like.type_explicit
        else:
            return NotImplemented
    else:
        like_ndarray = like
        type_like = na.ScalarArray

    if isinstance(like_ndarray, u.Quantity):
        like_ndarray = like_ndarray.value
    like_ndarray = func_numpy(like_ndarray)

    return type_like(
        ndarray=func_numpy(
            a=a_ndarray,
            dtype=dtype,
            order=order,
            like=like_ndarray,
        ),
        axes=a_axes,
    )


@_implements(na.arange)
def arange(
        start: float | complex | u.Quantity | na.AbstractArray,
        stop: float | complex | u.Quantity | na.AbstractArray,
        axis: str | na.AbstractArray,
        step: int | na.AbstractArray = 1,
) -> na.ScalarArray:

    start = scalars._normalize(start)
    stop = scalars._normalize(stop)

    if start.size > 1:
        raise ValueError(f"`start` must have only one element, got shape {start.shape}")

    if stop.size > 1:
        raise ValueError(f"`stop` must have only one element, got shape {stop.shape}")

    return na.ScalarArray(
        ndarray=np.arange(
            start=start.ndarray,
            stop=stop.ndarray,
            step=step,
        ),
        axes=(axis,),
    )


@_implements(na.unit)
def unit(a: na.AbstractScalarArray) -> None | u.UnitBase:
    return na.unit(a.ndarray)


@_implements(na.unit_normalized)
def unit_normalized(a: na.AbstractScalarArray) -> u.UnitBase:
    result = na.unit(a)
    if result is None:
        result = u.dimensionless_unscaled
    return result


def random(
        func: Callable,
        *args: float | u.Quantity | na.AbstractScalarArray,
        shape_random: None | dict[str, int] = None,
        seed: None | int = None,
        **kwargs: float | u.Quantity | na.AbstractScalarArray,
) -> na.ScalarArray:

    try:
        args = tuple(scalars._normalize(arg) for arg in args)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:
        return NotImplemented

    if shape_random is None:
        shape_random = dict()

    shape_base = na.shape_broadcasted(*args, *kwargs.values())
    shape = na.broadcast_shapes(shape_base, shape_random)

    args = tuple(arg.ndarray_aligned(shape) for arg in args)
    kwargs = {k: kwargs[k].ndarray_aligned(shape) for k in kwargs}

    unit = None
    for a in args + tuple(kwargs.values()):
        if isinstance(a, u.Quantity):
            unit = a.unit
            break

    if unit is not None:
        args = tuple(
            arg.value if isinstance(arg, u.Quantity)
            else (arg << u.dimensionless_unscaled).to_value(unit)
            for arg in args
        )
        kwargs = {
            k: kwargs[k].value if isinstance(kwargs[k], u.Quantity)
            else (kwargs[k] << u.dimensionless_unscaled).to_value(unit)
            for k in kwargs
        }

    if seed is None:
        func = getattr(np.random, func.__name__)
    else:
        func = getattr(np.random.default_rng(seed), func.__name__)

    value = func(
        *args,
        size=tuple(shape.values()),
        **kwargs,
    )

    if unit is not None:
        value = value << unit

    return na.ScalarArray(
        ndarray=value,
        axes=tuple(shape.keys()),
    )


def plt_plot_like(
        func: Callable,
        *args: na.AbstractScalarArray,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalarArray = True,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:

    try:
        args = tuple(scalars._normalize(arg) for arg in args)
        where = scalars._normalize(where)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:
        return NotImplemented

    if ax is None:
        ax = plt.gca()
    ax = na.as_named_array(ax)

    shape = na.shape_broadcasted(*args)

    if axis is None:
        if len(shape) != 1:
            raise ValueError(
                f"if `axis` is `None`, the broadcasted shape of `*args`, {shape}, should have one element"
            )
        axis = next(iter(shape))

    shape_orthogonal = {a: shape[a] for a in shape if a != axis}

    args = tuple(arg.broadcast_to(shape) for arg in args)

    if not set(ax.shape).issubset(shape_orthogonal):
        raise ValueError(
            f"the shape of `ax`, {ax.shape}, "
            f"should be a subset of the broadcasted shape of `*args` excluding `axis`, {shape_orthogonal}",
        )
    ax = ax.broadcast_to(shape_orthogonal)

    if not set(where.shape).issubset(shape_orthogonal):
        raise ValueError(
            f"the shape of `where`, {where.shape}, "
            f"should be a subset of the broadcasted shape of `*args` excluding `axis`, {shape_orthogonal}"
        )
    where = where.broadcast_to(shape_orthogonal)

    kwargs_broadcasted = dict()
    for k in kwargs:
        kwarg = kwargs[k]
        if not set(na.shape(kwarg)).issubset(shape_orthogonal):
            raise ValueError(
                f"the shape of `{k}`, {na.shape(kwarg)}, "
                f"should be a subset of the broadcasted shape of `*args` excluding `axis`, {shape_orthogonal}"
            )
        kwargs_broadcasted[k] = na.broadcast_to(kwarg, shape_orthogonal)
    kwargs = kwargs_broadcasted

    result = na.ScalarArray.empty(shape=shape_orthogonal, dtype=object)

    for index in na.ndindex(shape_orthogonal):
        if where[index]:
            func_matplotlib = getattr(ax[index].ndarray, func.__name__)
            args_index = tuple(arg[index].ndarray for arg in args)
            kwargs_index = {k: kwargs[k][index].ndarray for k in kwargs}
            result[index] = func_matplotlib(
                *args_index,
                **kwargs_index,
            )[0]

    return result


@_implements(na.optimize.root_secant)
def optimize_root_secant(
        function: Callable[[na.ScalarLike], na.ScalarLike],
        guess: na.ScalarLike,
        min_step_size: na.ScalarLike,
        max_abs_error: na.ScalarLike,
        max_iterations: int = 100,
        damping: None | float = None,
        callback: None | Callable[[int, na.ScalarLike, na.ScalarLike, na.ScalarLike], None] = None,
) -> na.ScalarArray:

    try:
        guess = scalars._normalize(guess)

        min_step_size = scalars._normalize(min_step_size)

        x0 = guess - 10 * min_step_size
        x1 = guess

        f0 = function(x0)
        f0 = scalars._normalize(f0)

        max_abs_error = scalars._normalize(max_abs_error)

    except scalars.ScalarTypeError:
        return NotImplemented

    if na.shape(max_abs_error):
        raise ValueError(f"argument `max_abs_error` should have an empty shape, got {na.shape(max_abs_error)}")

    shape = na.shape_broadcasted(f0, guess, min_step_size)

    converged = na.broadcast_to(0 * na.value(f0), shape=shape).astype(bool)

    x1 = na.broadcast_to(x1, shape).astype(float)

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

        df = f1 - f0

        if np.any(df == 0, where=active):
            raise ValueError("stationary point detected")

        dx_active = dx[active]
        f0_active = f0[active]
        f1_active = f1[active]

        df_active = f1_active - f0_active
        if np.any(df_active == 0):
            raise ValueError("stationary point detected")

        jacobian = df_active / dx_active

        correction = f1_active / jacobian
        if damping is not None:
            correction = damping * correction

        x2 = x1.copy()
        x2[active] -= correction

        x0 = x1
        x1 = x2
        f0 = f1

    raise ValueError("Max iterations exceeded")
