from typing import Callable, Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import astropy.units as u
import ndfilters
import colorsynth
import named_arrays as na
from . import scalars

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
    "RANDOM_FUNCTIONS",
    "PLT_PLOT_LIKE_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
    "jacobian",
    "colorsynth_rgb",
]

ASARRAY_LIKE_FUNCTIONS = (
    na.asarray,
    na.asanyarray,
)
RANDOM_FUNCTIONS = (
    na.random.uniform,
    na.random.normal,
    na.random.poisson,
    na.random.binomial,
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
def unit(
        a: na.AbstractScalarArray,
        unit_dimensionless: None | float | u.UnitBase = None,
        squeeze: bool = True,
) -> None | u.UnitBase:
    return na.unit(
        a=a.ndarray,
        unit_dimensionless=unit_dimensionless,
    )


@_implements(na.unit_normalized)
def unit_normalized(
        a: na.AbstractScalarArray,
        unit_dimensionless: float | u.UnitBase = u.dimensionless_unscaled,
        squeeze: bool = True,
) -> u.UnitBase:
    result = na.unit(
        a=a,
        unit_dimensionless=unit_dimensionless,
    )
    return result


@_implements(na.interp)
def interp(
        x: float | u.Quantity | na.AbstractScalarArray,
        xp:  na.AbstractScalarArray,
        fp: na.AbstractScalarArray,
        axis: None | str = None,
        left: None | float | u.Quantity | na.AbstractScalarArray = None,
        right: None | float | u.Quantity | na.AbstractScalarArray = None,
        period: None | float | u.Quantity | na.AbstractScalarArray = None,
):
    try:
        x = scalars._normalize(x)
        xp = scalars._normalize(xp)
        fp = scalars._normalize(fp)
        left = scalars._normalize(left) if left is not None else left
        right = scalars._normalize(right) if right is not None else right
        period = scalars._normalize(period) if period is not None else period
    except na.ScalarTypeError:
        return NotImplemented

    if axis is None:
        if xp.ndim != 1:
            raise ValueError("if `axis` is `None`, `xp` must have only one axis")
        axis = next(iter(xp.shape))

    shape = na.shape_broadcasted(
        # x,
        xp[{axis: 0}],
        fp[{axis: 0}],
        left,
        right,
        period,
    )

    x = na.broadcast_to(x, na.broadcast_shapes(x.shape, shape))
    xp = na.broadcast_to(xp, na.broadcast_shapes(xp.shape, shape))
    fp = na.broadcast_to(fp, na.broadcast_shapes(fp.shape, shape))
    left = na.broadcast_to(left, shape) if left is not None else left
    right = na.broadcast_to(right, shape) if right is not None else right
    period = na.broadcast_to(period, shape) if period is not None else period

    result = np.empty_like(x.value, dtype=fp.dtype)
    if fp.unit is not None:
        result = result << fp.unit

    for index in na.ndindex(shape):
        x_index = x[index]
        xp_index = xp[index]
        fp_index = fp[index]
        left_index = left[index].ndarray if left is not None else left
        right_index = right[index].ndarray if right is not None else right
        period_index = period[index].ndarray if period is not None else period

        shape_index = na.shape_broadcasted(xp_index, fp_index)

        result[index] = result.type_explicit(
            ndarray=np.interp(
                x=x_index.ndarray,
                xp=xp_index.ndarray_aligned(shape_index),
                fp=fp_index.ndarray_aligned(shape_index),
                left=left_index,
                right=right_index,
                period=period_index,
            ),
            axes=x.axes,
        )

    return result


@_implements(na.histogram2d)
def histogram2d(
    x: na.AbstractScalarArray,
    y: na.AbstractScalarArray,
    bins: dict[str, int] | na.AbstractCartesian2dVectorArray,
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalarArray | na.AbstractCartesian2dVectorArray = None,
    max: None | na.AbstractScalarArray | na.AbstractCartesian2dVectorArray = None,
    density: bool = False,
    weights: None | na.AbstractScalarArray = None,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    try:
        x = scalars._normalize(x)
        y = scalars._normalize(y)
        weights = scalars._normalize(weights) if weights is not None else weights
    except scalars.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(x, y, weights)

    x = na.broadcast_to(x, shape)
    y = na.broadcast_to(y, shape)
    weights = na.broadcast_to(weights, shape) if weights is not None else weights

    if axis is None:
        axis = tuple(shape)
    elif isinstance(axis, str):
        axis = (axis, )

    shape_orthogonal = {a: shape[a] for a in shape if a not in axis}

    if isinstance(bins, na.AbstractCartesian2dVectorArray):
        shape_bins = bins.shape
        if bins.x.ndim != 1:    # pragma: nocover
            raise ValueError(
                f"The x component of `bins` must have only one dimension, "
                f"got {bins.x.shape}."
            )
        if bins.y.ndim != 1:    # pragma: nocover
            raise ValueError(
                f"The y component of `bins` must have only one dimension, "
                f"got {bins.y.shape}."
            )
        bins = (
            bins.x.ndarray,
            bins.y.ndarray,
        )
    else:
        shape_bins = {ax: bins[ax] + 1 for ax in bins}
        bins = tuple(bins.values())

    if set(shape_bins).issubset(shape_orthogonal):  # pragma: nocover
        raise ValueError(
            f"The histogram axes, {shape_bins}, should not be a subset of "
            f"the orthogonal axes, {shape_orthogonal}."
        )

    axis_x, axis_y = shape_bins

    a = na.Cartesian2dVectorArray(x, y)

    if min is None:
        min = a.min(axis)
    elif not isinstance(min, na.AbstractCartesian2dVectorArray):
        min = na.broadcast_to(min, shape_orthogonal)
        min = na.Cartesian2dVectorArray.from_scalar(min)
    else:
        min = na.broadcast_to(min, shape_orthogonal)

    if max is None:
        max = a.max(axis)
    elif not isinstance(max, na.AbstractCartesian2dVectorArray):
        max = na.broadcast_to(max, shape_orthogonal)
        max = na.Cartesian2dVectorArray.from_scalar(max)
    else:
        max = na.broadcast_to(max, shape_orthogonal)

    shape_hist = {ax: shape_bins[ax] - 1 for ax in shape_bins}
    shape_hist = na.broadcast_shapes(shape_orthogonal, shape_hist)

    shape_x = na.broadcast_shapes(shape_orthogonal, {axis_x: shape_bins[axis_x]})
    shape_y = na.broadcast_shapes(shape_orthogonal, {axis_y: shape_bins[axis_y]})

    hist = na.ScalarArray.empty(shape_hist)
    xedges = na.ScalarArray.empty(shape_x)
    yedges = na.ScalarArray.empty(shape_y)

    unit_weights = na.unit(weights)
    unit_x = na.unit(x)
    unit_y = na.unit(y)

    hist = hist if unit_weights is None else hist << unit_weights
    xedges = xedges if unit_x is None else xedges << unit_x
    yedges = yedges if unit_y is None else yedges << unit_y

    for i in na.ndindex(shape_orthogonal):
        min_i = min[i]
        max_i = max[i]
        hist_i, xedges_i, yedges_i = np.histogram2d(
            x=x[i].ndarray_aligned(axis).reshape(-1),
            y=y[i].ndarray_aligned(axis).reshape(-1),
            bins=bins,
            range=[
                [min_i.x.ndarray, max_i.x.ndarray],
                [min_i.y.ndarray, max_i.y.ndarray],
            ],
            density=density,
            weights=weights[i].ndarray_aligned(axis).reshape(-1) if weights is not None else weights,
        )

        hist[i] = na.ScalarArray(
            ndarray=hist_i,
            axes=tuple(shape_bins),
        )
        xedges[i] = na.ScalarArray(
            ndarray=xedges_i,
            axes=axis_x
        )
        yedges[i] = na.ScalarArray(
            ndarray=yedges_i,
            axes=axis_y,
        )

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=xedges,
            y=yedges,
        ),
        outputs=hist,
    )


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
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:

    if components is not None:
        raise ValueError(f"`components` should be `None` for scalars, got {components}")

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


@_implements(na.plt.scatter)
def plt_scatter(
        *args: na.AbstractScalarArray,
        s: None | na.AbstractScalarArray = None,
        c: None | na.AbstractScalarArray = None,
        ax: None | matplotlib.axes.Axes | na.ScalarArray = None,
        where: bool | na.AbstractScalarArray = True,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.ScalarArray:

    if components is not None:
        raise ValueError(f"`components` should be `None` for scalars, got {components}")

    try:
        args = tuple(scalars._normalize(arg) for arg in args)
        s = scalars._normalize(s)
        c = scalars._normalize(c)
        where = scalars._normalize(where)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:
        return NotImplemented

    if ax is None:
        ax = plt.gca()
    ax = na.as_named_array(ax)

    shape_c = c.shape
    if "rgba" in c.shape:
        shape_c.pop("rgba")

    shape = na.shape_broadcasted(*args, s, ax, where)
    shape = na.broadcast_shapes(shape, shape_c)

    shape_orthogonal = ax.shape

    args = tuple(arg.broadcast_to(shape) for arg in args)

    if np.issubdtype(na.get_dtype(s), np.number):
        s = na.broadcast_to(s, shape)
    else:
        s = na.broadcast_to(s, shape_orthogonal)

    if np.issubdtype(na.get_dtype(c), np.number):
        if "rgba" in c.shape:
            c = na.broadcast_to(c, shape | dict(rgba=c.shape["rgba"]))
        else:
            c = na.broadcast_to(c, shape)
    else:
        c = na.broadcast_to(c, shape_orthogonal)

    where = where.broadcast_to(shape)

    args = tuple(np.where(where, arg, np.nan) for arg in args)

    kwargs_broadcasted = dict()
    for k in kwargs:
        kwarg = kwargs[k]
        if not set(na.shape(kwarg)).issubset(shape_orthogonal):
            raise ValueError(
                f"the shape of `{k}`, {na.shape(kwarg)}, "
                f"should be a subset of the shape of `ax`, {shape_orthogonal}"
            )
        kwargs_broadcasted[k] = na.broadcast_to(kwarg, shape_orthogonal)
    kwargs = kwargs_broadcasted

    result = na.ScalarArray.empty(shape=shape_orthogonal, dtype=object)

    for index in na.ndindex(shape_orthogonal):
        func_matplotlib = getattr(ax[index].ndarray, "scatter")
        args_index = tuple(arg[index].ndarray.reshape(-1) for arg in args)

        s_index = s[index].ndarray
        if s_index is not None:
            s_index = s_index.reshape(-1)

        c_index = c[index].ndarray
        if c_index is not None:
            if "rgba" in c.shape:
                c_index = c[index].ndarray.reshape(-1, c.shape["rgba"])
            else:
                c_index = c[index].ndarray.reshape(-1)

        kwargs_index = {k: kwargs[k][index].ndarray for k in kwargs}
        result[index] = func_matplotlib(
            *args_index,
            s=s_index,
            c=c_index,
            **kwargs_index,
        )

    return result


@_implements(na.plt.imshow)
def plt_imshow(
    X: na.AbstractScalarArray,
    axis_x: str,
    axis_y: str,
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractArray = None,
    cmap: None | str | matplotlib.colors.Colormap = None,
    norm: None | str | matplotlib.colors.Normalize = None,
    aspect: None | na.ArrayLike = None,
    alpha: None | na.ArrayLike = None,
    vmin: None | na.ArrayLike = None,
    vmax: None | na.ArrayLike = None,
    extent: None | na.ArrayLike = None,
    **kwargs,
) -> na.ScalarArray:
    try:
        X = scalars._normalize(X)
        aspect = scalars._normalize(aspect) if aspect is not None else aspect
        alpha = scalars._normalize(alpha) if alpha is not None else alpha
        vmin = scalars._normalize(vmin) if vmin is not None else vmin
        vmax = scalars._normalize(vmax) if vmax is not None else vmax
        extent = scalars._normalize(extent) if extent is not None else extent
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    if ax is None:
        ax = plt.gca()
    ax = na.as_named_array(ax)

    if axis_x not in X.shape:   # pragma: nocover
        raise ValueError(f"`{axis_x=}` must be a member of `{X.shape=}`")

    if axis_y not in X.shape:   # pragma: nocover
        raise ValueError(f"`{axis_y=}` must be a member of `{X.shape=}`")

    if axis_rgb is not None:    # pragma: nocover
        if axis_rgb not in X.shape:
            raise ValueError(f"`{axis_rgb=}` must be a member of `{X.shape=}`")

    shape_X = na.shape_broadcasted(X, ax)
    shape = {a: shape_X[a] for a in shape_X if a != axis_rgb}
    shape_orthogonal = ax.shape
    shape_extent = shape_orthogonal | {f"{axis_x},{axis_y}": 4}

    X = X.broadcast_to(shape_X)
    aspect = aspect.broadcast_to(shape_orthogonal) if aspect is not None else aspect
    alpha = alpha.broadcast_to(shape) if alpha is not None else alpha
    vmin = vmin.broadcast_to(shape_orthogonal) if vmin is not None else vmin
    vmax = vmax.broadcast_to(shape_orthogonal) if vmax is not None else vmax
    extent = extent.broadcast_to(shape_extent) if extent is not None else extent

    shape_index = {axis_y: shape[axis_y], axis_x: shape[axis_x]}

    if axis_rgb is not None:
        shape_X_index = shape_index | {axis_rgb: shape_X[axis_rgb]}
    else:
        shape_X_index = shape_index

    result = na.ScalarArray.empty(shape_orthogonal, dtype=object)

    for index in na.ndindex(shape_orthogonal):
        result[index] = ax[index].ndarray.imshow(
            X=X[index].ndarray_aligned(shape_X_index),
            cmap=cmap,
            norm=norm,
            aspect=aspect[index].ndarray if aspect is not None else aspect,
            alpha=alpha[index].ndarray_aligned(shape_index) if alpha is not None else alpha,
            vmin=vmin[index].ndarray if vmin is not None else vmin,
            vmax=vmax[index].ndarray if vmax is not None else vmax,
            extent=extent[index].ndarray if extent is not None else extent,
            **kwargs,
        )

    return result


@_implements(na.plt.pcolormesh)
def pcolormesh(
    *XY: na.AbstractScalarArray,
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

    if components is not None:  # pragma: nocover
        raise ValueError(f"`components` should be `None` for scalars, got {components}")

    try:
        XY = tuple(scalars._normalize(arg) for arg in XY)
        C = scalars._normalize(C)
        vmin = scalars._normalize(vmin) if vmin is not None else vmin
        vmax = scalars._normalize(vmax) if vmax is not None else vmax
    except na.ScalarTypeError:  # pragma: nocover
        pass

    if ax is None:
        ax = plt.gca()
    ax = na.as_named_array(ax)

    if axis_rgb is not None:    # pragma: nocover
        if axis_rgb not in C.shape:
            raise ValueError(f"`{axis_rgb=}` must be a member of `{C.shape=}`")

    shape_orthogonal = ax.shape

    shape_XY = na.shape_broadcasted(ax, *XY)
    shape_C = na.shape_broadcasted(ax, C)

    axes_XY = tuple(ax for ax in shape_C if ax not in shape_orthogonal)
    axes_XY = tuple(ax for ax in axes_XY if ax != axis_rgb)

    axes_C = axes_XY
    if axis_rgb is not None:
        axes_C = axes_C + (axis_rgb,)

    XY = tuple(arg.broadcast_to(shape_XY) for arg in XY)
    C = C.broadcast_to(shape_C)
    vmin = vmin.broadcast_to(shape_orthogonal) if vmin is not None else vmin
    vmax = vmax.broadcast_to(shape_orthogonal) if vmax is not None else vmax

    result = na.ScalarArray.empty(shape_orthogonal, dtype=object)

    for index in na.ndindex(shape_orthogonal):
        result[index] = ax[index].ndarray.pcolormesh(
            *[arg[index].ndarray_aligned(axes_XY) for arg in XY],
            C[index].ndarray_aligned(axes_C),
            cmap=cmap,
            norm=norm,
            vmin=vmin[index].ndarray if vmin is not None else vmin,
            vmax=vmax[index].ndarray if vmax is not None else vmax,
            **kwargs,
        )

    return result


@_implements(na.plt.text)
def plt_text(
    x: float | u.Quantity | na.AbstractScalarArray,
    y: float | u.Quantity | na.AbstractScalarArray,
    s: str | na.AbstractScalarArray,
    ax: None | matplotlib.axes.Axes | na.AbstractScalarArray = None,
    **kwargs,
) -> na.AbstractScalarArray:

    if ax is None:
        ax = plt.gca()

    try:
        x = scalars._normalize(x)
        y = scalars._normalize(y)
        s = scalars._normalize(s)
        ax = scalars._normalize(ax)
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(x, y, s, ax)

    x = x.broadcast_to(shape)
    y = y.broadcast_to(shape)
    s = s.broadcast_to(shape)
    ax = ax.broadcast_to(shape)

    result = na.ScalarArray.empty(shape, dtype=matplotlib.axes.Axes)

    for index in na.ndindex(shape):
        result[index] = ax[index].ndarray.text(
            x=x[index].ndarray,
            y=y[index].ndarray,
            s=s[index].ndarray,
            **kwargs,
        )

    return result


@_implements(na.jacobian)
def jacobian(
        function: Callable[[na.AbstractScalar], na.AbstractScalar],
        x: na.AbstractScalar,
        dx: None | na.AbstractScalar = None,
        like: None | na.AbstractScalar = None,
) -> na.AbstractScalar:

    f = function(x)

    if isinstance(f, na.AbstractScalar):
        if isinstance(x, na.AbstractScalar):
            x0 = x + dx
            f0 = function(x0)
            df = f0 - f
            return df / dx

        else:
            return NotImplemented
    else:
        return NotImplemented


@_implements(na.optimize.root_newton)
def optimize_root_newton(
        function: Callable[[na.ScalarLike], na.ScalarLike],
        guess: na.ScalarLike,
        jacobian: Callable[[na.ScalarLike], na.ScalarLike],
        max_abs_error: na.ScalarLike,
        max_iterations: int = 100,
        callback: None | Callable[[int, na.ScalarLike, na.ScalarLike, na.ScalarLike], None] = None,
) -> na.ScalarArray:

    if isinstance(guess, na.AbstractArray):
        if isinstance(guess, na.AbstractScalar):
            pass
        else:
            return NotImplemented
    else:
        guess = na.ScalarArray(guess)

    x = guess
    f = function(x)

    if isinstance(f, na.AbstractArray):
        if isinstance(f, na.AbstractScalar):
            pass
        else:
            return NotImplemented
    else:
        f = na.ScalarArray(f)

    if isinstance(max_abs_error, na.AbstractArray):
        if isinstance(max_abs_error, na.AbstractScalar):
            pass
        else:
            return NotImplemented
    else:
        max_abs_error = na.ScalarArray(max_abs_error)

    if na.shape(max_abs_error):
        raise ValueError(f"argument `max_abs_error` should have an empty shape, got {na.shape(max_abs_error)}")

    shape = na.shape_broadcasted(f, guess)

    converged = na.broadcast_to(0 * na.value(f), shape=shape).astype(bool)

    x = na.broadcast_to(x, shape).astype(float)

    for i in range(max_iterations):

        if callback is not None:
            callback(i, x, f, converged)

        converged |= np.abs(f) < max_abs_error

        if np.all(converged):
            return x

        jac = jacobian(x)

        correction = f / jac

        x = x - correction

        f = function(x)

    raise ValueError("Max iterations exceeded")


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


@_implements(na.colorsynth.rgb)
def colorsynth_rgb(
    spd: na.AbstractScalarArray,
    wavelength: None | na.AbstractScalarArray = None,
    axis: None | str = None,
    spd_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_norm: None | Callable = None,
) -> na.ScalarArray:
    try:
        spd = scalars._normalize(spd).astype(float)
        wavelength = scalars._normalize(wavelength) if wavelength is not None else wavelength
        spd_min = scalars._normalize(spd_min) if spd_min is not None else spd_min
        spd_max = scalars._normalize(spd_max) if spd_max is not None else spd_max
        wavelength_min = scalars._normalize(wavelength_min) if wavelength_min is not None else wavelength_min
        wavelength_max = scalars._normalize(wavelength_max) if wavelength_max is not None else wavelength_max
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(
        spd,
        wavelength,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

    axes = tuple(shape)
    if axis is None:
        if len(axes) != 1:
            raise ValueError(
                f"If `axis` is `None`, the broadcasted shape of the other"
                f"arguments must have exactly one axis, got {shape=}"
            )
        else:
            axis = axes[0]
    axis_ndarray = axes.index(axis)

    result_ndarray = colorsynth.rgb(
        spd=spd.ndarray_aligned(shape),
        wavelength=wavelength.ndarray_aligned(shape) if wavelength is not None else wavelength,
        axis=axis_ndarray,
        spd_min=spd_min.ndarray_aligned(shape) if spd_min is not None else spd_min,
        spd_max=spd_max.ndarray_aligned(shape) if spd_max is not None else spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min.ndarray_aligned(shape) if wavelength_min is not None else wavelength_min,
        wavelength_max=wavelength_max.ndarray_aligned(shape) if wavelength_max is not None else wavelength_max,
        wavelength_norm=wavelength_norm,
    )

    result = na.ScalarArray(
        ndarray=result_ndarray,
        axes=axes,
    )

    return result


@_implements(na.colorsynth.colorbar)
def colorsynth_colorbar(
    spd: na.AbstractScalarArray,
    wavelength: None | na.AbstractScalarArray = None,
    axis: None | str = None,
    axis_wavelength: str = "_wavelength",
    axis_intensity: str = "_intensity",
    spd_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_max: None | float | u.Quantity | na.AbstractScalarArray = None,
    wavelength_norm: None | Callable = None,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    try:
        spd = scalars._normalize(spd).astype(float)
        wavelength = scalars._normalize(wavelength) if wavelength is not None else wavelength
        spd_min = scalars._normalize(spd_min) if spd_min is not None else spd_min
        spd_max = scalars._normalize(spd_max) if spd_max is not None else spd_max
        wavelength_min = scalars._normalize(wavelength_min) if wavelength_min is not None else wavelength_min
        wavelength_max = scalars._normalize(wavelength_max) if wavelength_max is not None else wavelength_max
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(
        spd,
        wavelength,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

    axes = tuple(shape)
    if axis is None:
        if len(axes) != 1:
            raise ValueError(
                f"If `axis` is `None`, the broadcasted shape of the other"
                f"arguments must have exactly one axis, got {shape=}"
            )
        else:
            axis = axes[0]
    axis_ndarray = axes.index(axis)

    intensity, wavelength, rgb = colorsynth.colorbar(
        spd=spd.ndarray_aligned(shape),
        wavelength=wavelength.ndarray_aligned(shape) if wavelength is not None else wavelength,
        axis=axis_ndarray,
        spd_min=spd_min.ndarray_aligned(shape) if spd_min is not None else spd_min,
        spd_max=spd_max.ndarray_aligned(shape) if spd_max is not None else spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min.ndarray_aligned(shape) if wavelength_min is not None else wavelength_min,
        wavelength_max=wavelength_max.ndarray_aligned(shape) if wavelength_max is not None else wavelength_max,
        wavelength_norm=wavelength_norm,
        squeeze=False,
    )

    axes_new = (axis_wavelength, axis_intensity) + axes

    intensity = na.ScalarArray(intensity, axes_new)
    wavelength = na.ScalarArray(wavelength, axes_new)
    rgb = na.ScalarArray(rgb, axes_new)

    shape_intensity = intensity.shape
    shape_wavelength = wavelength.shape
    shape_rgb = rgb.shape

    intensity = intensity[{ax: 0 for ax in shape_intensity if shape_intensity[ax] == 1}]
    wavelength = wavelength[{ax: 0 for ax in shape_wavelength if shape_wavelength[ax] == 1}]
    rgb = rgb[{ax: 0 for ax in shape_rgb if shape_rgb[ax] == 1}]

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=intensity,
            y=wavelength,
        ),
        outputs=rgb,
    )


@_implements(na.ndfilters.mean_filter)
def mean_filter(
    array: na.AbstractScalarArray,
    size: dict[str, int],
    where: bool | na.AbstractScalarArray
) -> na.ScalarArray:

    try:
        array = scalars._normalize(array)
        where = scalars._normalize(where)
    except scalars.ScalarTypeError:   # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(array, where)
    axes = tuple(shape)
    if not set(size).issubset(axes):
        raise ValueError(
            f"the keys in {size=} must be a subset of the keys in {shape=}."
        )

    array = array.broadcast_to(shape)
    where = where.broadcast_to(shape)

    return array.type_explicit(
        ndarray=ndfilters.mean_filter(
            array=array.ndarray,
            size=tuple(size.values()),
            axis=tuple(axes.index(ax) for ax in size),
            where=where.ndarray,
        ),
        axes=axes,
    )
