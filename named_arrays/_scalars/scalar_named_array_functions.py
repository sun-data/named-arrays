from typing import Callable, Sequence, Any, Literal
import collections
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import matplotlib.animation
import astropy.units as u
import astroscrappy
import ndfilters
import colorsynth
import regridding
import named_arrays as na
from . import scalars

__all__ = [
    "ASARRAY_LIKE_FUNCTIONS",
    "RANDOM_FUNCTIONS",
    "PLT_PLOT_LIKE_FUNCTIONS",
    "PLT_AXES_SETTERS",
    "PLT_AXES_GETTERS",
    "PLT_AXES_ATTRIBUTES",
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
)
PLT_PLOT_LIKE_FUNCTIONS = (
    na.plt.plot,
    na.plt.fill,
)
PLT_AXES_SETTERS = (
    na.plt.set_xlabel,
    na.plt.set_ylabel,
    na.plt.set_xlim,
    na.plt.set_ylim,
    na.plt.set_title,
    na.plt.set_xscale,
    na.plt.set_yscale,
    na.plt.set_aspect,
    na.plt.axhline,
    na.plt.axvline,
    na.plt.axvspan,
    na.plt.axhspan,
)
PLT_AXES_GETTERS = (
    na.plt.get_xlabel,
    na.plt.get_ylabel,
    na.plt.get_title,
    na.plt.get_xscale,
    na.plt.get_yscale,
    na.plt.get_aspect,
    na.plt.twinx,
    na.plt.twiny,
    na.plt.invert_xaxis,
    na.plt.invert_yaxis,
)
PLT_GET_LIM = (
    na.plt.get_xlim,
    na.plt.get_ylim,
)
PLT_AXES_ATTRIBUTES = (
    na.plt.transAxes,
    na.plt.transData,
)
NDFILTER_FUNCTIONS = (
    na.ndfilters.mean_filter,
    na.ndfilters.trimmed_mean_filter,
    na.ndfilters.variance_filter,
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
            axes=x_index.axes,
        )

    return result


@_implements(na.histogram)
def histogram(
    a: na.AbstractScalarArray,
    bins: dict[str, int] | na.AbstractScalarArray,
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalarArray = None,
    max: None | na.AbstractScalarArray = None,
    density: bool = False,
    weights: None | na.AbstractScalarArray = None,
) -> na.FunctionArray[na.AbstractScalarArray, na.ScalarArray]:

    if isinstance(bins, na.AbstractArray):
        bins = (bins, )

    hist, edges = na.histogramdd(
        a,
        bins=bins,
        axis=axis,
        min=min,
        max=max,
        density=density,
        weights=weights,
    )
    edges = edges[0]

    return na.FunctionArray(
        inputs=edges,
        outputs=hist,
    )


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

        axis_x = set(bins.x.shape) - set(shape_orthogonal)
        if len(axis_x) != 1:  # pragma: nocover
            raise ValueError(
                f"if `bins` is a vector, `bins.x` must have exactly one new axis, "
                f"got {axis_x}"
            )

        axis_y = set(bins.y.shape) - set(shape_orthogonal)
        if len(axis_y) != 1:  # pragma: nocover
            raise ValueError(
                f"if `bins` is a vector, `bins.y` must have exactly one new axis, "
                f"got {axis_y}"
            )

        if axis_x == axis_y:  # pragma: nocover
            raise ValueError(
                f"if `bins` is a vector, `bins.x` and `bins.y` must be separable "
                f"along the new axes, found non-separable axis {axis_x}."
            )

        edges = bins

    elif isinstance(bins, dict):

        a = na.Cartesian2dVectorArray(x, y)

        if min is None:
            min = a.min(axis)
        elif not isinstance(min, na.AbstractCartesian2dVectorArray):
            min = na.Cartesian2dVectorArray(min, min)

        if max is None:
            max = a.max(axis)
        elif not isinstance(max, na.AbstractCartesian2dVectorArray):
            max = na.Cartesian2dVectorArray(max, max)

        if len(bins) != 2:  # pragma: nocover
            raise ValueError(
                f"if `bins` is a dictionary, it must have exactly two keys, "
                f"got {bins=}."
            )

        axis_x, axis_y = tuple(bins)

        edges = na.Cartesian2dVectorLinearSpace(
            start=min,
            stop=max,
            axis=na.Cartesian2dVectorArray(
                x=axis_x,
                y=axis_y,
            ),
            num=na.Cartesian2dVectorArray(
                x=bins[axis_x] + 1,
                y=bins[axis_y] + 1,
            ),
        )

    else:  # pragma: nocover
        return NotImplemented

    shape_edges_x = na.broadcast_shapes(shape_orthogonal, edges.x.shape)
    shape_edges_y = na.broadcast_shapes(shape_orthogonal, edges.y.shape)

    edges_broadcasted = edges.explicit.replace(
        x=edges.x.broadcast_to(shape_edges_x),
        y=edges.y.broadcast_to(shape_edges_y),
    )

    shape_edges = edges.shape
    shape_hist = {
        ax: shape_edges[ax] - 1
        for ax in shape_edges
        if ax not in shape_orthogonal
    }
    shape_hist = na.broadcast_shapes(shape_orthogonal, shape_hist)

    hist = na.ScalarArray.empty(shape_hist)

    unit_weights = na.unit(weights)
    hist = hist if unit_weights is None else hist << unit_weights

    for i in na.ndindex(shape_orthogonal):
        edges_x_i = edges_broadcasted.x[i]
        edges_y_i = edges_broadcasted.y[i]
        hist_i, _, _ = np.histogram2d(
            x=x[i].ndarray_aligned(axis).reshape(-1),
            y=y[i].ndarray_aligned(axis).reshape(-1),
            bins=(
                edges_x_i.ndarray,
                edges_y_i.ndarray,
            ),
            density=density,
            weights=weights[i].ndarray_aligned(axis).reshape(-1) if weights is not None else weights,
        )

        hist[i] = na.ScalarArray(
            ndarray=hist_i,
            axes=edges_x_i.axes + edges_y_i.axes,
        )

    return na.FunctionArray(
        inputs=edges,
        outputs=hist,
    )


@_implements(na.histogramdd)
def histogramdd(
    *sample: na.AbstractScalarArray,
    bins: dict[str, int] | na.AbstractScalarArray | Sequence[na.AbstractScalarArray],
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalarArray | Sequence[na.AbstractScalarArray] = None,
    max: None | na.AbstractScalarArray | Sequence[na.AbstractScalarArray] = None,
    density: bool = False,
    weights: None | na.AbstractScalarArray = None,
) -> tuple[na.AbstractScalarArray, tuple[na.AbstractScalarArray, ...]]:

    try:
        sample = [scalars._normalize(s) for s in sample]
        weights = scalars._normalize(weights) if weights is not None else weights
    except scalars.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(*sample, weights)

    sample = [s.broadcast_to(shape) for s in sample]
    weights = na.broadcast_to(weights, shape) if weights is not None else weights

    if axis is None:
        axis = tuple(shape)
    elif isinstance(axis, str):
        axis = (axis,)

    if isinstance(bins, dict):

        if min is None:
            min = [s.min(axis) for s in sample]
        elif not isinstance(min, collections.abc.Sequence):
            min = [min] * len(sample)

        if max is None:
            max = [s.max(axis) for s in sample]
        elif not isinstance(max, collections.abc.Sequence):
            max = [max] * len(sample)

        try:
            max = [scalars._normalize(m) for m in max]
            min = [scalars._normalize(m) for m in min]
        except scalars.ScalarTypeError:  # pragma: nocover
            return NotImplemented

        if len(bins) != len(sample):  # pragma: nocover
            raise ValueError(
                f"if {bins=} is a dictionary, it must have the same number of"
                f"elements as {len(sample)=}."
            )

        bins = [
            na.linspace(start, stop, axis=ax, num=bins[ax])
            for start, stop, ax in zip(min, max, bins)
        ]

    try:
        bins = [scalars._normalize(b) for b in bins]
    except scalars.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape_orthogonal = {a: shape[a] for a in shape if a not in axis}

    bins_broadcasted = [
        b.broadcast_to(na.broadcast_shapes(shape_orthogonal, b.shape))
        for b in bins
    ]

    shape_bins = na.shape_broadcasted(*bins)
    shape_hist = {
        ax: shape_bins[ax] - 1
        for ax in shape_bins
        if ax not in shape_orthogonal
    }
    shape_hist = na.broadcast_shapes(shape_orthogonal, shape_hist)

    hist = na.ScalarArray.empty(shape_hist)

    unit_weights = na.unit(weights)
    hist = hist if unit_weights is None else hist << unit_weights

    for i in na.ndindex(shape_orthogonal):
        hist_i, _ = np.histogramdd(
            sample=[s[i].ndarray_aligned(axis).reshape(-1) for s in sample],
            bins=[b[i].ndarray for b in bins_broadcasted],
            density=density,
            weights=weights[i].ndarray_aligned(axis).reshape(-1) if weights is not None else weights,
        )

        hist[i] = na.ScalarArray(
            ndarray=hist_i,
            axes=sum((b[i].axes for b in bins_broadcasted), ()),
        )

    return hist, tuple(bins)


@_implements(na.convolve)
def convolve(
    array: na.AbstractScalarArray,
    kernel: na.AbstractScalarArray,
    axis: None | str | Sequence[str] = None,
    where: bool | na.AbstractScalarArray = True,
    mode: str = "truncate",
) -> na.ScalarArray:

    try:
        array = scalars._normalize(array).explicit
        kernel = scalars._normalize(kernel).explicit
        where = scalars._normalize(where).explicit
    except scalars.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape_array = array.shape
    shape_kernel = kernel.shape
    shape_where = na.shape(where)

    if axis is None:
        axis = tuple(shape_kernel)
    elif isinstance(axis, str):
        axis = (axis,)

    if not set(axis).issubset(shape_array):  # pragma: nocover
        raise ValueError(
            f"{axis=} must be a subset of {array.axes=}"
        )

    if not set(axis).issubset(shape_kernel):  # pragma: nocover
        raise ValueError(
            f"{axis=} must be a subset of {kernel.axes=}"
        )

    shape_kernel_ortho = {
        ax: shape_kernel[ax]
        for ax in shape_kernel if ax not in axis
    }

    shape = na.broadcast_shapes(
        shape_array,
        shape_kernel_ortho,
        shape_where,
    )

    axes = tuple(shape)

    result = ndfilters.convolve(
        array=array.ndarray_aligned(axes),
        kernel=kernel.ndarray_aligned(axes),
        axis=[axes.index(ax) for ax in axis],
        where=where.ndarray_aligned(axes),
        mode=mode,
    )

    result = array.replace(
        ndarray=result,
        axes=axes,
    )

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
            arg.to_value(unit) if isinstance(arg, u.Quantity)
            else (arg << u.dimensionless_unscaled).to_value(unit)
            for arg in args
        )
        kwargs = {
            k: kwargs[k].to_value(unit) if isinstance(kwargs[k], u.Quantity)
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


@_implements(na.random.binomial)
def random_binomial(
    n: int | u.Quantity | na.AbstractScalarArray,
    p: float | na.AbstractScalarArray,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
):
    try:
        n = scalars._normalize(n)
        p = scalars._normalize(p)
    except na.ScalarTypeError:
        return NotImplemented

    if shape_random is None:
        shape_random = dict()

    shape_base = na.shape_broadcasted(n, p)
    shape = na.broadcast_shapes(shape_base, shape_random)

    n = n.ndarray_aligned(shape)
    p = p.ndarray_aligned(shape)

    unit = na.unit(n)

    if unit is not None:
        n = n.value

    if seed is None:
        func = np.random.binomial
    else:
        func = np.random.default_rng(seed).binomial

    value = func(
        n=n,
        p=p,
        size=tuple(shape.values()),
    )

    if unit is not None:
        value = value << unit

    return na.ScalarArray(
        ndarray=value,
        axes=tuple(shape.keys()),
    )


@_implements(na.random.gamma)
def random_gamma(
    shape: float | na.AbstractScalarArray,
    scale: float | u.Quantity | na.AbstractScalarArray = 1,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
) -> na.ScalarArray:
    alpha = shape
    theta = scale

    try:
        alpha = scalars._normalize(alpha)
        theta = scalars._normalize(theta)
    except na.ScalarTypeError:
        return NotImplemented

    if shape_random is None:
        shape_random = dict()

    shape_base = na.shape_broadcasted(alpha, theta)
    shape = na.broadcast_shapes(shape_base, shape_random)

    alpha = alpha.ndarray_aligned(shape)
    theta = theta.ndarray_aligned(shape)

    unit = na.unit(theta)

    if unit is not None:
        theta = theta.value

    if seed is None:
        func = np.random.gamma
    else:
        func = np.random.default_rng(seed).gamma

    value = func(
        shape=alpha,
        scale=theta,
        size=tuple(shape.values()),
    )

    if unit is not None:
        value = value << unit

    return na.ScalarArray(
        ndarray=value,
        axes=tuple(shape.keys()),
    )


@_implements(na.random.choice)
def random_choice(
    a: na.AbstractScalarArray,
    p: None | na.AbstractScalarArray = None,
    axis: None | str | Sequence[str] = None,
    replace: bool = True,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
) -> na.ScalarArray:

    try:
        a = scalars._normalize(a)
        p = scalars._normalize(p) if p is not None else p
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    unit = na.unit(a)
    if unit is not None:
        a = a.value

    if shape_random is None:
        shape_random = dict()

    if seed is None:
        func = np.random.choice
    else:
        func = np.random.default_rng(seed).choice

    shape_ap = na.shape_broadcasted(a, p)
    a = a.broadcast_to(shape_ap)
    p = p.broadcast_to(shape_ap) if p is not None else p

    if axis is None:
        axis = tuple(shape_ap)
    elif isinstance(axis, str):
        axis = (axis, )

    shape_ap_orthogonal = {ax: shape_ap[ax] for ax in shape_ap if ax not in axis}

    shape_result = na.broadcast_shapes(shape_random, shape_ap_orthogonal)

    result = na.ScalarArray.empty(shape_result)

    shape_i = {ax: shape_random[ax] for ax in shape_random if ax not in shape_ap_orthogonal}

    for i in na.ndindex(shape_ap, axis_ignored=axis):

        if p is not None:
            p_i = p[i].ndarray.reshape(-1)
            p_i = p_i / p_i.sum()
        else:
            p_i = None

        result_i = func(
            a=a[i].ndarray.reshape(-1),
            size=tuple(shape_i.values()),
            replace=replace,
            p=p_i,
        )
        result_i = na.ScalarArray(
            ndarray=result_i,
            axes=tuple(shape_i),
        )
        result[i] = result_i

    if unit is not None:
        result = result << unit

    return result


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

    shape_args = na.shape_broadcasted(*args)

    shape = na.broadcast_shapes(ax.shape, shape_args)

    if axis is None:
        if len(shape_args) != 1:
            raise ValueError(
                f"if `axis` is `None`, the broadcasted shape of `*args`, "
                f"{shape_args}, should have one element"
            )
        axis = next(iter(shape_args))

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


@_implements(na.plt.stairs)
def plt_stairs(
        *args: na.AbstractScalarArray,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalarArray = True,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:

    if len(args) == 1:
        edges = None
        values, = args
    elif len(args) == 2:
        edges, values = args
    else:   # pragma: nocover
        raise ValueError(
            f"incorrect number of arguments, expected 1 or 2, got {len(args)}"
        )

    try:
        values = scalars._normalize(values)
        edges = scalars._normalize(edges) if edges is not None else edges
        where = scalars._normalize(where)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    if ax is None:
        ax = plt.gca()
    ax = na.as_named_array(ax)

    if axis is None:
        if len(values.shape) != 1:
            raise ValueError(
                f"if {axis=}, {values.shape=} should have only one element."
            )
        axis = next(iter(values.shape))
    else:
        if axis not in values.shape:
            raise ValueError(
                f"{axis=} must be an element of {values.shape}"
            )

    shape_values = na.shape(values)
    shape_edges = na.shape(edges)
    shape_args = na.broadcast_shapes(
        shape_values,
        {a: shape_edges[a] for a in shape_edges if a != axis},
    )

    shape = na.broadcast_shapes(ax.shape, shape_args)

    shape_orthogonal = {a: shape[a] for a in shape if a != axis}

    values = na.broadcast_to(values, shape)

    if edges is not None:
        edges = na.broadcast_to(
            array=edges,
            shape=shape_orthogonal | {axis: shape[axis] + 1},
        )

    if not set(ax.shape).issubset(shape_orthogonal):    # pragma: nocover
        raise ValueError(
            f"the shape of `ax`, {ax.shape}, "
            f"should be a subset of the broadcasted shape of `*args` excluding `axis`, {shape_orthogonal}",
        )
    ax = ax.broadcast_to(shape_orthogonal)

    if not set(where.shape).issubset(shape_orthogonal):     # pragma: nocover
        raise ValueError(
            f"the shape of `where`, {where.shape}, "
            f"should be a subset of the broadcasted shape of `*args` excluding `axis`, {shape_orthogonal}"
        )
    where = where.broadcast_to(shape_orthogonal)

    kwargs_broadcasted = dict()
    for k in kwargs:
        kwarg = kwargs[k]
        if not set(na.shape(kwarg)).issubset(shape_orthogonal):     # pragma: nocover
            raise ValueError(
                f"the shape of `{k}`, {na.shape(kwarg)}, "
                f"should be a subset of the broadcasted shape of `*args` excluding `axis`, {shape_orthogonal}"
            )
        kwargs_broadcasted[k] = na.broadcast_to(kwarg, shape_orthogonal)
    kwargs = kwargs_broadcasted

    result = na.ScalarArray.empty(shape=shape_orthogonal, dtype=object)

    for index in na.ndindex(shape_orthogonal):
        if where[index]:
            values_index = values[index].ndarray
            edges_index = edges[index].ndarray if edges is not None else edges
            kwargs_index = {k: kwargs[k][index].ndarray for k in kwargs}
            result[index] = ax[index].ndarray.stairs(
                values=values_index,
                edges=edges_index,
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
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractScalarArray = None,
    components: None | tuple[str, str] = None,
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


@_implements(na.plt.pcolormovie)
def pcolormovie(
    *TXY: na.AbstractScalarArray,
    C: na.AbstractScalarArray,
    axis_time: str,
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractScalarArray = None,
    components: None | tuple[str, str] = None,
    cmap: None | str | matplotlib.colors.Colormap = None,
    norm: None | str | matplotlib.colors.Normalize = None,
    vmin: None | float | u.Quantity | na.AbstractScalarArray = None,
    vmax: None | float | u.Quantity | na.AbstractScalarArray = None,
    kwargs_pcolormesh: None | dict[str, Any] = None,
    kwargs_animation: None | dict[str, Any] = None,
) -> matplotlib.animation.FuncAnimation:

    t, x, y = TXY

    if ax is None:
        ax = plt.gca()
    ax = na.asanyarray(ax)

    ax0 = ax.ndarray.flat[0]
    fig = ax0.figure

    try:
        t = scalars._normalize(t)
        x = scalars._normalize(x)
        y = scalars._normalize(y)
        C = scalars._normalize(C)
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    time_is_atleast_2d = (t.ndim > 1)

    shape = na.shape_broadcasted(t, x, y)

    if time_is_atleast_2d:
        shape_time = na.shape_broadcasted(t, ax)
        t = t.broadcast_to(na.shape_broadcasted(t, ax))
    else:
        shape_time = {axis_time: shape[axis_time]}

    t = t.broadcast_to(shape_time)
    x = x.broadcast_to(shape)
    y = y.broadcast_to(shape)

    if kwargs_pcolormesh is None:
        kwargs_pcolormesh = dict()
    if kwargs_animation is None:
        kwargs_animation = dict()

    def func(frame: int):
        index_frame = {axis_time: frame}
        for i in ax.ndindex():
            ax_i = ax[i].ndarray
            for artist in ax_i.collections:
                artist.remove()
            ax_i.relim()
            if time_is_atleast_2d:
                ax_i.set_title(t[index_frame | i].ndarray)
            else:
                fig.suptitle(
                    t=t[index_frame].ndarray,
                    x=.99,
                    y=.01,
                    ha="right",
                    va="bottom",
                    fontsize="medium",
                )

        na.plt.pcolormesh(
            x[index_frame],
            y[index_frame],
            C=C[index_frame],
            axis_rgb=axis_rgb,
            ax=ax,
            components=components,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            **kwargs_pcolormesh,
        )

    result = matplotlib.animation.FuncAnimation(
        fig=fig,
        func=func,
        frames=shape[axis_time],
        **kwargs_animation,
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
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(x, y, s, ax)

    x = x.broadcast_to(shape)
    y = y.broadcast_to(shape)
    s = s.broadcast_to(shape)
    ax = ax.broadcast_to(shape)
    kwargs = {k: kwargs[k].broadcast_to(shape) for k in kwargs}

    result = na.ScalarArray.empty(shape, dtype=matplotlib.axes.Axes)

    for index in na.ndindex(shape):
        kwargs_index = {k: kwargs[k][index].ndarray for k in kwargs}
        result[index] = ax[index].ndarray.text(
            x=x[index].ndarray,
            y=y[index].ndarray,
            s=s[index].ndarray,
            **kwargs_index,
        )

    return result


def plt_axes_setter(
    method: str,
    *args,
    ax: None | matplotlib.axes.Axes | na.AbstractScalarArray = None,
    **kwargs,
) -> na.ScalarArray:

    if ax is None:
        ax = plt.gca()

    try:
        args = [scalars._normalize(arg) for arg in args]
        ax = scalars._normalize(ax)
        kwargs = {k: scalars._normalize(kwargs[k]) for k in kwargs}
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = ax.shape

    args = [arg.broadcast_to(shape) for arg in args]
    kwargs = {k: kwargs[k].broadcast_to(shape) for k in kwargs}

    result = ax.type_explicit.empty(shape=ax.shape, dtype=object)

    for index in na.ndindex(shape):
        args_index = [arg[index].ndarray for arg in args]
        kwargs_index = {k: kwargs[k][index].ndarray for k in kwargs}
        r = getattr(ax[index].ndarray, method.__name__)(*args_index, **kwargs_index)
        result[index] = r

    return result


def plt_axes_getter(
    method: str,
    ax: na.AbstractScalarArray,
) -> na.ScalarArray:

    try:
        ax = scalars._normalize(ax)
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    result = na.ScalarArray.empty(shape=ax.shape, dtype=object)

    for index in na.ndindex(ax.shape):
        ax_index = ax[index].ndarray
        if ax_index is None:
            ax_index = plt.gca()
        result[index] = getattr(ax_index, method.__name__)()

    return result


def plt_get_lim(
    method: str,
    ax: na.AbstractScalarArray,
) -> tuple[na.ScalarArray, na.ScalarArray]:

    try:
        ax = scalars._normalize(ax)
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    result_lower = na.ScalarArray.empty(shape=ax.shape, dtype=object)
    result_upper = na.ScalarArray.empty(shape=ax.shape, dtype=object)

    for index in na.ndindex(ax.shape):
        ax_index = ax[index].ndarray
        if ax_index is None:
            ax_index = plt.gca()
        result_lower[index], result_upper[index] = getattr(ax_index, method.__name__)()

    return result_lower, result_upper


def plt_axes_attribute(
    method: str,
    ax: na.AbstractScalarArray,
) -> na.ScalarArray:

    try:
        ax = scalars._normalize(ax)
    except na.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    result = na.ScalarArray.empty(shape=ax.shape, dtype=object)

    for index in na.ndindex(ax.shape):
        ax_index = ax[index].ndarray
        if ax_index is None:
            ax_index = plt.gca()
        result[index] = getattr(ax_index, method.__name__)

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

        converged = np.abs(f) < max_abs_error

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

    if axis is None:
        axes = tuple(set(na.shape(spd)) | set(na.shape(wavelength)))
        if len(axes) != 1:
            raise ValueError(
                f"If `axis` is `None`, the other arguments must have zero"
                f"or one axis, got {axes}."
            )
        axis = axes[0]

    if wavelength is not None:
        if axis in spd.shape:
            if wavelength.shape[axis] == spd.shape[axis] + 1:
                below = {axis: slice(None, ~0)}
                above = {axis: slice(+1, None)}
                wavelength = (wavelength[below] + wavelength[above]) / 2

    shape = na.shape_broadcasted(
        spd,
        wavelength,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

    axes = tuple(shape)
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

    if axis is None:
        axes = tuple(set(na.shape(spd)) | set(na.shape(wavelength)))
        if len(axes) != 1:
            raise ValueError(
                f"If `axis` is `None`, the other arguments must have zero "
                f"or one axis, got {axes}."
            )
        axis = axes[0]

    if wavelength is not None:
        if axis in spd.shape:
            if wavelength.shape[axis] == spd.shape[axis] + 1:
                below = {axis: slice(None, ~0)}
                above = {axis: slice(+1, None)}
                wavelength = (wavelength[below] + wavelength[above]) / 2

    shape = na.shape_broadcasted(
        spd,
        wavelength,
        spd_min,
        spd_max,
        wavelength_min,
        wavelength_max,
    )

    axes = tuple(shape)
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


def ndfilter(
    func: Callable,
    array: na.AbstractScalarArray,
    size: dict[str, int],
    where: bool | na.AbstractScalarArray,
    **kwargs,
) -> na.ScalarArray:

    func = getattr(ndfilters, func.__name__)

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
        ndarray=func(
            array=array.ndarray,
            size=tuple(size.values()),
            axis=tuple(axes.index(ax) for ax in size),
            where=where.ndarray,
            **kwargs,
        ),
        axes=axes,
    )


@_implements(na.regridding.weights)
def regridding_weights(
    coordinates_input: na.AbstractScalarArray | na.AbstractVectorArray,
    coordinates_output: na.AbstractScalarArray | na.AbstractVectorArray,
    axis_input: None | str | Sequence[str] = None,
    axis_output: None | str | Sequence[str] = None,
    method: Literal['multilinear', 'conservative'] = 'multilinear',
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:

    if not isinstance(coordinates_output, na.AbstractVectorArray):
        coordinates_output = na.CartesianNdVectorArray(dict(x=coordinates_output))

    if not isinstance(coordinates_input, na.AbstractVectorArray):
        coordinates_input = na.CartesianNdVectorArray(dict(x=coordinates_input))

    return na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
    )


@_implements(na.regridding.regrid_from_weights)
def regridding_regrid_from_weights(
    weights: na.AbstractScalarArray,
    shape_input: dict[str, int],
    shape_output: dict[str, int],
    values_input: na.AbstractScalarArray,
) -> na.ScalarArray:

    try:
        weights = scalars._normalize(weights)
        values_input = scalars._normalize(values_input)
    except scalars.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape_weights = weights.shape

    axis_input = tuple(a for a in shape_input if a not in shape_weights)
    axis_output = tuple(a for a in shape_output if a not in shape_weights)

    shape_values_input = values_input.shape
    shape_orthogonal = {
        a: shape_values_input[a]
        for a in shape_values_input
        if a not in axis_input
    }
    shape_orthogonal = na.broadcast_shapes(shape_orthogonal, shape_weights)

    shape_input = na.broadcast_shapes(shape_orthogonal, shape_input)
    shape_output = na.broadcast_shapes(shape_orthogonal, shape_output)

    weights = weights.broadcast_to({
        a: shape_input[a] if a not in axis_input else 1
        for a in shape_input
    })
    values_input = values_input.broadcast_to(shape_input)

    result = regridding.regrid_from_weights(
        weights=weights.ndarray,
        shape_input=tuple(shape_input.values()),
        shape_output=tuple(shape_output.values()),
        values_input=values_input.ndarray,
        axis_input=tuple(tuple(shape_input).index(a) for a in axis_input),
        axis_output=tuple(tuple(shape_output).index(a) for a in axis_output),
    )

    result = na.ScalarArray(
        ndarray=result,
        axes=tuple(shape_output),
    )

    return result

@_implements(na.regridding.transpose_weights)
def regridding_transpose_weights(
        weights: na.AbstractScalar,
        shape_input: dict[str, int],
        shape_output: dict[str, int],
) -> na.AbstractScalar:

    new_weights, _, _, = regridding.transpose_weights((weights.ndarray, tuple(), tuple()))

    return (na.ScalarArray(new_weights, axes=weights.axes), shape_output, shape_input)

@_implements(na.despike)
def despike(
    array: na.AbstractScalarArray,
    axis: tuple[str, str],
    where: None | bool | na.AbstractScalarArray,
    inbkg: None | na.AbstractScalarArray,
    invar: None | float | na.AbstractScalarArray,
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
    psfk: None | na.AbstractScalarArray,
    psfbeta: float,
    verbose: bool,
) -> na.ScalarArray:

    try:
        array = scalars._normalize(array)
        where = scalars._normalize(where) if where is not None else where
        inbkg = scalars._normalize(inbkg) if inbkg is not None else inbkg
        invar = scalars._normalize(invar) if invar is not None else invar
        psfk = scalars._normalize(psfk) if psfk is not None else psfk
    except scalars.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(
        array,
        where,
        inbkg,
        invar,
    )

    array = array.broadcast_to(shape)
    where = where.broadcast_to(shape) if where is not None else where
    inbkg = inbkg.broadcast_to(shape) if inbkg is not None else inbkg
    invar = invar.broadcast_to(shape) if invar is not None else invar

    if psfk is not None:
        shape_orthogonal = {ax: shape[ax] for ax in shape if ax not in axis}
        shape_psfk = na.broadcast_shapes(shape_orthogonal, psfk.shape)
        psfk = na.broadcast_to(psfk, shape_psfk)

    result = array.copy()
    inmask = ~where if where is not None else where

    for index in na.ndindex(shape, axis_ignored=axis):
        result_ndarray = astroscrappy.detect_cosmics(
            indat=array[index].ndarray,
            inmask=inmask[index].ndarray if inmask is not None else inmask,
            inbkg=inbkg[index].ndarray if inbkg is not None else inbkg,
            invar=invar[index].ndarray if invar is not None else invar,
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
            psfk=psfk[index].ndarray if psfk is not None else psfk,
            psfbeta=psfbeta,
            verbose=verbose,
        )

        result[index].value.ndarray[:] = result_ndarray[1]

    return result
