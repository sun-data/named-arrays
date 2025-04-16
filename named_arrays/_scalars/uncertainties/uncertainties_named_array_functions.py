import dataclasses
from typing import Callable, Sequence, Literal
import collections
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
RANDOM_FUNCTIONS = (
    na.random.uniform,
    na.random.normal,
    na.random.poisson,
    na.random.binomial,
    na.random.gamma,
)
PLT_PLOT_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS
NDFILTER_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.NDFILTER_FUNCTIONS
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


@_implements(na.nominal)
def nominal(
    a: na.AbstractUncertainScalarArray,
) -> float | u.Quantity | na.AbstractScalarArray:
    return a.nominal


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


@_implements(na.histogram)
def histogram(
    a: na.AbstractScalar,
    bins: dict[str, int] | na.AbstractScalar,
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalar = None,
    max: None | na.AbstractScalar = None,
    density: bool = False,
    weights: None | na.AbstractScalar = None,
) -> na.FunctionArray[na.AbstractScalar, na.AbstractScalar]:

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


@_implements(na.histogramdd)
def histogramdd(
    *sample: na.AbstractScalar,
    bins: dict[str, int] | na.AbstractScalar | Sequence[na.AbstractScalar],
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalar | Sequence[na.AbstractScalar] = None,
    max: None | na.AbstractScalar | Sequence[na.AbstractScalar] = None,
    density: bool = False,
    weights: None | na.AbstractScalar = None,
) -> tuple[na.AbstractScalar, tuple[na.AbstractScalar, ...]]:

    try:
        sample = [uncertainties._normalize(s) for s in sample]
        bins = [uncertainties._normalize(b) for b in bins] if not isinstance(bins, dict) else bins
        weights = uncertainties._normalize(weights) if weights is not None else weights
    except uncertainties.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape = na.shape_broadcasted(*sample, weights)

    if axis is None:
        axis = tuple(shape)
    elif isinstance(axis, str):
        axis = (axis,)

    shape_hist = {ax: shape[ax] for ax in axis}

    shape_sample = na.shape_broadcasted(*sample)
    shape_sample = na.broadcast_shapes(shape_sample, shape_hist)
    sample = [s.broadcast_to(shape_sample) for s in sample]

    if weights is not None:
        shape_weights = na.shape(weights)
        shape_weights = na.broadcast_shapes(shape_weights, shape_hist)
        weights = weights.broadcast_to(shape_weights)

    if min is None:
        min = [s.min(axis) for s in sample]
    elif not isinstance(min, collections.abc.Sequence):
        min = [min] * len(sample)

    if max is None:
        max = [s.max(axis) for s in sample]
    elif not isinstance(max, collections.abc.Sequence):
        max = [max] * len(sample)

    try:
        max = [uncertainties._normalize(m) for m in max]
        min = [uncertainties._normalize(m) for m in min]
    except uncertainties.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    hist_nominal, edges_nominal = na.histogramdd(
        *[na.as_named_array(s.nominal) for s in sample],
        bins=[b.nominal for b in bins] if not isinstance(bins, dict) else bins,
        axis=axis,
        min=[m.nominal for m in min],
        max=[m.nominal for m in max],
        density=density,
        weights=weights.nominal if weights is not None else weights,
    )

    hist_distribution, edges_distribution = na.histogramdd(
        *[na.as_named_array(s.distribution) for s in sample],
        bins=[b.distribution for b in bins] if not isinstance(bins, dict) else bins,
        axis=axis,
        min=[m.distribution for m in min],
        max=[m.distribution for m in max],
        density=density,
        weights=weights.distribution if weights is not None else weights,
    )

    hist = na.UncertainScalarArray(
        nominal=hist_nominal,
        distribution=hist_distribution,
    )

    edges = [
        na.UncertainScalarArray(e_n, e_d)
        for e_n, e_d in zip(edges_nominal, edges_distribution)
    ]

    return hist, edges


@_implements(na.convolve)
def convolve(
    array: na.AbstractScalar,
    kernel: na.AbstractScalar,
    axis: None | str | Sequence[str] = None,
    where: bool | na.AbstractScalar = True,
    mode: str = "truncate",
) -> na.UncertainScalarArray:

    try:
        array = uncertainties._normalize(array).explicit
        kernel = uncertainties._normalize(kernel).explicit
        where = uncertainties._normalize(where).explicit
    except uncertainties.ScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape_kernel = kernel.shape

    if axis is None:
        axis = tuple(shape_kernel)

    shape_kernel_parallel = {
        ax: shape_kernel[ax]
        for ax in axis
    }

    shape = na.shape_broadcasted(array, where)
    shape_parallel = {
        ax: shape[ax]
        for ax in axis
    }

    result_nominal = na.convolve(
        array=na.broadcast_to(
            array=array.nominal,
            shape=shape_parallel,
            append=True,
        ),
        kernel=na.broadcast_to(
            array=kernel.nominal,
            shape=shape_kernel_parallel,
            append=True,
        ),
        axis=axis,
        where=na.broadcast_to(
            array=where.nominal,
            shape=shape_parallel,
            append=True,
        ),
        mode=mode,
    )

    result_distribution = na.convolve(
        array=na.broadcast_to(
            array=array.distribution,
            shape=shape_parallel,
            append=True,
        ),
        kernel=na.broadcast_to(
            array=kernel.distribution,
            shape=shape_kernel_parallel,
            append=True,
        ),
        axis=axis,
        where=na.broadcast_to(
            array=where.distribution,
            shape=shape_parallel,
            append=True,
        ),
        mode=mode,
    )

    return dataclasses.replace(
        array,
        nominal=result_nominal,
        distribution=result_distribution,
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

    if seed is not None:
        seed_nominal = seed
        seed_distribution = seed + 1
    else:
        seed_nominal = seed_distribution = seed

    return na.UncertainScalarArray(
        nominal=func(
            *tuple(arg.nominal for arg in args),
            shape_random=shape_random,
            seed=seed_nominal,
            **{k: kwargs[k].nominal for k in kwargs},
        ),
        distribution=func(
            *tuple(arg.distribution for arg in args),
            shape_random=shape_random,
            seed=seed_distribution,
            **{k: kwargs[k].distribution for k in kwargs},
        )
    )


@_implements(na.random.choice)
def random_choice(
    a: na.AbstractScalarArray | na.AbstractUncertainScalarArray,
    p: None | na.AbstractScalarArray | na.AbstractUncertainScalarArray = None,
    axis: None | str | Sequence[str] = None,
    replace: bool = True,
    shape_random: None | dict[str, int] = None,
    seed: None | int = None,
) -> na.UncertainScalarArray:

    try:
        a = uncertainties._normalize(a)
        p = uncertainties._normalize(p) if p is not None else p
    except na.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    shape_ap = na.shape_broadcasted(a, p)
    a = a.broadcast_to(shape_ap)
    p = p.broadcast_to(shape_ap) if p is not None else p

    if axis is None:
        axis = tuple(shape_ap)

    if seed is not None:
        seed_nominal = seed
        seed_distribution = seed + 1
    else:
        seed_nominal = seed_distribution = seed

    return na.UncertainScalarArray(
        nominal=na.random.choice(
            a=a.nominal,
            p=p.nominal if p is not None else p,
            axis=axis,
            replace=replace,
            shape_random=shape_random,
            seed=seed_nominal,
        ),
        distribution=na.random.choice(
            a=a.distribution,
            p=p.distribution if p is not None else p,
            axis=axis,
            replace=replace,
            shape_random=shape_random,
            seed=seed_distribution,
        ),
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

    shape_args = na.shape_broadcasted(*args)

    shape = na.broadcast_shapes(na.shape(ax), shape_args)

    if axis is None:
        if len(shape_args) != 1:
            raise ValueError(
                f"if `axis` is `None`, the broadcasted shape of `*args`, "
                f"{shape_args}, should have one element"
            )
        axis = next(iter(shape_args))

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


@_implements(na.plt.stairs)
def plt_stairs(
        *args: na.AbstractScalar,
        ax: None | matplotlib.axes.Axes = None,
        axis: None | str = None,
        where: bool | na.AbstractScalarArray = True,
        **kwargs,
) -> na.UncertainScalarArray[
    npt.NDArray[matplotlib.artist.Artist],
    npt.NDArray[matplotlib.artist.Artist]
]:

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
        values = uncertainties._normalize(values)
        edges = uncertainties._normalize(edges) if edges is not None else edges
        where = uncertainties._normalize(where)
        kwargs = {k: uncertainties._normalize(kwargs[k]) for k in kwargs}
    except na.UncertainScalarTypeError:     # pragma: nocover
        return NotImplemented

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

    shape = na.broadcast_shapes(na.shape(ax), shape_args)

    values = na.broadcast_to(values, shape)

    if edges is not None:
        edges = na.broadcast_to(
            array=edges,
            shape={a: shape[a] + 1 if a == axis else shape[a] for a in shape},
        )

    axis_distribution = values.axis_distribution
    dshape_edges = na.shape(edges.distribution) if edges is not None else dict()
    shape_distribution = na.broadcast_shapes(
        na.shape(values.distribution),
        {a: dshape_edges[a] for a in dshape_edges if a != axis},
    )

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
        nominal=na.plt.stairs(
            edges.nominal if edges is not None else edges,
            values.nominal,
            ax=ax,
            axis=axis,
            where=where.nominal,
            **{k: kwargs[k].nominal for k in kwargs}
        ),
        distribution=na.plt.stairs(
            edges.distribution if edges is not None else edges,
            values.distribution,
            ax=ax,
            axis=axis,
            where=where.distribution,
            **{k: kwargs[k].distribution for k in kwargs}
        )
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
        axes = tuple(set(na.shape(spd)) | set(na.shape(wavelength)))
        if len(axes) != 1:
            raise ValueError(
                f"If `axis` is `None`, the other arguments must have zero "
                f"or one axis, got {axes}."
            )
        else:
            axis = axes[0]

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


def ndfilter(
    func: Callable,
    array: na.AbstractScalar,
    size: dict[str, int],
    where: na.AbstractScalar,
    **kwargs,
) -> na.UncertainScalarArray:

    try:
        array = uncertainties._normalize(array)
        where = uncertainties._normalize(where)
    except uncertainties.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    array = array.broadcasted
    where = where.broadcasted

    return array.type_explicit(
        nominal=func(
            array=array.nominal,
            size=size,
            where=where.nominal,
            **kwargs,
        ),
        distribution=func(
            array=array.distribution,
            size=size,
            where=where.distribution,
            **kwargs,
        )
    )


@_implements(na.despike)
def despike(
    array: na.AbstractScalar,
    axis: tuple[str, str],
    where: None | bool | na.AbstractScalar,
    inbkg: None | na.AbstractScalar,
    invar: None | float | na.AbstractScalar,
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

    try:
        array = uncertainties._normalize(array)
        where = uncertainties._normalize(where)
        inbkg = uncertainties._normalize(inbkg)
        invar = uncertainties._normalize(invar)
        psfk = uncertainties._normalize(psfk)
    except uncertainties.UncertainScalarTypeError:  # pragma: nocover
        return NotImplemented

    kwargs = dict(
        axis=axis,
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
        psfbeta=psfbeta,
        verbose=verbose,
    )

    result = array.copy_shallow()
    result.nominal = na.despike(
        array=array.nominal,
        where=where.nominal,
        inbkg=inbkg.nominal,
        invar=invar.nominal,
        psfk=psfk.nominal,
        **kwargs,
    )
    result.distribution = na.despike(
        array=array.distribution,
        where=where.distribution,
        inbkg=inbkg.distribution,
        invar=invar.distribution,
        psfk=psfk.distribution,
        **kwargs,
    )

    return result
