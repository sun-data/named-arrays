from typing import Callable
import numpy.typing as npt
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import named_arrays._scalars.scalar_named_array_functions
from . import uncertainties

__all__ = [
    "RANDOM_FUNCTIONS",
    "PLT_PLOT_LIKE_FUNCTIONS",
    "HANDLED_FUNCTIONS",
    "random",
]

RANDOM_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.RANDOM_FUNCTIONS
PLT_PLOT_LIKE_FUNCTIONS = named_arrays._scalars.scalar_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS
HANDLED_FUNCTIONS = dict()


def _implements(function: Callable):
    """Register a __named_array_function__ implementation for AbstractScalarArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[function] = func
        return func
    return decorator


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
        **kwargs,
) -> na.UncertainScalarArray[
    npt.NDArray[matplotlib.artist.Artist],
    npt.NDArray[matplotlib.artist.Artist]
]:
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

    alpha = max(1 / args[~0].num_distribution, 1/255)
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
