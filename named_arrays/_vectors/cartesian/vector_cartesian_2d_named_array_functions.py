from  typing import Callable
import numpy.typing as npt
import matplotlib.axes
import named_arrays as na
from named_arrays._scalars import scalar_named_array_functions

__all__ = [
    "PLT_PLOT_LIKE_FUNCTIONS"
]

PLT_PLOT_LIKE_FUNCTIONS = scalar_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS


def plt_plot_like(
        func: Callable,
        *args: na.AbstractCartesian2dVectorArray,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalarArray = True,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:

    if len(args) != 1:
        return NotImplemented

    a, = args

    if not isinstance(a, na.AbstractCartesian2dVectorArray):
        return NotImplemented

    return func(
        na.as_named_array(a.x),
        na.as_named_array(a.y),
        ax=ax,
        axis=axis,
        where=where,
        **kwargs,
    )
