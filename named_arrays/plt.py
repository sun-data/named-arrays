from __future__ import annotations
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import numpy.typing as npt
import named_arrays as na

__all__ = [
    "subplots",
    "plot",
]


def subplots(
        axis_rows: None | str = None,
        ncols: int = 1,
        axis_cols: None | str = None,
        nrows: int = 1,
        *,
        sharex: bool | str = False,
        sharey: bool | str = False,
        squeeze: bool = True,
        **kwargs,
) -> tuple[plt.Figure, na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]]]:
    """
    A thin wrapper around :func:`matplotlib.pyplot.subplots()` which allows for
    providing axis names to the rows and columns.

    Parameters
    ----------
    axis_rows
        Name of the axis representing the rows in the subplot grid.
        If :class:`None`, the ``squeeze`` argument must be :class:`True`.
    nrows
        Number of rows in the subplot grid
    axis_cols
        Name of the axis representing the columns in the subplot grid
        If :class:`None`, the ``squeeze`` argument must be :class:`True`.
    ncols
        Number of columns in the subplot grid
    sharex
        Controls whether all the :class`matplotlib.axes.Axes` instances share the same :math:`x` axis properties.
        See the documentation of :func:`matplotlib.pyplot.subplots` for more information.
    sharey
        Controls whether all the :class`matplotlib.axes.Axes` instances share the same :math:`y` axis properties.
        See the documentation of :func:`matplotlib.pyplot.subplots` for more information.
    squeeze
        If :class:`True`, :func:`numpy.squeeze` is called on the result, which removes singleton dimensions from the
        array.
        See the documentation of :func:`matplotlib.pyplot.subplots` for more information.
    kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
        The created figure and a named array of matplotlib axes.
    """

    axes = (axis_rows, axis_cols)
    axes = tuple(axis for axis in axes if axis is not None)

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        **kwargs,
    )

    return fig, na.ScalarArray(axs, axes)


def plot(
        *args: na.AbstractScalar,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalar = True,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:
    """
    A thin wrapper around :func:`matplotlib.axes.Axes.plot()` for named arrays.

    The main difference of this function from :func:`matplotlib.pyplot.plot() is the addition of the ``axis`` parameter
    indicating along which axis the lines should be connected.

    Parameters
    ----------
    args
        either `x, y` or `y`, same as `matplotlib.axes.Axes.plot`
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :class:`None`, calls :func:`matplotlib.pyploy.gca` to get the current axes.
        If an instance of :class:`ScalarArray`, ``ax.shape`` should be a subset of :func:`shape_broadcasted(*args)`
    axis
        The name of the axis that the plot lines should be connected along.
        If :class:`None`, the broadcasted shape of ``args`` should have only one element,
        otherwise a :class:`ValueError` is raised.
    where
        A boolean array that selects which elements to plot
    kwargs
        Additional keyword arguments passed to :class:`matplotlib.axes.Axes.plot()`.
        These can be instances of :class:`AbstractArray`.

    Returns
    -------
        An array of artists that were plotted
    """
    return na._named_array_function(
        plot,
        *args,
        ax=ax,
        axis=axis,
        where=where,
        **kwargs,
    )
