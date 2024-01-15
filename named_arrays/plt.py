from __future__ import annotations
from typing import Literal
import matplotlib.axes
import matplotlib.figure
import matplotlib.artist
import matplotlib.pyplot as plt
import numpy.typing as npt
import named_arrays as na

__all__ = [
    "subplots",
    "plot",
    "fill",
    "scatter",
]


def subplots(
        axis_rows: str = "subplots_row",
        ncols: int = 1,
        axis_cols: str = "subplots_col",
        nrows: int = 1,
        *,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        **kwargs,
) -> tuple[
    matplotlib.figure.Figure,
    matplotlib.axes.Axes | na.ScalarArray[npt.NDArray],
]:
    """
    A thin wrapper around :func:`matplotlib.pyplot.subplots()` which allows for
    providing axis names to the rows and columns.

    Parameters
    ----------
    axis_rows
        Name of the axis representing the rows in the subplot grid.
        If :obj:`None`, the ``squeeze`` argument must be :obj:`True`.
    nrows
        Number of rows in the subplot grid
    axis_cols
        Name of the axis representing the columns in the subplot grid
        If :obj:`None`, the ``squeeze`` argument must be :obj:`True`.
    ncols
        Number of columns in the subplot grid
    sharex
        Controls whether all the :class:`matplotlib.axes.Axes` instances share the same horizontal axis properties.
        See the documentation of :func:`matplotlib.pyplot.subplots` for more information.
    sharey
        Controls whether all the :class`matplotlib.axes.Axes` instances share the same vertical axis properties.
        See the documentation of :func:`matplotlib.pyplot.subplots` for more information.
    squeeze
        If :obj:`True`, :func:`numpy.squeeze` is called on the result, which removes singleton dimensions from the
        array.
        See the documentation of :func:`matplotlib.pyplot.subplots` for more information.
    kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`
    """

    shape = {axis_rows: nrows, axis_cols: ncols}

    if squeeze:
        shape = {axis: shape[axis] for axis in shape if shape[axis] != 1}

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        **kwargs,
    )

    return fig, na.ScalarArray(axs, axes=tuple(shape.keys()))


def plot(
        *args: na.AbstractScalar,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalar = True,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.plot` for named arrays.

    The main difference of this function from :func:`matplotlib.pyplot.plot` is the addition of the ``axis`` parameter
    indicating along which axis the lines should be connected.

    Parameters
    ----------
    args
        Same signature as :meth:`matplotlib.axes.Axes.plot`.
        If ``ax`` is a 2D plot, ``*args`` should be ``y`` or ``x, y``.
        If ``ax`` is a 3D plot, ``*args`` should be ``x, y`` or ``x, y, z``.
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :obj:`None`, calls :func:`matplotlib.pyplot.gca` to get the current axes.
        If an instance of :class:`named_arrays.ScalarArray`, ``ax.shape`` should be a subset of the broadcasted shape of
        ``*args``.
    axis
        The name of the axis that the plot lines should be connected along.
        If :obj:`None`, the broadcasted shape of ``args`` should have only one element,
        otherwise a :class:`ValueError` is raised.
    where
        A boolean array that selects which elements to plot
    transformation
        A callable that is applied to args before plotting
    components
        The component names of ``*args`` to plot, helpful if ``*args`` are an instance of
        :class:`named_arrays.AbstractVectorArray`.
    kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
        These can be instances of :class:`named_arrays.AbstractArray`.

    Returns
    -------
        An array of artists that were plotted

    Examples
    --------

    Plot a single scalar

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        x = na.linspace(0, 2 * np.pi, axis="x",  num=101)
        y = np.sin(x)

        plt.figure();
        na.plt.plot(x, y);

    Plot an array of scalars

    .. jupyter-execute::

        z = na.linspace(0, np.pi, axis="z", num=5)

        y = np.sin(x - z)

        plt.figure();
        na.plt.plot(x, y, axis="x");

    Plot an uncertain scalar

    .. jupyter-execute::

        ux = na.NormalUncertainScalarArray(x, width=0.2)
        uy = np.sin(ux)

        plt.figure();
        na.plt.plot(x, uy);

    Broadcast an array of scalars against an array of :class:`matplotlib.axes.Axes`

    .. jupyter-execute::

        fig, ax = na.plt.subplots(axis_rows="z", nrows=z.shape["z"], sharex=True)

        na.plt.plot(x, y, ax=ax, axis="x");

    Plot a 2D Cartesian vector

    .. jupyter-execute::

        v = na.Cartesian2dVectorArray(x, np.sin(x))

        plt.figure()
        na.plt.plot(v)

    """
    if transformation is not None:
        args = tuple(transformation(arg) for arg in args)
    return na._named_array_function(
        plot,
        *args,
        ax=ax,
        axis=axis,
        where=where,
        components=components,
        **kwargs,
    )


def fill(
        *args: na.AbstractArray,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        axis: None | str = None,
        where: bool | na.AbstractScalar = True,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.ScalarArray[npt.NDArray]:
    """
    Plot filled polygons

    This is a thin wrapper around :meth:`matplotlib.axes.Axes.fill` for named arrays.

    The main difference of this function from :func:`matplotlib.pyplot.fill` is the addition of the ``axis`` parameter
    indicating along which axis the lines should be connected.

    Parameters
    ----------
    args
        Same signature as :meth:`matplotlib.axes.Axes.fill`.
        If ``ax`` is a 2D plot, ``*args`` should be ``x, y``.
        If ``ax`` is a 3D plot, ``*args`` should be ``x, y, z``.
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :obj:`None`, calls :func:`matplotlib.pyplot.gca` to get the current axes.
        If an instance of :class:`named_arrays.ScalarArray`, ``ax.shape`` should be a subset of the broadcasted shape of
        ``*args``.
    axis
        The name of the axis that the plot lines should be connected along.
        If :obj:`None`, the broadcasted shape of ``args`` should have only one element,
        otherwise a :class:`ValueError` is raised.
    where
        A boolean array that selects which elements to plot
    transformation
        A callable that is applied to args before plotting
    components
        The component names of ``*args`` to plot, helpful if ``*args`` are an instance of
        :class:`named_arrays.AbstractVectorArray`.
    kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.fill`.
        These can be instances of :class:`named_arrays.AbstractArray`.
    """
    if transformation is not None:
        args = tuple(transformation(arg) for arg in args)
    return na._named_array_function(
        fill,
        *args,
        ax=ax,
        axis=axis,
        where=where,
        components=components,
        **kwargs,
    )


def scatter(
        *args: na.AbstractScalar,
        s: None | na.AbstractScalarArray = None,
        c: None | na.AbstractScalarArray = None,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray[matplotlib.axes.Axes]] = None,
        where: bool | na.AbstractScalar = True,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        **kwargs,
) -> na.ScalarArray[npt.NDArray[None | matplotlib.artist.Artist]]:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.scatter` for named arrays.

    Parameters
    ----------
    args
        Same signature as :meth:`matplotlib.axes.Axes.scatter`.
        If ``ax`` is a 2D plot, ``*args`` should be ``y`` or ``x, y``.
        If ``ax`` is a 3D plot, ``*args`` should be ``x, y`` or ``x, y, z``.
    s
        The marker size in points**2
    c
        The color of the markers
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :obj:`None`, calls :func:`matplotlib.pyplot.gca` to get the current axes.
        If an instance of :class:`named_arrays.ScalarArray`, ``ax.shape`` should be a subset of the broadcasted shape of
        ``*args``.
    where
        A boolean array that selects which elements to plot
    transformation
        A callable that is applied to args before plotting
    components
        The component names of ``*args`` to plot, helpful if ``*args`` are an instance of
        :class:`named_arrays.AbstractVectorArray`.
    kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.scatter`.
        These can be instances of :class:`named_arrays.AbstractArray`.

    Examples
    --------

    Plot a single scalar

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        x = na.linspace(0, 2 * np.pi, axis="x",  num=51)
        y = np.sin(x)

        plt.figure();
        na.plt.scatter(x, y);

    Plot an array of scalars

    .. jupyter-execute::

        z = na.linspace(0, np.pi, axis="z", num=5)

        y = np.sin(x - z)

        plt.figure();
        na.plt.scatter(x, y, c=z);

    Plot an uncertain scalar

    .. jupyter-execute::

        ux = na.NormalUncertainScalarArray(x, width=0.2)
        uy = np.sin(ux)

        plt.figure();
        na.plt.scatter(x, uy);

    Broadcast an array of scalars against an array of :class:`matplotlib.axes.Axes`

    .. jupyter-execute::

        fig, ax = na.plt.subplots(axis_rows="z", nrows=z.shape["z"], sharex=True)

        na.plt.scatter(x, y, ax=ax);
    """
    if transformation is not None:
        args = tuple(transformation(arg) for arg in args)
    return na._named_array_function(
        scatter,
        *args,
        s=s,
        c=c,
        ax=ax,
        where=where,
        components=components,
        **kwargs,
    )
