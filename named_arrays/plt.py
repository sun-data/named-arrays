from __future__ import annotations
from typing import Literal, Any
import matplotlib.axes
import matplotlib.transforms
import matplotlib.animation
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import numpy.typing as npt
import named_arrays as na

__all__ = [
    "subplots",
    "plot",
    "fill",
    "scatter",
    "imshow",
    "pcolormesh",
    "pcolormovie",
    "text",
    "brace_vertical",
    "set_xlabel",
    "get_xlabel",
    "set_ylabel",
    "get_ylabel",
    "set_title",
    "get_title",
    "set_xscale",
    "get_xscale",
    "set_yscale",
    "get_yscale",
    "set_aspect",
    "get_aspect",
    "transAxes",
    "transData",
]


def subplots(
        axis_rows: str = "subplots_row",
        axis_cols: str = "subplots_col",
        ncols: int = 1,
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

    Unlike :func:`matplotlib.pyplot.subplots()`,
    this function arranges the subplot grid with the origin in the lower-left
    corner as opposed to the upper-left corner.

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
        Controls whether all the :class:`matplotlib.axes.Axes` instances share the same vertical axis properties.
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

    axs = na.ScalarArray(axs, axes=tuple(shape.keys()))

    if axis_rows in shape:
        axs = axs[{axis_rows: slice(None, None, -1)}]

    return fig, axs


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


def imshow(
    X: na.AbstractArray,
    *,
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
) -> na.AbstractArray:
    """
    A thin wrappper around :func:`matplotlib.pyplot.imshow` for named arrays.

    Parameters
    ----------
    X
        The image data.
    axis_x
        The name of the horizontal axis.
    axis_y
        The name of the vertical axis
    axis_rgb
        The optional name of the color axis.
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :obj:`None`, calls :func:`matplotlib.pyplot.gca` to get the current axes.
        If an instance of :class:`named_arrays.ScalarArray`, ``ax.shape`` should be a subset of the broadcasted shape of
        ``*args``.
    cmap
        The Colormap instance or registered colormap name used to map scalar
        data to colors.
    norm
        The normalization method used to scale the data into the range 0 to 1
        before mapping colors.
    aspect
        The aspect ratio of the Axes.
    alpha
        The alpha blending value that ranges between 0 (transparent) and 1 (opaque).
    vmin
        The minimum value of the data range.
    vmax
        The maximum value of the data range.
    extent
        The bounding box in data coordinates that the image will fill.
        The logical axis name of the bounding box should be ``f"{axis_x},{axis_y}``
    kwargs
        An additional keyword arguments that are passed to :func:`matplotlib.pyplot.imshow`.

    Examples
    --------

    Plot a random 2d array.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the horizontal and vertical axis names of the image
        axis_x = "x"
        axis_y = "y"

        # Define a random dummy array to plot
        a = na.random.uniform(
            low=0,
            high=1,
            shape_random={axis_x: 16, axis_y: 16},
        )

        # Create the plot axes
        fig, ax = plt.subplots()

        # Plot the dummy array
        na.plt.imshow(
            a,
            axis_x=axis_x,
            axis_y=axis_y,
            ax=ax,
            extent=na.ScalarArray(
                ndarray=np.array([0, 1, 0, 1]),
                axes=f"{axis_x},{axis_y}",
            ),
        );

    |

    Plot a grid of 2d dummy arrays

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the horizontal and vertical axis names of the plot
        axis_x = "x"
        axis_y = "y"

        # Define the row and column axis names of the plot
        axis_rows = "row"
        axis_cols = "col"

        # Define the number of rows and columns of the plot
        num_rows = 2
        num_cols = 3

        # Define a random dummy array to plot
        a = na.random.uniform(
            low=0,
            high=1,
            shape_random={
                axis_rows: num_rows,
                axis_cols: num_cols,
                axis_x: 16,
                axis_y: 16,
            }
        )

        # Define the array of matplotlib axes
        fig, axs = na.plt.subplots(
            axis_rows=axis_rows,
            nrows=num_rows,
            axis_cols=axis_cols,
            ncols=num_cols,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        # Plot the dummy arrays
        na.plt.imshow(
            a,
            axis_x=axis_x,
            axis_y=axis_y,
            ax=axs,
            extent=na.ScalarArray(
                ndarray=np.array([0, 1, 0, 1]),
                axes=f"{axis_x},{axis_y}",
            ),
        );
    """
    return na._named_array_function(
        func=imshow,
        X=X,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_rgb=axis_rgb,
        ax=ax,
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        **kwargs,
    )


def pcolormesh(
    *XY: na.AbstractArray,
    C: na.AbstractArray,
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractArray = None,
    components: None | tuple[str, str] = None,
    cmap: None | str | matplotlib.colors.Colormap = None,
    norm: None | str | matplotlib.colors.Normalize = None,
    vmin: None | na.ArrayLike = None,
    vmax: None | na.ArrayLike = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :func:`matplotlib.pyplot.pcolormesh` for named arrays.

    Parameters
    ----------
    XY
        The coordinates of the mesh.
        If `C` is a scalar, `XY` can either be two scalars or one vector .
        If `C` is a function, `XY` is not specified.
        If `XY` is not specified as two scalars, the `components` must be given,
        see below.
    C
        The mesh data.
    axis_rgb
        The optional logical axis along which the RGB color channels are
        distributed.
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :obj:`None`, calls :func:`matplotlib.pyplot.gca` to get the current axes.
        If an instance of :class:`named_arrays.ScalarArray`, ``ax.shape`` should be a subset of the broadcasted shape of
        ``*args``.
    components
        If `XY` is not specified as two scalars, this parameter should
        be a tuple of two strings, specifying the vector components of `XY`
        to use as the horizontal and vertical components of the mesh.
    cmap
        The colormap used to map scalar data to colors.
    norm
        The normalization method used to scale data into the range [0, 1] before
        mapping to colors.
    vmin
        The minimum value of the data range.
    vmax
        The maximum value of the data range.
    kwargs
        Additional keyword arguments accepted by `matplotlib.pyplot.pcolormesh`

    Examples
    --------

    Plot a random 2D mesh

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the size of the grid
        shape = dict(x=16, y=16)

        # Define a simple coordinate grid
        x = na.linspace(-2, 2, axis="x", num=shape["x"])
        y = na.linspace(-1, 1, axis="y", num=shape["y"])

        # Define a random 2D array of values to plot
        a = na.random.uniform(-1, 1, shape_random=shape)

        # Plot the coordinates and values using pcolormesh
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.pcolormesh(x, y, C=a, ax=ax);

    |

    Plot a grid of random 2D meshes

    .. jupyter-execute::

        import named_arrays as na

        # Define the size of the grid
        shape = dict(row=2, col=3, x=16, y=16)

        # Define a simple coordinate grid
        x = na.linspace(-2, 2, axis="x", num=shape["x"])
        y = na.linspace(-1, 1, axis="y", num=shape["y"])

        # Define a random 2D array of values to plot
        a = na.random.uniform(-1, 1, shape_random=shape)

        # Plot the coordinates and values using pcolormesh
        fig, ax = na.plt.subplots(
            axis_rows="row",
            nrows=shape["row"],
            axis_cols="col",
            ncols=shape["col"],
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        na.plt.pcolormesh(x, y, C=a, ax=ax);
    """
    return na._named_array_function(
        pcolormesh,
        *XY,
        C=C,
        axis_rgb=axis_rgb,
        ax=ax,
        components=components,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )


def pcolormovie(
    *TXY: na.AbstractArray,
    C: na.AbstractArray,
    axis_time: str,
    axis_rgb: None | str = None,
    ax: None | matplotlib.axes.Axes | na.AbstractArray = None,
    components: None | tuple[str, str] = None,
    cmap: None | str | matplotlib.colors.Colormap = None,
    norm: None | str | matplotlib.colors.Normalize = None,
    vmin: None | na.ArrayLike = None,
    vmax: None | na.ArrayLike = None,
    kwargs_pcolormesh: None | dict[str, Any] = None,
    kwargs_animation: None | dict[str, Any] = None,
) -> matplotlib.animation.FuncAnimation:
    """
    Animate a sequence of images using :class:`matplotlib.animation.FuncAnimation`
    and repeated calls to :func:`pcolormesh`.

    Parameters
    ----------
    TXY
        The coordinates of the mesh, including the temporal coordinate.
        If `C` is a scalar, `TXY` can either be three scalars or one scalar and
        one vector.
        If `C` is a function, `TXY` is not specified.
        If `XY` is not specified as two scalars, the `components` must be given,
        see below.
    C
        The mesh data.
    axis_time
        The logical axis corresponding to the different frames in the animation.
    axis_rgb
        The optional logical axis along which the RGB color channels are
        distributed.
    ax
        The instances of :class:`matplotlib.axes.Axes` to use.
        If :obj:`None`, calls :func:`matplotlib.pyplot.gca` to get the current axes.
        If an instance of :class:`named_arrays.ScalarArray`, ``ax.shape`` should be a subset of the broadcasted shape of
        ``*args``.
    components
        If `XY` is not specified as two scalars, this parameter should
        be a tuple of two strings, specifying the vector components of `XY`
        to use as the horizontal and vertical components of the mesh.
    cmap
        The colormap used to map scalar data to colors.
    norm
        The normalization method used to scale data into the range [0, 1] before
        mapping to colors.
    vmin
        The minimum value of the data range.
    vmax
        The maximum value of the data range.
    kwargs_pcolormesh
        Additional keyword arguments accepted by :func:`pcolormesh`.
    kwargs_animation
        Additional keyword arguments accepted by
        :class:`matplotlib.animation.FuncAnimation`.

    Examples
    --------

    Plot a random 2D mesh

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import IPython.display
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na

        # Define the size of the grid
        shape = dict(
            t=3,
            x=16,
            y=16,
        )

        # Define a simple coordinate grid
        t = na.linspace(-1, 1, axis="t", num=shape["t"]) * u.s
        x = na.linspace(-2, 2, axis="x", num=shape["x"]) * u.mm
        y = na.linspace(-1, 1, axis="y", num=shape["y"]) * u.mm

        # Define a random 2D array of values to plot
        a = na.random.uniform(-1, 1, shape_random=shape)

        # Plot the coordinates and values using pcolormesh
        astropy.visualization.quantity_support()
        fig, ax = plt.subplots(constrained_layout=True)
        ani = na.plt.pcolormovie(t, x, y, C=a, axis_time="t", ax=ax);
        plt.close(fig)
        IPython.display.HTML(ani.to_jshtml())

    |

    Plot a grid of random 2D meshes

    .. jupyter-execute::

        import IPython.display
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na

        # Define the size of the grid
        shape = dict(
            t=3,
            row=2,
            col=3,
            x=16,
            y=16,
        )

        # Define a simple coordinate grid
        t = na.linspace(-1, 1, axis="t", num=shape["t"]) * u.s
        x = na.linspace(-2, 2, axis="x", num=shape["x"]) * u.mm
        y = na.linspace(-1, 1, axis="y", num=shape["y"]) * u.mm

        # Define a random 2D array of values to plot
        a = na.random.uniform(-1, 1, shape_random=shape)

        # Plot the coordinates and values using pcolormesh
        astropy.visualization.quantity_support()
        fig, ax = na.plt.subplots(
            axis_rows="row",
            axis_cols="col",
            nrows=shape["row"],
            ncols=shape["col"],
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        ani = na.plt.pcolormovie(t, x, y, C=a, axis_time="t", ax=ax);
        plt.close(fig)
        IPython.display.HTML(ani.to_jshtml())
    """
    return na._named_array_function(
        pcolormovie,
        *TXY,
        C=C,
        axis_time=axis_time,
        axis_rgb=axis_rgb,
        ax=ax,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        components=components,
        kwargs_pcolormesh=kwargs_pcolormesh,
        kwargs_animation=kwargs_animation,
    )


def text(
    x: float | u.Quantity | na.AbstractScalar,
    y: float | u.Quantity | na.AbstractScalar,
    s: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.text` for named arrays.

    Parameters
    ----------
    x
        The horizontal position of the text in data coordinates.
    y
        The vertical position of the text in data coordinates.
    s
        The text to plot.
    ax
        The matplotlib axes instance on which to plot the text.
    kwargs
        Additional keyword arguments to pass to :meth:`matplotlib.axes.Axes.text`.

    Examples
    --------

    Plot an array of text values at different locations.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na

        # Create an array of azimuths where the next will be placed
        azimuth = na.linspace(0, 360, axis="azimuth", num=4, endpoint=False) * u.deg

        # Compute the x and y coordinates for the given azimuth
        x = np.cos(azimuth)
        y = np.sin(azimuth)

        # Create an array of strings to plot at the chosen positions
        s = na.ScalarArray(
            ndarray=np.array(["East", "North", "West", "South"]),
            axes="azimuth"
        )

        # Plot the array of strings
        fig, ax = plt.subplots()
        na.plt.text(
            x=x,
            y=y,
            s=s,
            ax=ax,
        );
        ax.set_xlim(-2, 2);
        ax.set_ylim(-2, 2);
    """
    return na._named_array_function(
        text,
        x=x,
        y=y,
        s=s,
        ax=ax,
        **kwargs,
    )


def set_xlabel(
    xlabel: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.set_xlabel` for named arrays.

    Parameters
    ----------
    xlabel
        The horizontal axis label for each axis.
    ax
        The matplotlib axes instance on which to apply the label.
    """
    return na._named_array_function(
        set_xlabel,
        xlabel=na.as_named_array(xlabel),
        ax=ax,
        **kwargs,
    )


def get_xlabel(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> str | na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.get_xlabel` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the horizontal axis label from.
    """
    return na._named_array_function(
        get_xlabel,
        ax=na.as_named_array(ax),
    )



def set_ylabel(
    ylabel: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.set_ylabel` for named arrays.

    Parameters
    ----------
    ylabel
        The vertical axis label for each axis.
    ax
        The matplotlib axes instance on which to apply the label.
    """
    return na._named_array_function(
        set_ylabel,
        ylabel=na.as_named_array(ylabel),
        ax=ax,
        **kwargs,
    )


def get_ylabel(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> str | na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.get_ylabel` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the vertical axis label from.
    """
    return na._named_array_function(
        get_ylabel,
        ax=na.as_named_array(ax),
    )


def set_title(
    label: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.set_title` for named arrays.

    Parameters
    ----------
    label
        The title for each axis.
    ax
        The matplotlib axes instance on which to apply the label.
    """
    return na._named_array_function(
        set_title,
        label=na.as_named_array(label),
        ax=ax,
        **kwargs,
    )


def get_title(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> str | na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.get_title` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the title label from.
    """
    return na._named_array_function(
        get_title,
        ax=na.as_named_array(ax),
    )


def set_xscale(
    value: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.set_xscale` for named arrays.

    Parameters
    ----------
    value
        The scale type to apply to the horizontal scale of each axis.
    ax
        The matplotlib axes instance on which to apply the label.
    """
    return na._named_array_function(
        set_xscale,
        value=na.as_named_array(value),
        ax=ax,
        **kwargs,
    )


def get_xscale(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> str | na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.get_xscale` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the horizontal axis scale from.
    """
    return na._named_array_function(
        get_xscale,
        ax=na.as_named_array(ax),
    )


def set_yscale(
    value: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.set_yscale` for named arrays.

    Parameters
    ----------
    value
        The scale type to apply to the vertical scale of each axis.
    ax
        The matplotlib axes instance on which to apply the label.
    """
    return na._named_array_function(
        set_yscale,
        value=na.as_named_array(value),
        ax=ax,
        **kwargs,
    )


def get_yscale(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> str | na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.get_yscale` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the vertical axis scale from.
    """
    return na._named_array_function(
        get_yscale,
        ax=na.as_named_array(ax),
    )


def set_aspect(
    aspect: float | str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.set_aspect` for named arrays.

    Parameters
    ----------
    aspect
        The aspect ratio to apply to each axis
    ax
        The matplotlib axes instance on which to apply the label.
    """
    return na._named_array_function(
        set_aspect,
        aspect=na.as_named_array(aspect),
        ax=ax,
        **kwargs,
    )


def get_aspect(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> str | na.AbstractScalar:
    """
    A thin wrapper around :meth:`matplotlib.axes.Axes.get_aspect` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the aspect ratio from.
    """
    return na._named_array_function(
        get_aspect,
        ax=na.as_named_array(ax),
    )


def transAxes(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> matplotlib.transforms.Transform | na.AbstractScalar:
    """
    A thin wrapper around :attr:`matplotlib.axes.Axes.transAxes` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the axes transformation from.
    """
    return na._named_array_function(
        transAxes,
        ax=na.as_named_array(ax),
    )


def transData(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar = None,
) -> matplotlib.transforms.Transform | na.AbstractScalar:
    """
    A thin wrapper around :attr:`matplotlib.axes.Axes.transData` for named arrays.

    Parameters
    ----------
    ax
        The matplotlib axes instance(s) to get the axes transformation from.
    """
    return na._named_array_function(
        transData,
        ax=na.as_named_array(ax),
    )


def brace_vertical(
    x: float | u.Quantity | na.AbstractScalar,
    width: float | u.Quantity | na.AbstractScalar,
    ymin: float | u.Quantity | na.AbstractScalar,
    ymax: float | u.Quantity | na.AbstractScalar,
    ax: None | matplotlib.axes | na.AbstractScalar = None,
    label: None | str | na.AbstractScalar = None,
    beta: None | float | na.AbstractScalar = None,
    kind: Literal["left", "right"] = "left",
    kwargs_plot: None | dict[str, Any] = None,
    kwargs_text: None | dict[str, Any] = None,
    **kwargs,
) -> na.AbstractScalar:
    """
    Plot a vertical curly bracket at the given coordinates.

    Parameters
    ----------
    x
        The horizontal position of the vertical curly bracket.
    width
        The width of the curly bracket in data coordinates.
    ymin
        The minimum span of the vertical curly bracket.
    ymax
        The maximum span of the vertical curly bracket.
    ax
        A matplotlib axes instance on which to plot the curly bracket.
    label
        The optional text label for the curly bracket.
    beta
        Parameter which controls the "curlyness" of the bracket.
        If :obj:`None`, ``beta = 2 / width``.
    kind
        The kind of the brace, left or right.
    kwargs_plot
        Additional keyword arguments that are passed to :func:`plot`.
    kwargs_text
        Additional keyword arguments that are passed to :func:`text`.
    kwargs
        Additional keyword arguments that are passed to both
        :func:`plot` and :func:`text`.

    Examples
    --------

    Plot an array of braces with different lengths

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the number of braces to plot
        num = 5

        # Define the x coordinate of the braces
        x = na.linspace(0.2, 0.9, axis="y", num=num)

        # Define the y ranges of the braces
        ymin = -na.linspace(.3, .8, axis="y", num=num)
        ymax = +na.linspace(.3, .8, axis="y", num=num)

        # Define the label as the length of the brace
        label = ymax - ymin

        # Plot the braces
        fig, ax = plt.subplots()
        na.plt.brace_vertical(
            x=x,
            width=0.05,
            ymin=ymin,
            ymax=ymax,
            ax=ax,
            kind="left",
            label=label,
        );
        ax.set_xlim(0, 1);
        ax.set_ylim(-1, 1);
    """
    if kwargs_plot is None:
        kwargs_plot = dict()
    kwargs_plot = kwargs | kwargs_plot

    if kwargs_text is None:
        kwargs_text = dict()
    kwargs_text = kwargs | kwargs_text

    label = na.as_named_array(label).astype(str).astype(object)

    if beta is None:
        beta = 1 / (width / 2)

    axis = "_brace"

    y = na.linspace(ymin, ymax, axis=axis, num=1001)

    ycen = (ymin + ymax) / 2
    z = np.abs(y - ycen)

    f_outer = 1 / (1 + np.exp(-beta * (z - z.min(axis))))
    f_inner = 1 / (1 + np.exp(-beta * (z - z.max(axis))))

    f = f_outer + f_inner

    f = f - 1.5

    if kind == "left":
        x_text = x - width
        ha = "right"
        label = label + "  "
    elif kind == "right":
        f = -f
        x_text = x + width
        ha = "left"
        label = "  " + label
    else:   # pragma: nocover
        raise ValueError(
            f"Invalide kind of brace '{kind}', the only supported options are "
            f"'left' and 'right'."
        )

    f = x + width * f

    result = plot(
        f,
        y,
        ax=ax,
        axis=axis,
        **kwargs_plot,
    )

    text(
        x=x_text,
        y=ycen,
        s=label,
        ax=ax,
        ha=ha,
        va="center",
        **kwargs_text
    )

    return result
