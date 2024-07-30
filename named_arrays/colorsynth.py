"""
A thin wrapper around the :mod:`colorsynth` package for named arrays.
"""

from __future__ import annotations
from typing import Callable
import named_arrays as na

__all__ = [
    "rgb",
    "colorbar",
    "rgb_and_colorbar",
]


def rgb(
    spd: na.AbstractArray,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | na.ArrayLike = None,
    spd_max: None | na.ArrayLike = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | na.ScalarLike = None,
    wavelength_max: None | na.ScalarLike = None,
    wavelength_norm: None | Callable = None,
) -> na.AbstractArray:
    """
    A thin wrapper around :func:`colorsynth.rgb()` for named arrays.

    Parameters
    ----------
    spd
        A spectral power distribution to convert to a RGB array.
    wavelength
        The wavelength coordinates corresponding to the spectral power distribution.
        If :obj:`None` (the default), the wavelength is assumed to be evenly
        sampled over the human visible range.
    axis
        The logical axis corresponding to changing wavelength.
        If :obj:`None` (the default), `spd` must be one-dimensional.
    spd_min
        The value of the spectral power distribution representing minimum
        intensity
    spd_max
        The vale of the spectral power distribution representing maximum
        intensity
    spd_norm
        An optional function that transforms the spectral power distribution
        values before mapping to RGB.
    wavelength_min
        The wavelength value that is mapped to the minimum wavelength of the
        human visible color range, 380 nm.
    wavelength_max
        The wavelength value that is mapped to the maximum wavelength of the
        human visible color range, 700 nm
    wavelength_norm
        An optional function to transform the wavelength values before they
        are mapped into the human visible color range.

    See Also
    --------
    :func:`colorsynth.rgb` :  Equivalent function for instances of :class:`numpy.ndarray`.

    Examples
    --------
    Plot the color of a random 3d cube

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define a random 3d cube
        a = na.random.uniform(
            low=0,
            high=1,
            shape_random=dict(x=16, y=16, wavelength=11)
        )

        # Compute the RGB colors of the 3d cube
        rgb = na.colorsynth.rgb(
            spd=a,
            axis="wavelength",
        )

        # Plot the RGB image
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.imshow(
            rgb,
            axis_x="x",
            axis_y="y",
            axis_rgb="wavelength",
            ax=ax,
        );
    """
    return na._named_array_function(
        func=rgb,
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )


def colorbar(
    spd: na.AbstractArray,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | na.ArrayLike = None,
    spd_max: None | na.ArrayLike = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | na.ScalarLike = None,
    wavelength_max: None | na.ScalarLike = None,
    wavelength_norm: None | Callable = None,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
    """
    A thin wrapper around :func:`colorsynth.colorbar()` for named arrays.

    Parameters
    ----------
    spd
        A spectral power distribution to convert to a RGB array.
    wavelength
        The wavelength coordinates corresponding to the spectral power distribution.
        If :obj:`None` (the default), the wavelength is assumed to be evenly
        sampled over the human visible range.
    axis
        The logical axis corresponding to changing wavelength.
        If :obj:`None` (the default), `spd` must be one-dimensional.
    spd_min
        The value of the spectral power distribution representing minimum
        intensity
    spd_max
        The vale of the spectral power distribution representing maximum
        intensity
    spd_norm
        An optional function that transforms the spectral power distribution
        values before mapping to RGB.
    wavelength_min
        The wavelength value that is mapped to the minimum wavelength of the
        human visible color range, 380 nm.
    wavelength_max
        The wavelength value that is mapped to the maximum wavelength of the
        human visible color range, 700 nm
    wavelength_norm
        An optional function to transform the wavelength values before they
        are mapped into the human visible color range.

    See Also
    --------
    :func:`colorsynth.colorbar` :  Equivalent function for instances of :class:`numpy.ndarray`.

    Examples
    --------

    Plot the colorbar of a random, 3D cube.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na

        # Define a random 3d cube
        a = na.random.uniform(
            low=0 * u.photon,
            high=1000 * u.photon,
            shape_random=dict(x=16, y=16, wavelength=11),
        )

        # Define wavelength axis
        wavelength = na.linspace(
            start=100 * u.AA,
            stop=200 * u.AA,
            axis="wavelength",
            num=a.shape["wavelength"],
        )

        # Compute the colorbar corresponding to the random 3d cube.
        colorbar = na.colorsynth.colorbar(
            spd=a,
            wavelength=wavelength,
            axis="wavelength",
        )

        # Plot the colorbar
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb="wavelength",
                ax=ax,
            )
    """
    return na._named_array_function(
        func=colorbar,
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )


def rgb_and_colorbar(
    spd: na.AbstractArray,
    wavelength: None | na.AbstractScalar = None,
    axis: None | str = None,
    spd_min: None | na.ArrayLike = None,
    spd_max: None | na.ArrayLike = None,
    spd_norm: None | Callable = None,
    wavelength_min: None | na.ScalarLike = None,
    wavelength_max: None | na.ScalarLike = None,
    wavelength_norm: None | Callable = None,
) -> tuple[
    na.AbstractArray,
    na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar],
]:
    """
    Convenience function that calls :func:`rgb` and :func:`colorbar` and
    returns the results as a tuple.

    Parameters
    ----------
    spd
        A spectral power distribution to convert to a RGB array.
    wavelength
        The wavelength coordinates corresponding to the spectral power distribution.
        If :obj:`None` (the default), the wavelength is assumed to be evenly
        sampled over the human visible range.
    axis
        The logical axis corresponding to changing wavelength.
        If :obj:`None` (the default), `spd` must be one-dimensional.
    spd_min
        The value of the spectral power distribution representing minimum
        intensity
    spd_max
        The vale of the spectral power distribution representing maximum
        intensity
    spd_norm
        An optional function that transforms the spectral power distribution
        values before mapping to RGB.
    wavelength_min
        The wavelength value that is mapped to the minimum wavelength of the
        human visible color range, 380 nm.
    wavelength_max
        The wavelength value that is mapped to the maximum wavelength of the
        human visible color range, 700 nm
    wavelength_norm
        An optional function to transform the wavelength values before they
        are mapped into the human visible color range.

    See Also
    --------
    :func:`colorsynth.colorbar` :  Equivalent function for instances of :class:`numpy.ndarray`.

    Examples
    --------

    Plot the colorbar of a random, 3D cube.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na

        # Define a random 3d cube
        a = na.random.uniform(
            low=0 * u.photon,
            high=1000 * u.photon,
            shape_random=dict(x=16, y=16, wavelength=11),
        )

        # Define wavelength axis
        wavelength = na.linspace(
            start=100 * u.AA,
            stop=200 * u.AA,
            axis="wavelength",
            num=a.shape["wavelength"],
        )

        # Compute the colorbar corresponding to the random 3d cube.
        rgb, colorbar = na.colorsynth.rgb_and_colorbar(
            spd=a,
            wavelength=wavelength,
            axis="wavelength",
        )

        # Plot the colorbar
        with astropy.visualization.quantity_support():
            fig, axs = plt.subplots(
                ncols=2,
                gridspec_kw=dict(width_ratios=[.9,.1]),
                constrained_layout=True,
            )
            na.plt.imshow(
                rgb,
                axis_x="x",
                axis_y="y",
                axis_rgb="wavelength",
                ax=axs[0],
            );
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb="wavelength",
                ax=axs[1],
            )
            axs[1].yaxis.tick_right()
            axs[1].yaxis.set_label_position("right")

    """
    kwargs = dict(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )
    result_rgb = rgb(**kwargs)
    result_colorbar = colorbar(**kwargs)
    return result_rgb, result_colorbar
