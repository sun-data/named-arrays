"""
A thin wrapper around the :mod:`ndfilters` package for named arrays.
"""

from __future__ import annotations
from typing import TypeVar, Literal
import named_arrays as na

__all__ = [
    "mean_filter",
    "trimmed_mean_filter",
    "variance_filter",
]

ArrayT = TypeVar("ArrayT", bound="na.AbstractArray")
WhereT = TypeVar("WhereT", bound="bool | na.AbstractArray")


def mean_filter(
    array: ArrayT,
    size: dict[str, int],
    where: WhereT = True,
) -> ArrayT | WhereT:
    """
    A thin wrapper around :func:`ndfilters.mean_filter` for named arrays.

    Parameters
    ----------
    array
        The input array to be filtered.
    size
        The shape of the kernel over which the mean will be calculated.
    where
        A boolean mask used to select which elements of the input array are to
        be filtered.

    Examples
    --------

    Filter a sample image.

    .. jupyter-execute::
        :stderr:

        import matplotlib.pyplot as plt
        import scipy.datasets
        import named_arrays as na

        img = na.ScalarArray(scipy.datasets.ascent(), axes=("y", "x"))

        img_filtered = na.ndfilters.mean_filter(img, size=dict(x=21, y=21))

        fig, axs = plt.subplots(
            ncols=2,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs[0].set_title("original image");
        na.plt.imshow(
            X=img,
            axis_x="x",
            axis_y="y",
            ax=axs[0],
            cmap="gray",
        );
        axs[1].set_title("filtered image");
        na.plt.imshow(
            X=img_filtered,
            axis_x="x",
            axis_y="y",
            ax=axs[1],
            cmap="gray",
        );
    """
    return na._named_array_function(
        func=mean_filter,
        array=array,
        size=size,
        where=where,
    )


def trimmed_mean_filter(
    array: ArrayT,
    size: dict[str, int],
    where: WhereT = True,
    mode: Literal["mirror", "nearest", "wrap", "truncate"] = "mirror",
    proportion: float = 0.25,
) -> ArrayT | WhereT:
    """
    A thin wrapper around :func:`ndfilters.trimmed_mean_filter` for named arrays.

    Parameters
    ----------
    array
        The input array to be filtered.
    size
        The shape of the kernel over which the mean will be calculated.
    where
        A boolean mask used to select which elements of the input array are to
        be filtered.
    mode
        The method used to extend the input array beyond its boundaries.
        See :func:`scipy.ndimage.generic_filter` for the definitions.
        Currently, only "mirror", "nearest", "wrap", and "truncate" modes are
        supported.
    proportion
        The proportion to cut from the top and bottom of the distribution.

    Examples
    --------

    Filter a sample image.

    .. jupyter-execute::
        :stderr:

        import matplotlib.pyplot as plt
        import scipy.datasets
        import named_arrays as na

        img = na.ScalarArray(scipy.datasets.ascent(), axes=("y", "x"))

        img_filtered = na.ndfilters.trimmed_mean_filter(
            array=img,
            size=dict(x=21, y=21),
        )

        fig, axs = plt.subplots(
            ncols=2,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs[0].set_title("original image");
        na.plt.imshow(
            X=img,
            axis_x="x",
            axis_y="y",
            ax=axs[0],
            cmap="gray",
        );
        axs[1].set_title("filtered image");
        na.plt.imshow(
            X=img_filtered,
            axis_x="x",
            axis_y="y",
            ax=axs[1],
            cmap="gray",
        );
    """
    return na._named_array_function(
        func=trimmed_mean_filter,
        array=array,
        size=size,
        where=where,
        mode=mode,
        proportion=proportion,
    )


def variance_filter(
    array: ArrayT,
    size: dict[str, int],
    where: WhereT = True,
) -> ArrayT | WhereT:
    """
    A thin wrapper around :func:`ndfilters.variance_filter` for named arrays.

    Parameters
    ----------
    array
        The input array to be filtered.
    size
        The shape of the kernel over which the variance will be calculated.
    where
        A boolean mask used to select which elements of the input array are to
        be filtered.

    Examples
    --------

    Filter a sample image.

    .. jupyter-execute::
        :stderr:

        import matplotlib.pyplot as plt
        import scipy.datasets
        import named_arrays as na

        img = na.ScalarArray(scipy.datasets.ascent(), axes=("y", "x"))

        img_filtered = na.ndfilters.variance_filter(img, size=dict(x=21, y=21))

        fig, axs = plt.subplots(
            ncols=2,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs[0].set_title("original image");
        na.plt.imshow(
            X=img,
            axis_x="x",
            axis_y="y",
            ax=axs[0],
            cmap="gray",
        );
        axs[1].set_title("filtered image");
        na.plt.imshow(
            X=img_filtered,
            axis_x="x",
            axis_y="y",
            ax=axs[1],
            cmap="gray",
        );
    """
    return na._named_array_function(
        func=variance_filter,
        array=array,
        size=size,
        where=where,
    )
