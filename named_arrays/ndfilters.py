from __future__ import annotations
from typing import TypeVar
import named_arrays as na

__all__ = [
    "mean_filter",
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

        img = na.ScalarArray(scipy.datasets.ascent(), axes=("x", "y"))

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
        axs[1].set_title("mean filtered image");
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
