"""
Array resampling and interpolation.

A wrapper around the :mod:`regridding` module for named arrays.
"""

from __future__ import annotations
from typing import Sequence, Literal
import astropy.units as u
import named_arrays as na

__all__ = [
    "regrid",
    "weights",
    "regrid_from_weights",
]


def regrid(
    coordinates_input: na.AbstractScalar | na.AbstractVectorArray,
    coordinates_output: float | u.Quantity,
    values_input: na.AbstractScalarArray,
    axis_input: None | Sequence[str] = None,
    axis_output: None | Sequence[str] = None,
    method: Literal['multilinear', 'conservative'] = 'multilinear',
) -> na.AbstractScalarArray:
    """
    Regrid an array of values defined on a logically-rectangular curvilinear
    grid onto a new logically-rectangular curvilinear grid.

    Parameters
    ----------
    coordinates_input
        Coordinates of the input grid.
    coordinates_output
        Coordinates of the output grid.
        Should have the same number of components as the input grid.
    values_input
        Input array of values to be resampled.
    axis_input
        Logical axes of the input grid to resample.
        If :obj:`None`, resample all the axes of the input grid.
        The number of axes should be equal to the number of
        coordinates in the input grid.
    axis_output
        Logical axes of the output grid corresponding to the resampled axes
        of the input grid.
        If :obj:`None`, all the axes of the output grid correspond to resampled
        axes in the input grid.
        The number of axes should be equal to the number of
        coordinates in the output grid.
    method
        The type of regridding to use.

    Examples
    --------

    Regrid a 2D array using conservative resampling.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the number of edges in the input grid
        num_x = 66
        num_y = 66

        # Define a dummy linear grid
        x = na.linspace(-5, 5, axis="x", num=num_x)
        y = na.linspace(-5, 5, axis="y", num=num_y)

        # Define the curvilinear input grid using the dummy grid
        angle = 0.4
        coordinates_input = na.Cartesian2dVectorArray(
            x=x * np.cos(angle) - y * np.sin(angle) + 0.05 * x * x,
            y=x * np.sin(angle) + y * np.cos(angle) + 0.05 * y * y,
        )

        # Define the test pattern
        a_input = np.cos(np.square(x)) * np.cos(np.square(y))
        a_input = a_input.cell_centers()

        # Define a rectilinear output grid using the limits of the input grid
        coordinates_output = na.Cartesian2dVectorLinearSpace(
            start=coordinates_input.min(),
            stop=coordinates_input.max(),
            axis=na.Cartesian2dVectorArray("x2", "y2"),
            num=66,
        )

        # Regrid the test pattern onto the new grid
        a_output = na.regridding.regrid(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            values_input=a_input,
            method="conservative",
        )

        fig, ax = plt.subplots(
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=(8, 4),
            constrained_layout=True,
        );
        na.plt.pcolormesh(coordinates_input, C=a_input, ax=ax[0])
        na.plt.pcolormesh(coordinates_output, C=a_output, ax=ax[1])
        ax[0].set_title("input array");
        ax[1].set_title("regridded array");
    """
    _weights, shape_input, shape_output = weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method
    )

    result = regrid_from_weights(
        weights=_weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values_input,
    )

    return result


def weights(
    coordinates_input: na.AbstractScalar | na.AbstractVectorArray,
    coordinates_output: na.AbstractScalar | na.AbstractVectorArray,
    axis_input: None | str | Sequence[str] = None,
    axis_output: None | str | Sequence[str] = None,
    method: Literal['multilinear', 'conservative'] = 'multilinear',
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Save the results of a regridding operation as a sequence of weights,
    which can be used in subsequent regridding operations on the same grid.

    The results of this function are designed to be used by
    :func:`regrid_from_weights`

    This function returns a tuple containing a ragged array of weights,
    the shape of the input coordinates, and the shape of the output coordinates.

    Parameters
    ----------
    coordinates_input
        Coordinates of the input grid.
    coordinates_output
        Coordinates of the output grid.
        Should have the same number of coordinates as the input grid.
    axis_input
        Logical axes of the input grid to resample.
        If :obj:`None`, resample all the axes of the input grid.
        The number of axes should be equal to the number of
        coordinates in the input grid.
    axis_output
        Logical axes of the output grid corresponding to the resampled axes
        of the input grid.
        If :obj:`None`, all the axes of the output grid correspond to resampled
        axes in the input grid.
        The number of axes should be equal to the number of
        coordinates in the output grid.
    method
        The type of regridding to use.

    See Also
    --------
    :func:`regridding.weights`: An equivalent function for instances of :class:`numpy.ndarray`.
    :func:`regrid_from_weights`: A function designed to use the outputs of this function.
    :func:`regrid`: Resample an array without saving the weights

    """
    return na._named_array_function(
        func=weights,
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
    )


def regrid_from_weights(
    weights: na.AbstractScalar,
    shape_input: dict[str, int],
    shape_output: dict[str, int],
    values_input: na.AbstractScalar | na.AbstractVectorArray,
) -> na.AbstractArray:
    """
    Regrid an array of values using weights computed by
    :func:`weights`.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`weights`.
    shape_input
        Broadcasted shape of the input coordinates computed by :func:`weights`.
    shape_output
        Broadcasted shape of the output coordinates computed by :func:`weights`.
    values_input
        Input array of values to be resampled.
    """
    return na._named_array_function(
        func=regrid_from_weights,
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values_input,
    )
