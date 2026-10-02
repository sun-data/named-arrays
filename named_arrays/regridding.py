"""
Array resampling and interpolation.

A wrapper around the :mod:`regridding` module for named arrays.
"""

from __future__ import annotations
from typing import Sequence, Literal
import numpy as np
import named_arrays as na

__all__ = [
    "regrid",
    "weights",
    "regrid_from_weights",
    "transpose_weights",
    "transpose_weights_conservative",
    "convolve_weights",
]

_seed_default = 42
"""
The default seed used to perturb the output coordinates.

Fixed so that repeated calls on the same grids return identical results.
"""


def regrid(
    coordinates_input: na.AbstractScalar | na.AbstractVectorArray,
    coordinates_output: na.AbstractScalar | na.AbstractVectorArray,
    values_input: na.AbstractScalarArray,
    axis_input: None | Sequence[str] = None,
    axis_output: None | Sequence[str] = None,
    method: Literal['multilinear', 'conservative'] = 'multilinear',
    perturb: None | bool = None,
    seed: None | int | np.random.Generator = _seed_default,
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
    perturb
        Whether to perturb `coordinates_output` by a small value to avoid degenerate
        grids. This is helpful for some methods, like ``conservative``, which
        sometimes cannot handle degenerate grids.
        If :obj:`None` (the default), no perturbation is applied unless `method`
        is ``conservative`` and the dimensions of the grid are 2D or higher.
        If :obj:`True`, each point is perturbed using a normal distribution
        with standard deviation equal to ``1e-9`` of the grid width.
    seed
        The seed used by the pseudo-random number generator which perturbs
        `coordinates_output`.
        May be an integer or an instance of :class:`numpy.random.Generator`.
        The default is a fixed integer, so that repeated calls using the same
        grids return identical results.
        If :obj:`None`, the generator is seeded from fresh entropy,
        and each call draws an independent perturbation.

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
        method=method,
        perturb=perturb,
        seed=seed,
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
    weights_input: None | na.AbstractScalar = None,
    method: Literal['multilinear', 'conservative'] = 'multilinear',
    perturb: None | bool = None,
    seed: None | int | np.random.Generator = _seed_default,
    device: None | str = None,
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
    weights_input
        Weights applied to the values of the input grid before resampling.
    method
        The type of regridding to use.
    perturb
        Whether to perturb `coordinates_output` by a small value to avoid degenerate
        grids. This is helpful for some methods, like ``conservative``, which
        sometimes cannot handle degenerate grids.
        If :obj:`None` (the default), no perturbation is applied unless `method`
        is ``conservative`` and the dimensions of the grid are 2D or higher.
        If :obj:`True`, each point is perturbed using a normal distribution
        with standard deviation equal to ``1e-9`` of the grid width.
    seed
        The seed used by the pseudo-random number generator which perturbs
        `coordinates_output`.
        May be an integer or an instance of :class:`numpy.random.Generator`.
        The default is a fixed integer, so that repeated calls using the same
        grids return identical results.
        If :obj:`None`, the generator is seeded from fresh entropy,
        and each call draws an independent perturbation.
    device
        The device on which to build the weights, passed through to
        :func:`regridding.weights`. :obj:`None` (the default) builds them on
        the host; ``"cuda"`` builds them on the GPU and leaves the weight
        values there, so that :func:`regrid_from_weights` can apply them
        without a round trip.

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
        weights_input=weights_input,
        method=method,
        perturb=perturb,
        seed=seed,
        device=device,
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


def transpose_weights(
    weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Transpose indices of weights for use backward transformation.  This is a thin wrapper around
    :func:`regridding.transpose_weights`.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`weights`.
    """

    weights, shape_input, shape_output = weights

    return na._named_array_function(
        func=transpose_weights,
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
    )


def transpose_weights_conservative(
    weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
    coordinates_input: na.AbstractScalar | na.AbstractVectorArray,
    coordinates_output: na.AbstractScalar | na.AbstractVectorArray,
    axis_input: None | str | Sequence[str] = None,
    axis_output: None | str | Sequence[str] = None,
    weights_input: None | na.AbstractScalar = None,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    Transpose weight matrix and normalize to be conservative.

    This is a thin wrapper around :func:`regridding.transpose_weights_conservative`.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`weights`.
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
    weights_input
        The weights that were applied to the input values by
        :func:`weights`. The transpose inverts this weighting (retaining a
        factor of ``1 / weights_input``) so that regridding a
        forward-transformed array with the transposed weights recovers the
        original input values.

    Examples
    --------
    Regrid a 2D array using conservative resampling, and then transform back with transposed_weights.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        import astropy.units as u

        # Define the number of edges in the input grid
        num_x = 11
        num_y = 11

        # Define a linear grid
        coordinates_input = na.Cartesian2dVectorArray(
            x=na.linspace(-5, 5, axis="x", num=num_x),
            y=na.linspace(-5, 5, axis="y", num=num_y),
        )

        # Define array of values that on grid cell centers
        values_input = na.ScalarArray.zeros(shape = dict(x=num_x-1, y=num_y-1))
        values_input[dict(x=4,y=4)] = 1

        # Rotate grid
        rot_matrix = na.Cartesian2dRotationMatrixArray(20*u.deg)
        coordinates_output = rot_matrix @ coordinates_input

        # Calculate transformation between input and output coordinates:
        weights = na.regridding.weights(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            method="conservative",
        )

        # Regrid values onto output coordinates
        values_output = na.regridding.regrid_from_weights(
            *weights,
            values_input=values_input
        )

        # Transpose weights
        weights_transposed = na.regridding.transpose_weights_conservative(
            weights=weights,
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
        )

        # Regrid the regridded values back onto original grid using transposed weights.
        values_transposed = na.regridding.regrid_from_weights(
            *weights_transposed,
            values_input=values_output
        )

        # Plot the original and regridded arrays of values
        fig, ax = plt.subplots(
            ncols=3,
            sharex=True,
            sharey=True,
            figsize=(8, 4),
            constrained_layout=True,
        );
        na.plt.pcolormesh(coordinates_input, C=values_input, ax=ax[0], vmin=0, vmax=1)
        na.plt.pcolormesh(coordinates_output, C=values_output, ax=ax[1], vmin=0, vmax=1)
        na.plt.pcolormesh(coordinates_input, C=values_transposed, ax=ax[2], vmin=0, vmax=1)
        ax[0].set_title("original");
        ax[1].set_title("rotated");
        ax[2].set_title("rotated and transposed");
    """

    weights, shape_input, shape_output = weights

    return na._named_array_function(
        func=transpose_weights_conservative,
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
    )


def convolve_weights(
    weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
    kernel: na.AbstractScalar,
    axis: dict[str, str],
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    r"""
    Convolve the output of a set of weights with a kernel, such as a
    point-spread function.

    If the weights computed by :func:`weights` are the sparse matrix
    :math:`W`, this computes :math:`P W`, where :math:`P` spreads whatever
    lands in each output cell over the cells around it.  Applying the
    result with :func:`regrid_from_weights` resamples and blurs in one step,
    and :func:`transpose_weights_conservative` transposes the whole
    operation.

    This is the named-axis form of :func:`regridding.convolve_weights`.

    Parameters
    ----------
    weights
        Weights computed by :func:`weights`.
    kernel
        The kernel: the fraction of the light landing in a cell which
        reaches each of the cells around it.
        It must be dimensionless, finite, and without uncertainty.

        The axes named in `axis` are the axes of the kernel itself.
        Along an axis of length :math:`n`, the element at index
        :math:`\lfloor n / 2 \rfloor` is the cell the light lands in, which
        is where :func:`named_arrays.convolve` and
        :func:`scipy.ndimage.convolve` place the center.

        Every other axis is broadcast by name, as in any other operation:

        * an axis the weights are an array over, such as a wavelength or a
          channel, gives a different kernel for each of its elements;
        * a resampled axis of the output grid gives a kernel which varies
          across the grid, indexed by the cell the light lands in, which is
          how a kernel which varies across the field is expressed;
        * any other axis is added to the weights and to both shapes, so the
          weights become a set for each of its elements.

        The kernel cannot vary along a resampled axis of the input grid
        unless that axis is also one of the output grid, in which case it is
        the output grid's axis.
    axis
        A dict mapping each resampled axis of the output grid to convolve
        along onto the kernel axis which runs along it, such as
        ``dict(sensor_x="kernel_x", sensor_y="kernel_y")``.
        A resampled axis left out is not convolved along.
        The axes of the kernel itself have to be named differently from the
        axes of the grids.

    Returns
    -------
    The convolved weights, with the orthogonal axes first in both shapes, as
    :func:`weights` returns them.

    Raises
    ------
    TypeError
        If `kernel` is not a scalar without uncertainty, or `axis` is not a
        dict.
    ValueError
        If `axis` names an axis which is not a resampled output axis, or a
        kernel axis which the kernel lacks, which is empty, which is also an
        axis of the grids, or which runs along two output axes; if the kernel
        is not dimensionless or not finite; if it varies along a resampled
        input axis, or along an output axis with a different number of cells;
        or if its other axes cannot be broadcast against the weights.

    See Also
    --------
    :func:`regridding.convolve_weights`: The equivalent function for
    instances of :class:`numpy.ndarray`, which describes the algorithm.

    Notes
    -----
    The kernel acts on cells, so for a point-spread function it should be
    the point-spread function convolved with a cell twice, once for the cell
    the light lands in and once for the cell it is collected in.

    A kernel held as a :class:`~named_arrays.FunctionArray` of the offsets of
    its elements can be passed as its ``outputs`` only if the offsets along
    each axis run from :math:`-\lfloor n / 2 \rfloor`, so that the element
    at index :math:`\lfloor n / 2 \rfloor` is the one at zero offset.
    Otherwise the result is shifted by the difference.

    Examples
    --------

    Rotate an array onto a new grid, and blur it with a Gaussian kernel in
    the same set of weights.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na

        # Define a grid of vertices, and rotate it
        coordinates_input = na.Cartesian2dVectorLinearSpace(
            start=-4,
            stop=4,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=17,
        ).explicit
        coordinates_input = na.Cartesian2dRotationMatrixArray(20 * u.deg) @ coordinates_input

        # Define a uniform output grid
        coordinates_output = na.Cartesian2dVectorLinearSpace(
            start=-6,
            stop=6,
            axis=na.Cartesian2dVectorArray("x_new", "y_new"),
            num=25,
        )

        # Define an array of values with two bright cells
        values_input = na.ScalarArray.zeros(dict(x=16, y=16))
        values_input[dict(x=4, y=4)] = 1
        values_input[dict(x=10, y=8)] = 1

        # Save the weights which rotate the input array onto the output grid
        weights = na.regridding.weights(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            method="conservative",
        )

        # Define a Gaussian kernel, five cells across
        offset_x = na.arange(-2, 3, axis="kernel_x")
        offset_y = na.arange(-2, 3, axis="kernel_y")
        kernel = np.exp(-(np.square(offset_x) + np.square(offset_y)) / 2)
        kernel = kernel / kernel.sum()

        # Blur the output of the weights with the kernel
        weights_blurred = na.regridding.convolve_weights(
            weights=weights,
            kernel=kernel,
            axis=dict(x_new="kernel_x", y_new="kernel_y"),
        )

        # Apply both sets of weights
        values_rotated = na.regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )
        values_blurred = na.regridding.regrid_from_weights(
            *weights_blurred,
            values_input=values_input,
        )

        # Plot the original, rotated, and blurred arrays
        fig, ax = plt.subplots(
            ncols=3,
            sharex=True,
            sharey=True,
            figsize=(9, 3.4),
            constrained_layout=True,
        )
        na.plt.pcolormesh(coordinates_input, C=values_input, ax=ax[0])
        na.plt.pcolormesh(coordinates_output, C=values_rotated, ax=ax[1])
        na.plt.pcolormesh(coordinates_output, C=values_blurred, ax=ax[2])
        ax[0].set_title(f"original, total {values_input.sum().ndarray:.2f}")
        ax[1].set_title(f"rotated, total {values_rotated.sum().ndarray:.2f}")
        ax[2].set_title(f"rotated and blurred, total {values_blurred.sum().ndarray:.2f}")
        for a in ax:
            a.set_aspect("equal")
    """

    weights, shape_input, shape_output = weights

    return na._named_array_function(
        func=convolve_weights,
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        kernel=kernel,
        axis=axis,
    )
