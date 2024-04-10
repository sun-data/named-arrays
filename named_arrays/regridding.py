from __future__ import annotations
import named_arrays as na
import numpy as np
from typing import Sequence, Literal
import regridding

__all__ = [
    "regrid",
    "weights",
    "regrid_from_weights",
]


def regrid(
        coordinates_input: na.AbstractArray,
        coordinates_output: na.AbstractArray,
        values_input: na.AbstractScalarArray,
        values_output: None | na.AbstractScalarArray = None,
        axis_input: None | Sequence[str] = None,
        axis_output: None | Sequence[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
) -> na.AbstractScalarArray:
    weights, shape_input, shape_output = na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method
    )

    result = regrid_from_weights(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    return result


def weights(
        coordinates_input: na.AbstractArray,
        coordinates_output: na.AbstractArray,
        axis_input: None | Sequence[str] = None,
        axis_output: None | Sequence[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
) -> na.AbstractArray:

    if axis_output is not None:
        return NotImplementedError

    if np.any(coordinates_output.cartesian_nd.components.keys == coordinates_input.cartesian_nd.components.keys):
        return ValueError

    input_coordinates_broadcasted = coordinates_input.cartesian_nd.broadcasted
    output_coordinates_broadcasted = coordinates_output.cartesian_nd.broadcasted

    # broadcast input(old) coordinates against output(new) coordinates along missing axes
    broadcasted_input_shape = input_coordinates_broadcasted.shape
    for key in output_coordinates_broadcasted.shape:
        if key not in input_coordinates_broadcasted.shape:
            broadcasted_input_shape[key] = output_coordinates_broadcasted.shape[key]

    input_coordinates_broadcasted = input_coordinates_broadcasted.broadcast_to(broadcasted_input_shape)

    # broadcast output(new) coordinates against input(old) coordinates along missing axes
    broadcasted_output_shape = output_coordinates_broadcasted.shape
    for key in input_coordinates_broadcasted.shape:
        if key not in output_coordinates_broadcasted.shape:
            broadcasted_output_shape[key] = input_coordinates_broadcasted.shape[key]

    output_coordinates_broadcasted = output_coordinates_broadcasted.broadcast_to(broadcasted_output_shape)

    orthogonal_axes = [key for key in input_coordinates_broadcasted.shape if key not in axis_input]

    coords_input = []
    for c in input_coordinates_broadcasted.components:
        component = input_coordinates_broadcasted.components[c]
        coords_input.append(
            component.broadcast_to(dict([(c, component.shape[c]) for c in broadcasted_output_shape])).ndarray
        )

    coords_output = []
    for c in output_coordinates_broadcasted.components:
        component = output_coordinates_broadcasted.components[c]
        coords_output.append(
            component.broadcast_to(dict([(c, component.shape[c]) for c in broadcasted_output_shape])).ndarray
        )

    interp_axes = tuple(list(broadcasted_output_shape.keys()).index(key) for key in axis_input)

    if len(coords_input) == 1:
        method = 'multilinear'
    elif len(coords_input) == 2:
        method = 'conservative'

    weights, shape_input, shape_output = regridding.weights(
        coordinates_input=tuple(coords_input),
        coordinates_output=tuple(coords_output),
        axis_input=interp_axes,
        axis_output=interp_axes,
        method=method,
    )

    shape_input = dict([(key, shape) for key, shape in zip(broadcasted_output_shape.keys(), shape_input)])
    shape_output = dict([(key, shape) for key, shape in zip(broadcasted_output_shape.keys(), shape_output)])
    weights = na.ScalarArray(np.array(weights), axes=orthogonal_axes)

    return weights, shape_input, shape_output


def regrid_from_weights(
        weights: np.AbstractScalarArray,
        shape_input: na.AbstractArray,
        shape_output: na.AbstractArray,
        values_input: na.AbstractScalarArray,
        values_output: None | na.AbstractScalarArray = None,
        axis_input: None | Sequence[str] = None,
        axis_output: None | Sequence[str] = None,
):
    if axis_output is None:
        axis_output = axis_input


    #broadcast values along any new input coordinate dimensions
    values_input_broadcasted_shape = na.broadcast_shapes(values_input.shape, shape_input)
    values_input = values_input.broadcast_to(values_input_broadcasted_shape)
    shape_input = values_input_broadcasted_shape

    shape_orthogonal = {}
    for key in shape_input:
        if key not in axis_input:
            shape_orthogonal[key] = shape_input[key]

    weights = weights.broadcast_to(shape_orthogonal)

    for key in shape_input:
        if key not in shape_output:
            shape_output[key] = shape_input[key]

    values_input_ndarray = values_input.ndarray
    interp_axes_input = tuple(list(shape_input).index(key) for key in axis_input)

    values_output_ndarray = regridding.regrid_from_weights(
        weights=weights.ndarray,
        shape_input=tuple([shape_input[key] for key in shape_input]),
        shape_output=tuple([shape_output[key] for key in shape_input]),
        # every ndarray should be in the axes order of shape_input
        values_input=values_input_ndarray,
        # values_output=values_output,
        axis_input=interp_axes_input,
        axis_output=interp_axes_input,
    )
    values_output = na.ScalarArray(values_output_ndarray, axes=values_input.shape.keys())

    return values_output
