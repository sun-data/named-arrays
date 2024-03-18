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
    if coordinates_output.ndim == 1:
        method = 'multilinear'
    else:
        method = 'conservative'

    if axis_output is not None:
        return NotImplementedError

    # check old and new coordinates are the same type AbstractArray.  May want to compare compenents of
    # cartesian_nd instead
    if np.any(coordinates_output.cartesian_nd.components.keys == coordinates_input.cartesian_nd.components.keys):
        return ValueError

    input_coordinates_broadcasted = coordinates_input.cartesian_nd.broadcasted
    output_coordinates_broadcasted = coordinates_output.cartesian_nd.broadcasted

    #broadcast input coordinates against new output if missing axes
    broadcasted_input_shape = {}
    for key in output_coordinates_broadcasted.shape:
        if key in input_coordinates_broadcasted.shape:
            broadcasted_input_shape[key] = input_coordinates_broadcasted.shape[key]
        else:
            broadcasted_input_shape[key] = output_coordinates_broadcasted.shape[key]

    input_coordinates_broadcasted = input_coordinates_broadcasted.broadcast_to(broadcasted_input_shape)

    output_broadcasted_shape = output_coordinates_broadcasted.shape
    print(f'{output_broadcasted_shape=}')
    print(f'{input_coordinates_broadcasted.shape=}')

    coords_input = []
    for c in input_coordinates_broadcasted.components:
        component = input_coordinates_broadcasted.components[c]
        coords_input.append(
            component.broadcast_to(dict([(c, component.shape[c]) for c in output_broadcasted_shape])).ndarray
        )

    coords_output = []
    for c in output_coordinates_broadcasted.components:
        component = output_coordinates_broadcasted.components[c]
        coords_output.append(
            component.broadcast_to(dict([(c, component.shape[c]) for c in output_broadcasted_shape])).ndarray
        )

    interp_axes = tuple(list(output_broadcasted_shape.keys()).index(key) for key in axis_input)
    # print(f'{coords_input.shape=}')
    # print(f'{coords_output.shape=}')
    print(f'{axis_input=}')
    print(f'{interp_axes=}')

    weights, shape_input, shape_output = regridding.weights(
        coordinates_input=tuple(coords_input),
        coordinates_output=tuple(coords_output),
        axis_input=interp_axes,
        axis_output=interp_axes,
        method=method,
    )

    shape_input = dict([(key, shape) for key, shape in zip(axis_input, shape_input)])
    shape_output = dict([(key, shape) for key, shape in zip(axis_input, shape_output)])

    # values_input.broadcast_to(dict([(c, values_input.shape[c]) for c in new_shape]))
    print(f'{weights=}')
    print(f'{weights.shape=}')
    print(f'{shape_input=}')
    print(f'{shape_output=}')

    return weights, shape_input, shape_output


def regrid_from_weights(
        weights: np.ndarray,
        shape_input: na.AbstractArray,
        shape_output: na.AbstractArray,
        values_input: na.AbstractScalarArray,
        values_output: None | na.AbstractScalarArray = None,
        axis_input: None | Sequence[str] = None,
        axis_output: None | Sequence[str] = None,
):
    values_input_ndarray = values_input.ndarray
    interp_axes = tuple(list(values_input.shape.keys()).index(key) for key in axis_input)
    print(f'{interp_axes=}')

    values_output_ndarray = regridding.regrid_from_weights(
        weights=weights,
        shape_input=tuple([shape_input[key] for key in shape_input]),
        shape_output=tuple([shape_output[key] for key in shape_input]),
        # every ndarray should be in the axes order of shape_input
        values_input=values_input_ndarray,
        # values_output=values_output,
        axis_input=interp_axes,
        axis_output=interp_axes,
    )
    values_output = na.ScalarArray(values_output_ndarray, axes=values_input.shape.keys())
    print(f'{values_output.shape=}')

    return values_output
