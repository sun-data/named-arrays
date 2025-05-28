import pytest
import numpy as np
import named_arrays as na

shape_vertices = dict(x=10, y=11)
shape_centers = {a: shape_vertices[a] - 1 for a in shape_vertices}

x = na.linspace(-1, 1, axis="x", num=shape_vertices["x"])
y = na.linspace(-1, 1, axis="y",  num=shape_vertices["y"])
z = na.linspace(-1, 1, axis="z", num=3)

x_new = na.linspace(-1, 1, axis="x_new", num=5)
y_new = na.linspace(-1, 1, axis="y_new", num=6)


@pytest.mark.parametrize(
    argnames="coordinates_input,coordinates_output,values_input,axis_input,axis_output,result_expected",
    argvalues=[
        (
            na.linspace(-1, 1, axis="x_input", num=11),
            na.linspace(-1, 1, axis="x_output", num=11),
            np.square(na.linspace(-1, 1, axis="x_input", num=11)),
            None,
            None,
            np.square(na.linspace(-1, 1, axis="x_output", num=11)),
        ),
        (
            y,
            y_new,
            x + y,
            "y",
            "y_new",
            x + y_new,
        ),
        (
            x,
            x_new,
            x + y,
            ("x",),
            ("x_new",),
            x_new + y,
        ),
        (
            x,
            0.1 * x_new + 0.001 * y_new,
            x,
            ("x",),
            ("x_new",),
            0.1 * x_new + 0.001 * y_new,
        ),
    ],
)
def test_regrid_multilinear_1d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    result_expected: np.ndarray,
):
    result = na.regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        axis_input=axis_input,
        axis_output=axis_output,
        method="multilinear",
    )
    assert isinstance(result, na.AbstractArray)
    assert np.issubdtype(result.dtype, float)
    assert np.allclose(result, result_expected)


@pytest.mark.parametrize(
    argnames="coordinates_input, values_input, axis_input, coordinates_output, axis_output",
    argvalues=[
        (
            na.Cartesian2dVectorArray(x, y),
            na.random.normal(0, 1, shape_random=shape_centers),
            None,
            na.Cartesian2dVectorArray(
                x=1.1 * x + 0.01,
                y=1.2 * y + 0.01,
            ),
            None,
        ),
        (
            na.Cartesian2dVectorArray(
                x=x + 0.01 * z,
                y=y + 0.01 * z,
            ),
            na.random.normal(0, 1, shape_random=shape_centers | z.shape),
            ("x", "y"),
            na.Cartesian2dVectorArray(
                x=1.1 * (x + 0.001 * z) + 0.01,
                y=1.2 * (y + 0.01 * z) + 0.001,
            ),
            ("x", "y"),
        ),
    ],
)
def test_regrid_conservative_2d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
):
    result = na.regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )

    if axis_output is None:
        axis_output = tuple(coordinates_output.shape)
    elif isinstance(axis_output, str):
        axis_output = (axis_output, )

    shape_result = coordinates_output.shape
    shape_result = {
        a: shape_result[a] - 1 if a in axis_output
        else shape_result[a]
        for a in shape_result
    }

    assert np.issubdtype(result.dtype, float)
    assert result.shape == shape_result
    assert np.allclose(result.sum(), values_input.sum())

@pytest.mark.parametrize(
    argnames="coordinates_input, values_input, axis_input, coordinates_output, axis_output",
    argvalues=[
        (
                na.Cartesian2dVectorArray(x, y),
                na.random.normal(0, 1, shape_random=shape_centers),
                None,
                na.Cartesian2dVectorArray(
                    x=1.1 * x + 0.01,
                    y=1.2 * y + 0.01,
                ),
                None,
        ),
        (
                na.Cartesian2dVectorArray(
                    x=x + 0.01 * z,
                    y=y + 0.01 * z,
                ),
                na.random.normal(0, 1, shape_random=shape_centers | z.shape),
                ("x", "y"),
                na.Cartesian2dVectorArray(
                    x=1.1 * (x + 0.001 * z) + 0.01,
                    y=1.2 * (y + 0.01 * z) + 0.001,
                ),
                ("x", "y"),
        ),
    ],
)
def test_transpose_weights(
        coordinates_input: tuple[np.ndarray, ...],
        coordinates_output: tuple[np.ndarray, ...],
        values_input: np.ndarray,
        axis_input: None | int | tuple[int, ...],
        axis_output: None | int | tuple[int, ...],
):

    weights = na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )

    data = na.regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
    )

    transposed_weights = na.regridding.transpose_weights(weights)

    reversed_data = na.regridding.regrid_from_weights(
        *transposed_weights,
        values_input=data,
    )

    assert values_input.shape == reversed_data.shape
