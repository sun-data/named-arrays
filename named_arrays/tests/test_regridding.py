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
    argnames="coordinates_input, values_input, axis_input, "
    "coordinates_output, axis_output, weights_input",
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
            1,
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
            1,
        ),
        (
            # distinct output axis names with a per-input-cell ``weights_input``
            na.Cartesian2dVectorArray(x, y),
            na.random.normal(0, 1, shape_random=shape_centers),
            ("x", "y"),
            na.Cartesian2dVectorArray(
                x=1.1 * na.linspace(-1, 1, axis="x_new", num=shape_vertices["x"]) + 0.01,
                y=1.2 * na.linspace(-1, 1, axis="y_new", num=shape_vertices["y"]) + 0.01,
            ),
            ("x_new", "y_new"),
            na.random.uniform(0.5, 1.5, shape_random=shape_centers),
        ),
    ],
)
def test_regrid_conservative_2d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    weights_input: int | na.AbstractScalar,
):
    result = na.regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )

    weights = na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=1,
        method="conservative",
    )
    result2 = na.regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
    )

    assert np.allclose(result, result2, atol=1e-6)

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

    # a non-scalar ``weights_input`` is applied per *input* cell, so passing it
    # to ``weights`` must be equivalent to folding it into the input values
    # before regridding.  ``perturb=False`` keeps the two geometric weights
    # identical so the results can be compared exactly.
    kwargs_weights = dict(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
        perturb=False,
    )
    result_weighted = na.regridding.regrid_from_weights(
        *na.regridding.weights(weights_input=weights_input, **kwargs_weights),
        values_input=values_input,
    )
    result_folded = na.regridding.regrid_from_weights(
        *na.regridding.weights(**kwargs_weights),
        values_input=values_input * weights_input,
    )
    assert np.allclose(result_weighted, result_folded)


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
    coordinates_input: na.AbstractVectorArray,
    coordinates_output: na.AbstractVectorArray,
    values_input: na.AbstractScalarArray,
    axis_input: None | str | tuple[str, ...],
    axis_output: None | str | tuple[str, ...],
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


@pytest.mark.parametrize(
    argnames="coordinates_input,"
    "values_input,"
    "axis_input,"
    "coordinates_output,"
    "axis_output,"
    "weights_input",
    argvalues=[
        (
            na.Cartesian2dVectorArray(x, y),
            na.random.uniform(0, 1, shape_random=shape_centers),
            None,
            na.Cartesian2dVectorArray(x, y),
            None,
            1,
        ),
        (
            na.Cartesian2dVectorArray(
                x=x + 0.01 * z,
                y=y + 0.01 * z,
            ),
            na.random.uniform(0, 1, shape_random=shape_centers | z.shape),
            ("x", "y"),
            na.Cartesian2dVectorArray(
                x=x + 0.01 * z,
                y=y + 0.01 * z,
            ),
            ("x", "y"),
            None,
        ),
        (
            x,
            na.random.uniform(0, 1, shape_random=shape_centers),
            "x",
            x,
            "x",
            None,
        )
    ],
)
def test_transpose_weights_conservative(
    coordinates_input: na.AbstractVectorArray,
    coordinates_output: na.AbstractVectorArray,
    values_input: na.AbstractScalarArray,
    axis_input: None | str | tuple[str, ...],
    axis_output: None | str | tuple[str, ...],
    weights_input: None | na.AbstractScalarArray,
):

    weights = na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
        weights_input=weights_input,
    )

    data = na.regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
    )

    transposed_weights = na.regridding.transpose_weights_conservative(
        weights=weights,
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
    )

    reversed_data = na.regridding.regrid_from_weights(
        *transposed_weights,
        values_input=data,
    )

    assert values_input.shape == reversed_data.shape

    assert np.allclose(values_input.sum(axis_input), data.sum(axis_output))
    assert np.allclose(data.sum(axis_output), reversed_data.sum(axis_output))


def test_weights_seed():
    """
    A swept conservative build perturbs the output grid to break degenerate
    overlaps, so its result is only reproducible if that perturbation is seeded.

    The grid is sheared so that it is not axis-aligned. `regridding` 3.4 resamples
    a uniform, axis-aligned lattice with a clipping kernel that resolves
    degeneracies geometrically and so is never perturbed; that case is covered by
    :func:`test_weights_seed_unperturbed` below.
    """
    kwargs = dict(
        coordinates_input=na.Cartesian2dVectorArray(x, y),
        coordinates_output=na.Cartesian2dVectorArray(
            x=1.1 * x + 0.01 + 0.05 * y,
            y=1.2 * y + 0.01,
        ),
        values_input=na.random.normal(0, 1, shape_random=shape_centers),
        method="conservative",
    )

    result = na.regridding.regrid(**kwargs)
    result_expected = na.regridding.regrid(**kwargs)
    assert np.all(result == result_expected)

    # a different seed moves the result, but only in the last few digits
    result_seed = na.regridding.regrid(seed=1, **kwargs)
    assert not np.all(result_seed == result)
    assert np.allclose(result_seed, result, atol=1e-6)

    # an unseeded generator draws a fresh perturbation for every call
    result_none = na.regridding.regrid(seed=None, **kwargs)
    result_none_expected = na.regridding.regrid(seed=None, **kwargs)
    assert not np.all(result_none == result_none_expected)

    # the seed is inert if the grid is not perturbed
    result_unperturbed = na.regridding.regrid(perturb=False, seed=0, **kwargs)
    result_unperturbed_expected = na.regridding.regrid(perturb=False, seed=1, **kwargs)
    assert np.all(result_unperturbed == result_unperturbed_expected)


def test_weights_seed_unperturbed():
    """
    A uniform, axis-aligned output lattice is resampled by the clipping kernel
    added in `regridding` 3.4, which resolves degenerate overlaps geometrically.
    Nothing is perturbed, so the seed cannot change the result and every call
    already reproduces.
    """
    kwargs = dict(
        coordinates_input=na.Cartesian2dVectorArray(x, y),
        coordinates_output=na.Cartesian2dVectorArray(
            x=1.1 * x + 0.01,
            y=1.2 * y + 0.01,
        ),
        values_input=na.random.normal(0, 1, shape_random=shape_centers),
        method="conservative",
    )

    result = na.regridding.regrid(**kwargs)

    assert np.all(na.regridding.regrid(seed=1, **kwargs) == result)
    assert np.all(na.regridding.regrid(seed=None, **kwargs) == result)
