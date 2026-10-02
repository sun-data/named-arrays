from typing import Any
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

shape_vertices = dict(x=10, y=11)
shape_centers = {a: shape_vertices[a] - 1 for a in shape_vertices}

x = na.linspace(-1, 1, axis="x", num=shape_vertices["x"])
y = na.linspace(-1, 1, axis="y",  num=shape_vertices["y"])
z = na.linspace(-1, 1, axis="z", num=3)

x_new = na.linspace(-1, 1, axis="x_new", num=5)
y_new = na.linspace(-1, 1, axis="y_new", num=6)


def _cuda_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:  # pragma: nocover
        return False


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


def test_regrid_from_weights_quantity():
    """
    Values with a unit are resampled as plain floats and the unit is restored
    on the result, which is what a device kernel needs and what the host path
    accepted already.
    """
    kwargs = dict(
        coordinates_input=na.Cartesian2dVectorArray(x, y),
        coordinates_output=na.Cartesian2dVectorArray(
            x=1.1 * x + 0.01, y=1.2 * y + 0.01
        ),
        axis_input=("x", "y"),
        axis_output=("x", "y"),
        method="conservative",
    )
    weights, shape_input, shape_output = na.regridding.weights(**kwargs)
    values = na.random.normal(0, 1, shape_random=shape_centers)
    result = na.regridding.regrid_from_weights(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values * u.erg,
    )
    result_plain = na.regridding.regrid_from_weights(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values,
    )
    assert na.unit(result) == u.erg
    assert np.all(na.value(result) == result_plain)


def test_regrid_from_weights_quantity_weights():
    """
    Weights built from a ``weights_input`` with a unit carry that unit onto
    the result, multiplied by the unit of the values, and a dimensionless
    ``weights_input`` leaves the unit of the values alone.
    """
    kwargs = dict(
        coordinates_input=na.Cartesian2dVectorArray(x, y),
        coordinates_output=na.Cartesian2dVectorArray(
            x=1.1 * x + 0.01, y=1.2 * y + 0.01
        ),
        axis_input=("x", "y"),
        axis_output=("x", "y"),
        method="conservative",
    )
    values = na.random.normal(0, 1, shape_random=shape_centers)
    weights_input = na.random.uniform(0.5, 1.5, shape_random=shape_centers)
    result_plain = na.regridding.regrid_from_weights(
        *na.regridding.weights(weights_input=weights_input, **kwargs),
        values_input=values,
    )
    for unit_weights, unit_expected in (
        (u.dimensionless_unscaled, u.erg),
        (u.mm, u.erg * u.mm),
    ):
        result = na.regridding.regrid_from_weights(
            *na.regridding.weights(
                weights_input=weights_input * unit_weights,
                **kwargs,
            ),
            values_input=values * u.erg,
        )
        assert na.unit(result) == unit_expected
        assert np.allclose(na.value(result), result_plain)

    # values without a unit take the unit of the weights alone
    result = na.regridding.regrid_from_weights(
        *na.regridding.weights(weights_input=weights_input * u.mm, **kwargs),
        values_input=values,
    )
    assert na.unit(result) == u.mm
    assert np.allclose(na.value(result), result_plain)


def test_weights_device_host():
    """The host is the default device, so asking for it changes nothing."""
    kwargs = dict(
        coordinates_input=na.Cartesian2dVectorArray(x, y),
        coordinates_output=na.Cartesian2dVectorArray(
            x=1.1 * x + 0.01, y=1.2 * y + 0.01
        ),
        axis_input=("x", "y"),
        axis_output=("x", "y"),
        method="conservative",
    )
    values = na.random.normal(0, 1, shape_random=shape_centers)
    result = na.regridding.regrid_from_weights(
        *na.regridding.weights(**kwargs),
        values_input=values,
    )
    result_host = na.regridding.regrid_from_weights(
        *na.regridding.weights(device=None, **kwargs),
        values_input=values,
    )
    assert np.all(result_host == result)


@pytest.mark.skipif(
    not _cuda_available(),
    reason="a CUDA device is needed to build weights on one",
)
def test_weights_device_cuda():  # pragma: nocover
    """Weights built on a device apply to the same result as the host build."""
    import warnings

    from numba.core.errors import NumbaPerformanceWarning

    # the test grid is tiny, which numba flags as an under-used GPU
    warnings.simplefilter("ignore", NumbaPerformanceWarning)
    kwargs = dict(
        coordinates_input=na.Cartesian2dVectorArray(x, y),
        coordinates_output=na.Cartesian2dVectorArray(
            x=1.1 * x + 0.01, y=1.2 * y + 0.01
        ),
        axis_input=("x", "y"),
        axis_output=("x", "y"),
        method="conservative",
    )
    values = na.random.normal(0, 1, shape_random=shape_centers)
    host = na.regridding.regrid_from_weights(
        *na.regridding.weights(**kwargs),
        values_input=values,
    )
    device = na.regridding.regrid_from_weights(
        *na.regridding.weights(device="cuda", **kwargs),
        values_input=values,
    )
    device = na.ScalarArray(np.asarray(device.ndarray.copy_to_host()), axes=device.axes)
    assert np.allclose(device, host)


grid_rotated = na.Cartesian2dRotationMatrixArray(
    na.linspace(0.1, 0.7, axis="channel", num=2) * u.rad
) @ na.Cartesian2dVectorLinearSpace(
    start=-4,
    stop=4,
    axis=na.Cartesian2dVectorArray("x", "y"),
    num=17,
).explicit
"""A square grid of vertices, rotated by two angles along an orthogonal axis."""

grid_lattice = na.Cartesian2dVectorLinearSpace(
    start=na.Cartesian2dVectorArray(-6, -6),
    stop=na.Cartesian2dVectorArray(6, 5),
    axis=na.Cartesian2dVectorArray("x_new", "y_new"),
    num=na.Cartesian2dVectorArray(25, 23),
)
"""A uniform lattice of vertices, larger than `grid_rotated`."""

weights_rotated = na.regridding.weights(
    coordinates_input=grid_rotated,
    coordinates_output=grid_lattice,
    axis_input=("x", "y"),
    axis_output=("x_new", "y_new"),
    method="conservative",
)

values_rotated = na.random.uniform(0, 1, shape_random=dict(channel=2, x=16, y=16), seed=1)


def _shift(a: na.ScalarArray, axis: str, offset: int) -> na.ScalarArray:
    """Move the contents of `a` by `offset` cells along `axis`, filling with zeros."""
    num = a.shape[axis]
    result = 0 * a
    if offset >= 0:
        result[{axis: slice(offset, num)}] = a[{axis: slice(0, num - offset)}]
    else:
        result[{axis: slice(0, num + offset)}] = a[{axis: slice(-offset, num)}]
    return result


def _convolve_reference(
    image: na.ScalarArray,
    kernel: na.FunctionArray,
    axis: tuple[str, ...],
) -> na.ScalarArray:
    """
    Convolve an image with a kernel by shifting it once for each element of
    the kernel, independently of how `convolve_weights` builds its stencil.
    """
    offsets = kernel.inputs
    if isinstance(offsets, na.AbstractVectorArray):
        components = list(offsets.cartesian_nd.components.values())
    else:
        components = [offsets]
    components = [na.as_named_array(na.value(c)) for c in components]
    shape_kernel = na.broadcast_shapes(*[c.shape for c in components])

    result = 0 * image
    for index in na.ndindex(shape_kernel):
        contribution = image * kernel.outputs[index]
        for a, c in zip(axis, components):
            contribution = _shift(contribution, a, int(np.rint(c[index].ndarray)))
        result = result + contribution
    return result


def _kernel(
    offset_x: na.AbstractScalar,
    offset_y: na.AbstractScalar,
    shape_extra: None | dict[str, int] = None,
    seed: int = 2,
) -> na.FunctionArray:
    """A random kernel at the given offsets, which may vary along other axes."""
    offsets = na.Cartesian2dVectorArray(offset_x, offset_y)
    shape = na.broadcast_shapes(offsets.shape, shape_extra or dict())
    return na.FunctionArray(
        inputs=offsets,
        outputs=na.random.uniform(0, 1, shape_random=shape, seed=seed),
    )


kx = na.arange(-1, 2, axis="kernel_x")
ky = na.arange(-2, 3, axis="kernel_y")


@pytest.mark.parametrize(
    argnames="kernel,axis_output",
    argvalues=[
        (_kernel(kx, ky), ("x_new", "y_new")),
        (_kernel(kx, ky), None),
        (_kernel(kx, ky), ("y_new", "x_new")),
        (_kernel(na.arange(0, 3, axis="kernel_x"), na.arange(-1, 1, axis="kernel_y")), None),
        (_kernel(kx * u.pix, ky * u.pix), None),
        (_kernel(na.linspace(-1, 1, axis="kernel_x", num=3), ky), None),
        (_kernel(kx, ky, dict(channel=2)), None),
        (_kernel(kx, ky, dict(x_new=24)), None),
        (_kernel(kx, ky, dict(channel=2, y_new=22)), None),
    ],
    ids=[
        "centered",
        "axes inferred",
        "axes reversed",
        "uncentered",
        "pixels",
        "float offsets",
        "per channel",
        "varying along x",
        "per channel varying along y",
    ],
)
def test_convolve_weights(
    kernel: na.FunctionArray,
    axis_output: None | tuple[str, ...],
) -> None:
    """Convolving the weights is the same as convolving what they produce."""
    result = na.regridding.convolve_weights(
        weights=weights_rotated,
        kernel=kernel,
        axis_output=axis_output,
    )

    assert result[0].shape == weights_rotated[0].shape
    assert result[1] == weights_rotated[1]
    assert result[2] == weights_rotated[2]

    actual = na.regridding.regrid_from_weights(*result, values_input=values_rotated)
    expected = _convolve_reference(
        image=na.regridding.regrid_from_weights(
            *weights_rotated,
            values_input=values_rotated,
        ),
        kernel=kernel,
        axis=axis_output or ("x_new", "y_new"),
    )

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_1d() -> None:
    """A scalar kernel convolves one-dimensional weights."""
    x_in = na.linspace(-1, 1, axis="x", num=21)
    x_out = na.linspace(-1.1, 1.1, axis="x_new", num=12)
    weights = na.regridding.weights(x_in, x_out, method="conservative")
    kernel = na.FunctionArray(
        inputs=na.arange(-1, 2, axis="kernel"),
        outputs=na.ScalarArray(np.array([0.25, 0.5, 0.25]), axes="kernel"),
    )
    values = na.random.uniform(0, 1, shape_random=dict(x=20), seed=3)

    result = na.regridding.convolve_weights(weights, kernel)

    actual = na.regridding.regrid_from_weights(*result, values_input=values)
    expected = _convolve_reference(
        image=na.regridding.regrid_from_weights(*weights, values_input=values),
        kernel=kernel,
        axis=("x_new",),
    )
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_new_axis() -> None:
    """A kernel which varies along an axis the weights lack adds that axis."""
    kernel = _kernel(kx, ky, dict(wavelength=3))

    result = na.regridding.convolve_weights(weights_rotated, kernel)

    assert result[0].shape == dict(channel=2, wavelength=3)
    assert result[1] == weights_rotated[1] | dict(wavelength=3)
    assert result[2] == weights_rotated[2] | dict(wavelength=3)

    actual = na.regridding.regrid_from_weights(*result, values_input=values_rotated)
    expected = _convolve_reference(
        image=na.regridding.regrid_from_weights(
            *weights_rotated,
            values_input=values_rotated,
        ),
        kernel=kernel,
        axis=("x_new", "y_new"),
    )
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_transpose() -> None:
    """
    The conservative transpose of the convolved weights is their adjoint:
    the input and output cells have the same area here, so the inner
    products of the forward and the transposed operation agree.
    """
    result = na.regridding.convolve_weights(weights_rotated, _kernel(kx, ky))
    transposed = na.regridding.transpose_weights_conservative(
        weights=result,
        coordinates_input=grid_rotated,
        coordinates_output=grid_lattice,
        axis_input=("x", "y"),
        axis_output=("x_new", "y_new"),
    )

    image = na.random.uniform(0, 1, shape_random=dict(channel=2, x_new=24, y_new=22), seed=4)

    forward = na.regridding.regrid_from_weights(*result, values_input=values_rotated)
    backward = na.regridding.regrid_from_weights(*transposed, values_input=image)

    assert np.allclose(
        (forward * image).sum(("x_new", "y_new")),
        (values_rotated * backward).sum(("x", "y")),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    argnames="kernel,axis_output,error",
    argvalues=[
        (_kernel(kx + 0.5, ky), None, ValueError),
        (_kernel(kx * u.mm, ky * u.mm), None, ValueError),
        (na.FunctionArray(na.Cartesian2dVectorArray(kx, ky), _kernel(kx, ky).outputs * u.mm), None, ValueError),
        (na.FunctionArray(kx, na.ScalarArray.ones(dict(kernel_x=3))), None, ValueError),
        (_kernel(kx, ky), ("x_new", "z_new"), ValueError),
        (_kernel(na.arange(-1, 2, axis="x_new"), ky), None, ValueError),
        (na.ScalarArray.ones(dict(kernel_x=3)), None, TypeError),
    ],
    ids=[
        "fractional offsets",
        "offsets in mm",
        "kernel in mm",
        "too few components",
        "unknown axis",
        "kernel axis is an output axis",
        "not a function",
    ],
)
def test_convolve_weights_errors(
    kernel: Any,
    axis_output: None | tuple[str, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        na.regridding.convolve_weights(weights_rotated, kernel, axis_output=axis_output)


@pytest.mark.skipif(
    not _cuda_available(),
    reason="a CUDA device is needed to build weights on one",
)
def test_convolve_weights_device_cuda() -> None:  # pragma: nocover
    """Weights on a device are convolved there, to the same result as the host."""
    import warnings

    from numba.core.errors import NumbaPerformanceWarning

    # the test grid is tiny, which numba flags as an under-used GPU
    warnings.simplefilter("ignore", NumbaPerformanceWarning)

    weights = na.regridding.weights(
        coordinates_input=grid_rotated,
        coordinates_output=grid_lattice,
        axis_input=("x", "y"),
        axis_output=("x_new", "y_new"),
        method="conservative",
        device="cuda",
    )
    kernel = _kernel(kx, ky, dict(channel=2))

    result = na.regridding.convolve_weights(weights, kernel)

    device = na.regridding.regrid_from_weights(*result, values_input=values_rotated)
    device = na.ScalarArray(np.asarray(device.ndarray.copy_to_host()), axes=device.axes)
    expected = _convolve_reference(
        image=na.regridding.regrid_from_weights(
            *na.regridding.weights(
                coordinates_input=grid_rotated,
                coordinates_output=grid_lattice,
                axis_input=("x", "y"),
                axis_output=("x_new", "y_new"),
                method="conservative",
            ),
            values_input=values_rotated,
        ),
        kernel=kernel,
        axis=("x_new", "y_new"),
    )
    assert np.allclose(device, expected, rtol=1e-10, atol=1e-12)
