import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

from . import test_functions
from . import test_functions_vertices

__all__ = []


# All FunctionArray fixture sources -- center, vertex, and polynomial-fit --
# combined so the shape-contract test exercises every family in one place.
_fixtures = (
    test_functions._function_arrays()
    + test_functions._polynomial_function_arrays()
    + test_functions_vertices._function_arrays()
)


def _axis_and_component(
    array: na.AbstractFunctionArray,
) -> tuple[str, None | str]:
    """
    An axis to differentiate along and the variable to differentiate against.

    The variable has to vary along the axis, so for a vector input this looks
    for the component which does.
    """
    inputs = array.inputs

    if not isinstance(inputs, na.AbstractVectorArray):
        return next(iter(na.shape(inputs))), None

    return next(
        (axis, component)
        for component, value in inputs.explicit.components.items()
        for axis in na.shape(value)
    )


@pytest.mark.parametrize("array", _fixtures)
def test_gradient_shape_contract(array: na.AbstractFunctionArray):
    """The inputs are untouched and the outputs keep their shape."""
    array = array.explicit

    axis, component = _axis_and_component(array)

    result = array.gradient(axis, component=component)

    assert isinstance(result, na.FunctionArray)
    assert np.all(result.inputs == array.inputs)
    assert result.outputs.shape == array.broadcasted.outputs.shape
    assert result.outputs.type_abstract == array.outputs.type_abstract


@pytest.mark.parametrize("num", [3, 11])
@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_constant_center(num: int, edge_order: int):
    """The derivative of a constant is zero everywhere."""
    f = na.FunctionArray(
        inputs=na.ScalarLinearSpace(0, 2, axis="x", num=num) * u.nm,
        outputs=na.ScalarArray.full(dict(x=num), 3.0) * u.ph,
    )
    result = f.gradient("x", edge_order=edge_order)
    assert np.allclose(result.outputs, 0 * u.ph / u.nm)


@pytest.mark.parametrize("num", [3, 11])
@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_linear_center(num: int, edge_order: int):
    """Central and one-sided differences are both exact for a straight line."""
    x = na.ScalarLinearSpace(0, 2, axis="x", num=num) * u.nm
    f = na.FunctionArray(inputs=x, outputs=3 * x / u.nm * u.ph)
    result = f.gradient("x", edge_order=edge_order)
    assert np.allclose(result.outputs, 3 * u.ph / u.nm)


def test_gradient_quadratic_edge_order():
    """
    Second-order edges are exact for a parabola where first-order edges are not.

    The interior is second order either way, so only the two ends differ.
    """
    x = na.ScalarLinearSpace(0, 2, axis="x", num=5) * u.nm
    f = na.FunctionArray(inputs=x, outputs=np.square(x / u.nm) * u.ph)

    expected = 2 * x / u.nm * u.ph / u.nm

    first_order = f.gradient("x", edge_order=1).outputs
    second_order = f.gradient("x", edge_order=2).outputs

    assert np.allclose(second_order, expected)
    assert not np.allclose(first_order, expected)

    # they agree away from the two ends
    interior = {"x": slice(1, ~0)}
    assert np.allclose(first_order[interior], expected[interior])


@pytest.mark.parametrize("edge_order", [1, 2])
@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.linspace(0, 2, num=7),
        # unevenly spaced samples, where the central difference weights each
        # neighbor by the gap on the opposite side
        np.array([0.0, 0.1, 0.4, 0.9, 1.1, 1.8, 2.0]),
    ],
    ids=["even", "uneven"],
)
def test_gradient_matches_numpy(x: np.ndarray, edge_order: int):
    """For coordinates along one axis this is what :func:`numpy.gradient` gives."""
    outputs = np.sin(x)

    f = na.FunctionArray(
        inputs=na.ScalarArray(x, axes=("x",)) * u.nm,
        outputs=na.ScalarArray(outputs, axes=("x",)) * u.ph,
    )

    result = np.gradient(f, axis="x", edge_order=edge_order)
    expected = np.gradient(outputs, x, edge_order=edge_order)

    assert np.allclose(result.outputs.ndarray, expected * u.ph / u.nm)


def test_gradient_vertex():
    """
    A vertex axis differentiates against the cell centers.

    The inputs are bin edges, so there is one more of them than there are
    outputs, and the outputs live at the centers between them.
    """
    num = 11
    f = na.FunctionArray(
        inputs=na.ScalarLinearSpace(0, 2, axis="x", num=num + 1) * u.nm,
        # a straight line in the cell centers, which are spaced by 2/11 nm
        outputs=na.ScalarArray(np.arange(num, dtype=float), axes=("x",)) * u.ph,
    )

    assert f.axes_vertex == ("x",)

    result = f.gradient("x")

    # the edges are untouched, and the derivative sits on the centers
    assert result.inputs.shape == {"x": num + 1}
    assert result.outputs.shape == {"x": num}
    assert np.allclose(result.outputs, 1 * u.ph / ((2 / num) * u.nm))


def test_gradient_component():
    """A vector input names the component to differentiate against."""
    num_x, num_y = 7, 5
    inputs = na.Cartesian2dVectorLinearSpace(
        start=0,
        stop=2,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=na.Cartesian2dVectorArray(num_x, num_y),
    )
    f = na.FunctionArray(
        inputs=inputs,
        outputs=3 * inputs.explicit.x + 5 * inputs.explicit.y,
    )

    assert np.allclose(f.gradient("x", component="x").outputs, 3)
    assert np.allclose(f.gradient("y", component="y").outputs, 5)

    # the inputs are carried through untouched
    assert isinstance(f.gradient("x", component="x").inputs, na.AbstractVectorArray)


def test_gradient_component_required():
    """Without a component, a vector input has no one differentiation variable."""
    inputs = na.Cartesian2dVectorLinearSpace(
        start=0,
        stop=2,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=na.Cartesian2dVectorArray(7, 5),
    )
    f = na.FunctionArray(inputs=inputs, outputs=inputs.explicit.x)

    with pytest.raises(ValueError, match="must be a scalar"):
        f.gradient("x")

    with pytest.raises(ValueError, match="must be a scalar"):
        np.gradient(f, axis="x")


@pytest.mark.parametrize(
    argnames="outputs",
    argvalues=[
        # the axis is absent, so slicing it is a no-op and the arithmetic
        # would broadcast on its own
        na.ScalarArray(3.0) * u.ph,
        # the axis is present with length one, where slicing gives an empty
        # array and only the explicit broadcast saves it
        na.ScalarArray(np.array([3.0]), axes=("x",)) * u.ph,
    ],
    ids=["absent", "length one"],
)
def test_gradient_constant_outputs(outputs: na.AbstractScalar):
    """
    Outputs which do not vary along the axis are broadcast against it.

    The derivative of a constant is zero, and it is zero at every point along
    the axis rather than at the single point the outputs were stored at.
    """
    num = 5
    f = na.FunctionArray(
        inputs=na.linspace(0, 2, axis="x", num=num) * u.nm,
        outputs=outputs,
    )

    assert na.shape(f.outputs).get("x", 1) != num

    result = f.gradient("x")

    assert result.outputs.shape == {"x": num}
    assert np.allclose(result.outputs, 0 * u.ph / u.nm)


@pytest.mark.parametrize(
    argnames="dtype",
    argvalues=[np.int32, np.int64, np.uint16],
)
def test_gradient_integer_coordinates(dtype: type):
    """
    An integer coordinate is differentiated in floating point.

    The gaps, their squares, and their product overflow in a narrow integer
    type, which would corrupt the interior of the result without any warning,
    so the variable is promoted first, as :func:`numpy.gradient` does.
    """
    # descending for the unsigned case, where a wrapped subtraction would
    # flip the sign of the whole derivative rather than only disturb it
    gap = 2000
    if np.issubdtype(dtype, np.unsignedinteger):
        x = (np.arange(3, -1, -1) * gap).astype(dtype)
    else:
        x = (np.arange(4) * gap).astype(dtype)

    outputs = np.array([0.0, 1.0, 4.0, 9.0])

    f = na.FunctionArray(
        inputs=na.ScalarArray(x, axes=("x",)),
        outputs=na.ScalarArray(outputs, axes=("x",)),
    )

    result = f.gradient("x")
    expected = np.gradient(outputs, x.astype(float))

    assert np.allclose(result.outputs.ndarray, expected)


def test_gradient_float32_coordinates_keep_their_dtype():
    """Only an integer variable is promoted, so a narrow float is left alone."""
    x = na.ScalarArray(np.arange(4, dtype=np.float32), axes=("x",))
    f = na.FunctionArray(inputs=x, outputs=x)

    assert f.gradient("x").outputs.dtype == np.float32


def test_gradient_axis_must_be_one_name():
    """
    A sequence of axes belongs to :func:`numpy.gradient`, not to the method.

    Differentiating along several axes gives one result per axis, where this
    method returns a single array, so a sequence is refused rather than
    quietly treated as a name.
    """
    x = na.linspace(0, 2, axis="x", num=5) * u.nm
    f = na.FunctionArray(inputs=x, outputs=x)

    with pytest.raises(TypeError, match="must be the name of a single axis"):
        f.gradient(("x",))

    # the same call through numpy is the supported spelling
    assert isinstance(np.gradient(f, axis=("x",)), tuple)


def test_gradient_result_has_its_own_inputs():
    """
    The result is not the source's inputs under another name.

    The inputs are unchanged by the derivative, but handing back the very
    same object would let a write through one reach the other, which no
    neighboring operation does.
    """
    x = na.ScalarArray(np.arange(5.0), axes=("x",)) * u.nm
    f = na.FunctionArray(inputs=x, outputs=na.ScalarArray(np.arange(5.0), axes=("x",)) * u.ph)

    result = f.gradient("x")

    assert result.inputs is not f.inputs
    assert np.all(result.inputs == f.inputs)


def test_gradient_distorted_grid():
    """
    The differentiation variable may vary along axes other than its own.

    This is what a distorted grid gives, and it is not something
    :func:`numpy.gradient` can express, since it takes the coordinates along
    an axis as a one-dimensional array.
    """
    num_x, num_y = 9, 4

    # each row is shifted along `x` by a different amount, and stretched
    shift = na.ScalarArray(np.array([0.0, 1.0, 2.0, 3.0]), axes=("y",))
    scale = na.ScalarArray(np.array([1.0, 2.0, 3.0, 4.0]), axes=("y",))
    x = scale * na.linspace(0, 2, axis="x", num=num_x) + shift

    f = na.FunctionArray(inputs=x * u.nm, outputs=(3 * x) * u.ph)

    result = f.gradient("x")

    assert result.outputs.shape == {"x": num_x, "y": num_y}
    # the slope is three per unit of `x` on every row, however it is stretched
    assert np.allclose(result.outputs, 3 * u.ph / u.nm)


def test_gradient_uncertain_outputs():
    """The nominal value and the distribution are differentiated alike."""
    num = 7
    x = na.ScalarLinearSpace(0, 2, axis="x", num=num) * u.nm
    outputs = na.NormalUncertainScalarArray(
        nominal=3 * x / u.nm * u.ph,
        width=0.1 * u.ph,
        num_distribution=5,
    )
    f = na.FunctionArray(inputs=x, outputs=outputs)

    result = f.gradient("x")

    assert isinstance(result.outputs, na.AbstractUncertainScalarArray)
    assert np.allclose(result.outputs.nominal, 3 * u.ph / u.nm)
    assert "_distribution" in result.outputs.distribution.shape


def test_gradient_multiple_axes():
    """A sequence of axes gives one array per axis, as :mod:`numpy` does."""
    inputs = na.Cartesian2dVectorLinearSpace(
        start=0,
        stop=2,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=na.Cartesian2dVectorArray(7, 5),
    )
    f = na.FunctionArray(inputs=inputs.explicit.x, outputs=3 * inputs.explicit.x)

    result = np.gradient(f, axis=("x",))

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert np.allclose(result[0].outputs, np.gradient(f, axis="x").outputs)


def test_gradient_rejects_spacing():
    """The spacing of a function array is its inputs, so it cannot be given."""
    x = na.ScalarLinearSpace(0, 2, axis="x", num=5) * u.nm
    f = na.FunctionArray(inputs=x, outputs=x)

    with pytest.raises(ValueError, match="spacing argument does not apply"):
        np.gradient(f, 1 * u.nm, axis="x")


def test_gradient_requires_an_axis():
    """Unlike :mod:`numpy`, there is no positional order to differentiate along."""
    x = na.ScalarLinearSpace(0, 2, axis="x", num=5) * u.nm
    f = na.FunctionArray(inputs=x, outputs=x)

    with pytest.raises(ValueError, match="`axis` is required"):
        np.gradient(f)


def test_gradient_invalid():
    x = na.ScalarLinearSpace(0, 2, axis="x", num=5) * u.nm
    f = na.FunctionArray(inputs=x, outputs=x)

    with pytest.raises(ValueError, match="must be a member of"):
        f.gradient("z")

    with pytest.raises(ValueError, match="must be either 1 or 2"):
        f.gradient("x", edge_order=3)


def test_gradient_variable_missing_axis():
    """The variable must vary along the axis for the derivative to be defined."""
    f = na.FunctionArray(
        inputs=na.ScalarLinearSpace(0, 2, axis="x", num=5) * u.nm,
        outputs=na.ScalarArray(np.ones((5, 3)), axes=("x", "y")) * u.ph,
    )

    with pytest.raises(ValueError, match="does not vary along"):
        f.gradient("y")


@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_too_few_points(edge_order: int):
    """Each edge order needs enough points to build its one-sided difference."""
    num = edge_order
    x = na.ScalarLinearSpace(0, 2, axis="x", num=num) * u.nm
    f = na.FunctionArray(inputs=x, outputs=x)

    with pytest.raises(ValueError, match="points are required"):
        f.gradient("x", edge_order=edge_order)
