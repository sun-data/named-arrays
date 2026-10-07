"""
Indexing a function array whose inputs or outputs, or a component of either,
have a single element along an axis where the others have more.
"""

import pytest
import numpy as np
import named_arrays as na

_num = 10


def _function_arrays() -> list[na.FunctionArray]:
    center = na.linspace(0, 1, axis="x", num=_num)
    vertex = na.linspace(0, 1, axis="x", num=_num + 1)
    outputs = na.arange(0, _num, axis="x")
    single = na.ScalarArray(np.array([2.0]), axes="x")
    return [
        # inputs with a single element along a center axis
        na.FunctionArray(inputs=single, outputs=outputs),
        # outputs with a single element along a center axis, as after a
        # reduction which keeps the reduced axis
        na.FunctionArray(inputs=center, outputs=single + na.arange(0, 3, axis="y")),
        # a component of the inputs with a single element along a center axis
        na.FunctionArray(
            inputs=na.Cartesian2dVectorArray(x=center, y=single),
            outputs=outputs,
        ),
        # a component of the inputs with a single element along a vertex axis
        na.FunctionArray(
            inputs=na.Cartesian2dVectorArray(x=vertex, y=single),
            outputs=outputs,
        ),
        # a component of the outputs with a single element along a vertex axis
        na.FunctionArray(
            inputs=vertex,
            outputs=na.Cartesian2dVectorArray(x=outputs, y=single),
        ),
    ]


@pytest.mark.parametrize("array", _function_arrays())
@pytest.mark.parametrize(
    argnames="item",
    argvalues=[
        dict(x=slice(2, 5)),
        dict(x=slice(2, 4)),
        dict(x=slice(5, 2)),
        dict(x=slice(None, None, -1)),
        dict(x=3),
        dict(x=-1),
        dict(x=slice(2, 5), y=1),
    ],
)
def test__getitem__(
    array: na.FunctionArray,
    item: dict[str, int | slice],
) -> None:
    """
    Indexing a function array gives the same function as indexing the
    function with its inputs and outputs broadcast against each other, and a
    sliced axis stays a center or a vertex axis.
    """
    result = array[item]
    expected = array.broadcasted[item]
    assert result.shape == expected.shape
    assert result.axes_center == expected.axes_center
    assert result.axes_vertex == expected.axes_vertex
    if isinstance(item["x"], slice):
        assert result.axes_vertex == array.axes_vertex
    # `broadcasted` also broadcasts the inputs along the axes of the outputs
    assert result.inputs.shape.get("x") == expected.inputs.shape.get("x")
    assert result.outputs.shape.get("x") == expected.outputs.shape.get("x")
    assert np.all(result.inputs == expected.inputs)
    assert np.all(result.outputs == expected.outputs)


@pytest.mark.parametrize(
    argnames="item",
    argvalues=[
        dict(x=na.ScalarArray(np.array([7, 2]), axes="x")),
        dict(x=na.ScalarArray(np.array([7, 2]), axes="z")),
    ],
)
def test__getitem__array_center(
    item: dict[str, na.AbstractArray],
) -> None:
    """
    An index array along a center axis selects the same elements from inputs
    with a single element along it as from the outputs.
    """
    array = na.FunctionArray(
        inputs=na.ScalarArray(np.array([2.0]), axes="x"),
        outputs=na.arange(0, _num, axis="x"),
    )
    result = array[item]
    expected = array.broadcasted[item]
    assert result.shape == expected.shape
    assert result.inputs.shape == expected.inputs.shape
    assert np.all(result.inputs == expected.inputs)
    assert np.all(result.outputs == expected.outputs)
