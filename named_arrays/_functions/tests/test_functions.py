import pytest
import named_arrays as na
import named_arrays.tests.test_core

__all__ = [
    "AbstractTestAbstractFunctionArray",
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_distribution = named_arrays.tests.test_core.num_distribution


def _function_arrays():
    functions_1d = [
        na.FunctionArray(
            inputs=na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            outputs=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x))
        )
    ]

    inputs_2d = [
        na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
        na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=1,
            axis=na.Cartesian2dVectorArray('x', 'y'),
            num=na.Cartesian2dVectorArray(_num_x, _num_y)
        )
    ]

    outputs_2d = [
        na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y))
    ]

    functions_2d = [
        na.FunctionArray(
            inputs=inputs,
            outputs=outputs,
        )
        for inputs in inputs_2d
        for outputs in outputs_2d
    ]

    functions = functions_1d + functions_2d

    return functions


class AbstractTestAbstractFunctionArray(
    named_arrays.tests.test_core.AbstractTestAbstractArray,
):

    def test_inputs(self, array: na.AbstractFunctionArray):
        assert isinstance(array.inputs, na.AbstractArray)

    def test_outputs(self, array: na.AbstractFunctionArray):
        assert isinstance(array.outputs, na.AbstractArray)


@pytest.mark.parametrize("array", _function_arrays())
class TestFunctionArray(
    AbstractTestAbstractFunctionArray,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray,
):
    pass
