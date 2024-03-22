import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from . import test_vectors
from ..cartesian.tests import test_vectors_cartesian

_num_x = test_vectors._num_x
_num_y = test_vectors._num_y
_num_z = test_vectors._num_z
_num_distribution = test_vectors._num_distribution


def _input_output_arrays() -> list[na.InputOutputVectorArray]:
    return [
        na.InputOutputVectorArray(
            input=0,
            output=1,
        ),
        na.InputOutputVectorArray(
            input=na.linspace(0, 1, axis="y", num=_num_y),
            output=2,
        ),
    ]


def _input_output_arrays_2() -> list[na.InputOutputVectorArray]:
    return [
        na.InputOutputVectorArray(
            input=3,
            output=4,
        ),
        na.InputOutputVectorArray(
            input=na.NormalUncertainScalarArray(100, width=1),
            output=5,
        )
    ]


def _input_output_items() -> list[na.AbstractInputOutputVectorArray | dict[str, int | slice | na.AbstractInputOutputVectorArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
    ]


class AbstractTestAbstractInputOutputVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):
    def test_input(self, array: na.AbstractInputOutputVectorArray):
        assert isinstance(na.as_named_array(array.input), na.AbstractArray)

    def test_output(self, array: na.AbstractInputOutputVectorArray):
        assert isinstance(na.as_named_array(array.output), na.AbstractArray)

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_input_output_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractInputOutputVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _input_output_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _input_output_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors.AbstractTestAbstractVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _input_output_arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions
        ):
            pass

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
            ]
        )
        class TestReductionFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                .25,
                25 * u.percent,
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
            ]
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
            .TestPercentileLikeFunctions,
        ):
            pass

    class TestNamedArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions,
    ):
        @pytest.mark.skip
        class TestPltPlotLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass


@pytest.mark.parametrize("array", _input_output_arrays())
class TestInputOutputVectorArray(
    AbstractTestAbstractInputOutputVectorArray,
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArray,
):
    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(y=slice(None)),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            700,
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


class AbstractTestAbstractImplicitInputOutputVectorArray(
    AbstractTestAbstractInputOutputVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractParameterizedInputOutputVectorArray(
    AbstractTestAbstractImplicitInputOutputVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


class AbstractTestAbstractInputOutputVectorSpace(
    AbstractTestAbstractParameterizedInputOutputVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


def _input_output_linear_spaces() -> list[na.InputOutputVectorLinearSpace]:
    return [
        na.InputOutputVectorLinearSpace(400 * u.nm, 600 * u.nm, axis="y", num=_num_y)
    ]


@pytest.mark.parametrize("array", _input_output_linear_spaces())
class TestInputOutputVectorLinearSpace(
    AbstractTestAbstractInputOutputVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass
