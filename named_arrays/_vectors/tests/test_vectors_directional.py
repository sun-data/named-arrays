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


def _directional_arrays() -> list[na.DirectionalVectorArray]:
    return [
        na.DirectionalVectorArray(direction=na.Cartesian2dVectorArray(1, 2) * u.deg),
        na.DirectionalVectorArray(direction=na.Cartesian2dVectorLinearSpace(1, 2, axis="y", num=_num_y).explicit * u.deg),
    ]


def _directional_arrays_2() -> list[na.DirectionalVectorArray]:
    return [
        na.DirectionalVectorArray(direction=na.Cartesian2dVectorArray(3, 4) * u.deg),
        na.DirectionalVectorArray(direction=na.Cartesian2dVectorArray(
            x=na.NormalUncertainScalarArray(3, width=1) * u.deg,
            y=na.NormalUncertainScalarArray(4, width=1) * u.deg,
        ))
    ]


def _directional_items() -> list[na.DirectionalVectorArray | dict[str, int | slice | na.DirectionalVectorArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
    ]


class AbstractTestAbstractDirectionalVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):
    def test_direction(self, array: na.AbstractDirectionalVectorArray):
        assert isinstance(na.as_named_array(array.direction), (na.AbstractScalar, na.AbstractVectorArray))

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_directional_items(),
    )
    def test__getitem__(
            self,
            array: na.AbstractDirectionalVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _directional_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _directional_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _directional_arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions
        ):
            pass

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
            ]
        )
        class TestReductionFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                25 * u.percent,
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


@pytest.mark.parametrize("array", _directional_arrays())
class TestDirectionalVectorArray(
    AbstractTestAbstractDirectionalVectorArray,
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
            700 * u.nm,
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


class AbstractTestAbstractImplicitDirectionalVectorArray(
    AbstractTestAbstractDirectionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractParameterizedDirectionalVectorArray(
    AbstractTestAbstractImplicitDirectionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


class AbstractTestAbstractDirectionalVectorSpace(
    AbstractTestAbstractParameterizedDirectionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


def _directional_linear_spaces() -> list[na.DirectionalVectorLinearSpace]:
    return [
        na.DirectionalVectorLinearSpace(1 * u.m, 2 * u.m, axis="y", num=_num_y)
    ]


@pytest.mark.parametrize("array", _directional_linear_spaces())
class TestDirectionalVectorlinearSpace(
    AbstractTestAbstractDirectionalVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass
