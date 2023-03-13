from typing import Type, Callable, Sequence

import pytest
import numpy as np
import astropy.units as u

import named_arrays as na
import named_arrays.tests.test_core
import named_arrays.scalars.uncertainties.tests.test_uncertainties
from . import test_cartesian

__all__ = [
    'AbstractTestAbstractCartesian2dVectorArray',
    'TestCartesian2dVectorArray',
    'TestCartesian2dVectorArrayCreation',
    'AbstractTestAbstractImplicitCartesian2dVectorArray',
    'AbstractTestAbstractCartesian2dVectorRandomSample',
    'TestCartesian2dVectorUniformRandomSample',
    'TestCartesian2dVectorNormalRandomSample',
    'AbstractTestAbstractParameterizedCartesian2dVectorArray',
    'TestCartesian2dVectorArrayRange',
    'AbstractTestAbstractCartesian2dVectorSpace',
    'TestCartesian2dVectorLinearSpace',
    'TestCartesian2dVectorStratifiedRandomSpace',
    'TestCartesian2dVectorLogarithmicSpace',
    'TestCartesian2dVectorGeometricSpace',
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_distribution = named_arrays.scalars.uncertainties.tests.test_uncertainties._num_distribution


def _cartesian2d_arrays():
    arrays_numeric_x = [
        4,
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(y=_num_y)),
    ]
    units_x = [1, u.mm]
    arrays_numeric_x = [a * unit for a in arrays_numeric_x for unit in units_x]

    arrays_numeric_y = [
        5.,
        na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
        na.UniformUncertainScalarArray(
            nominal=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
            width=1,
            num_distribution=_num_distribution
        )
    ]
    units_y = [1, u.mm]
    arrays_numeric_y = [a * unit for a in arrays_numeric_y for unit in units_y]

    arrays = [na.Cartesian2dVectorArray(x=ax, y=ay) for ax in arrays_numeric_x for ay in arrays_numeric_y]
    return arrays


def _cartesian2d_arrays_2():
    units = [1, u.mm]
    arrays_scalar = [
        6,
        na.ScalarArray(6),
        na.ScalarUniformRandomSample(-6, 6, shape_random=dict(y=_num_y, x=_num_x)),
    ]
    arrays_scalar = [a * unit for a in arrays_scalar for unit in units]
    arrays_vector_x = [
        7,
        na.ScalarUniformRandomSample(-7, 7, shape_random=dict(y=_num_y)),
    ]
    arrays_vector_x = [a * unit for a in arrays_vector_x for unit in units]
    arrays_vector_y = [
        8,
        na.ScalarUniformRandomSample(-8, 8, shape_random=dict(y=_num_y, x=_num_x)),
    ]
    arrays_vector_y = [a * unit for a in arrays_vector_y for unit in units]
    arrays_vector = [na.Cartesian2dVectorArray(x=ax, y=ay) for ax in arrays_vector_x for ay in arrays_vector_y]
    arrays = arrays_scalar + arrays_vector
    return arrays


class AbstractTestAbstractCartesian2dVectorArray(
    test_cartesian.AbstractTestAbstractCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=slice(0, 1)),
            dict(y=na.ScalarArrayRange(0, 2, axis='y')),
            dict(
                y=na.Cartesian2dVectorArray(
                    x=na.ScalarArrayRange(0, 2, axis='y'),
                    y=na.ScalarArrayRange(0, 2, axis='y'),
                )
            ),
            dict(
                y=na.UncertainScalarArray(
                    nominal=na.ScalarArray(np.array([0, 1]), axes=('y',)),
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0, ], [1, ]]),
                        axes=('y', na.UncertainScalarArray.axis_distribution),
                    )
                ),
                _distribution=na.UncertainScalarArray(
                    nominal=None,
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0], [0]]),
                        axes=('y', na.UncertainScalarArray.axis_distribution),
                    )
                )
            ),
            na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
            na.UniformUncertainScalarArray(
                nominal=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                width=0.1,
                num_distribution=_num_distribution,
            ) > 0.5,
            na.Cartesian2dVectorArray(
                x=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.3,
                y=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
            ),
        ]
    )
    def test__getitem__(
            self,
            array: na.AbstractCartesian2dVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _cartesian2d_arrays_2())
    class TestUfuncBinary(
        test_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _cartesian2d_arrays_2())
    class TestMatmul(
        test_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
                (na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                na.Cartesian2dVectorArray(
                    x=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                    y=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) <= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) <= 0),
                )
            ]
        )
        class TestReductionFunctions(
            test_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                .25,
                25 * u.percent,
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
                na.Cartesian2dVectorArray(x=25 * u.percent, y=35 * u.percent),
                na.Cartesian2dVectorArray(
                    x=na.ScalarLinearSpace(.25, .50, axis='q', num=3, endpoint=True),
                    y=na.ScalarLinearSpace(50 * u.percent, 75 * u.percent, axis='q', num=3, endpoint=True)
                )
            ]
        )
        class TestPercentileLikeFunctions(
            test_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestPercentileLikeFunctions,
        ):
            pass


@pytest.mark.parametrize('array', _cartesian2d_arrays())
class TestCartesian2dVectorArray(
    AbstractTestAbstractCartesian2dVectorArray,
    test_cartesian.AbstractTestAbstractExplicitCartesianVectorArray,
):
    pass


class TestCartesian2dVectorArrayCreation(
    test_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation,
):
    @property
    def type_array(self) -> Type[na.Cartesian2dVectorArray]:
        return na.Cartesian2dVectorArray


class AbstractTestAbstractImplicitCartesian2dVectorArray(
    AbstractTestAbstractCartesian2dVectorArray,
    test_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractCartesian2dVectorRandomSample(
    AbstractTestAbstractImplicitCartesian2dVectorArray,
    test_cartesian.AbstractTestAbstractCartesianVectorRandomSample,
):
    pass


def _cartesian_2d_uniform_random_samples() -> list[na.Cartesian2dVectorUniformRandomSample]:
    starts = [
        0,
        na.Cartesian2dVectorArray(x=0, y=1),
        na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(1, 2, axis='x', num=_num_x)
        )
    ]
    stops = [
        10,
        na.Cartesian2dVectorArray(x=10, y=11),
        na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(10, 11, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(11, 12, axis='x', num=_num_x)
        ),
    ]
    units = [None, u.mm]
    shapes_random = [dict(y=_num_y)]
    return [
        na.Cartesian2dVectorUniformRandomSample(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            shape_random=shape_random,
        ) for start in starts for stop in stops for unit in units for shape_random in shapes_random
    ]


@pytest.mark.skip
@pytest.mark.parametrize('array', _cartesian_2d_uniform_random_samples())
class TestCartesian2dVectorUniformRandomSample(
    AbstractTestAbstractCartesian2dVectorRandomSample,
    test_cartesian.AbstractTestAbstractCartesianVectorUniformRandomSample,
):
    pass


@pytest.mark.skip
class TestCartesian2dVectorNormalRandomSample(
    AbstractTestAbstractCartesian2dVectorRandomSample,
    test_cartesian.AbstractTestAbstractCartesianVectorNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedCartesian2dVectorArray(
    AbstractTestAbstractImplicitCartesian2dVectorArray,
    test_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


@pytest.mark.skip
class TestCartesian2dVectorArrayRange(
    AbstractTestAbstractParameterizedCartesian2dVectorArray,
    test_cartesian.AbstractTestAbstractCartesianVectorArrayRange,
):
    pass


class AbstractTestAbstractCartesian2dVectorSpace(
    AbstractTestAbstractParameterizedCartesian2dVectorArray,
    test_cartesian.AbstractTestAbstractCartesianVectorSpace,
):
    pass

@pytest.mark.skip
class TestCartesian2dVectorLinearSpace(
    AbstractTestAbstractCartesian2dVectorSpace,
    test_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass

@pytest.mark.skip
class TestCartesian2dVectorStratifiedRandomSpace(
    TestCartesian2dVectorLinearSpace,
    test_cartesian.AbstractTestAbstractCartesianVectorStratifiedRandomSpace,
):
    pass

@pytest.mark.skip
class TestCartesian2dVectorLogarithmicSpace(
    AbstractTestAbstractCartesian2dVectorSpace,
    test_cartesian.AbstractTestAbstractCartesianVectorLogarithmicSpace,
):
    pass

@pytest.mark.skip
class TestCartesian2dVectorGeometricSpace(
    AbstractTestAbstractCartesian2dVectorSpace,
    test_cartesian.AbstractTestAbstractCartesianVectorGeometricSpace,
):
    pass
