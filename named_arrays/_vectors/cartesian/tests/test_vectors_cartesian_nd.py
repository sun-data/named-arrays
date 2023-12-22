from typing import Type, Callable, Sequence
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
from . import test_vectors_cartesian

__all__ = [
    'AbstractTestAbstractCartesianNdVectorArray',
    'TestCartesianNdVectorArray',
]

_num_x = test_vectors_cartesian._num_x
_num_y = test_vectors_cartesian._num_y
_num_z = test_vectors_cartesian._num_z
_num_distribution = test_vectors_cartesian._num_distribution


def _cartesian_nd_arrays():
    units = [1, u.mm]
    arrays_numeric_x = [
        4,
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(y=_num_y)),
    ]

    arrays_numeric_y = [
        5.,
        na.UniformUncertainScalarArray(
            nominal=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
            width=1,
            num_distribution=_num_distribution
        ),
        na.Cartesian2dVectorArray(x=7.3, y=5)
    ]

    arrays = [
        na.CartesianNdVectorArray(components=dict(x=ax, y=ay)) * unit
        for ax in arrays_numeric_x
        for ay in arrays_numeric_y
        for unit in units
    ]

    return arrays


def _cartesian_nd_arrays_2():
    units = [1, u.mm]
    arrays_scalar = [
        6,
        na.ScalarArray(6),
        na.ScalarUniformRandomSample(-6, 6, shape_random=dict(y=_num_y, x=_num_x)),
        na.UniformUncertainScalarArray(
            nominal=na.ScalarUniformRandomSample(-6, 6, shape_random=dict(y=_num_y, x=_num_x)),
            width=1,
            num_distribution=_num_distribution
        )
    ]
    arrays_scalar = [a * unit for a in arrays_scalar for unit in units]

    arrays_vector_x = [
        7,
        na.ScalarUniformRandomSample(-7, 7, shape_random=dict(y=_num_y)),
    ]
    arrays_vector_y = [
        8,
        na.ScalarUniformRandomSample(-8, 8, shape_random=dict(y=_num_y, x=_num_x)),
    ]
    arrays_vector = [
        na.CartesianNdVectorArray(dict(x=ax, y=ay))  * unit
        for ax in arrays_vector_x
        for ay in arrays_vector_y
        for unit in units
    ]

    arrays = arrays_scalar + arrays_vector
    return arrays


def _cartesian_nd_items() -> list[na.AbstractArray | dict[str, int, slice, na.AbstractArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
        dict(
            y=na.CartesianNdVectorArray(dict(
                x=na.ScalarArrayRange(0, 2, axis='y'),
                y=na.ScalarArrayRange(0, 2, axis='y'),
            ))
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
        na.CartesianNdVectorArray(dict(
            x=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.3,
            y=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
        )),
    ]


class AbstractTestAbstractCartesianNdVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_cartesian_nd_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractCartesianNdVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _cartesian_nd_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _cartesian_nd_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):
        @pytest.mark.parametrize("array_2", _cartesian_nd_arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                (na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                na.CartesianNdVectorArray(dict(
                    x=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                    y=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) <= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) <= 0),
                ))
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
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
                na.CartesianNdVectorArray(dict(x=25 * u.percent, y=35 * u.percent)),
                na.CartesianNdVectorArray(dict(
                    x=na.ScalarLinearSpace(.25, .50, axis='q', num=3, endpoint=True),
                    y=na.ScalarLinearSpace(50 * u.percent, 75 * u.percent, axis='q', num=3, endpoint=True)
                ))
            ]
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.
            TestPercentileLikeFunctions,
        ):
            pass

    class TestNamedArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions,
    ):
        @pytest.mark.xfail(raises=TypeError)
        class TestPltPlotLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass


@pytest.mark.parametrize('array', _cartesian_nd_arrays())
class TestCartesianNdVectorArray(
    AbstractTestAbstractCartesianNdVectorArray,
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(x=0, y=0),
            dict(y=slice(None)),
            dict(y=na.ScalarArrayRange(0, _num_y, axis='y')),
            dict(x=na.ScalarArrayRange(0, _num_x, axis='x'), y=na.ScalarArrayRange(0, _num_y, axis='y')),
            dict(
                y=na.CartesianNdVectorArray(dict(
                    x=na.ScalarArrayRange(0, _num_y, axis='y'),
                    y=na.ScalarArrayRange(0, _num_y, axis='y'),
                ))
            ),
            na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
            np.ones_like(na.CartesianNdVectorArray(dict(x=0, y=0)), dtype=bool, shape=dict(y=_num_y)),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.ScalarUniformRandomSample(-5, 5, dict(y=_num_y)),
            na.CartesianNdVectorArray(
                dict(
                    x=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(y=_num_y)),
                    y=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(y=_num_y))
                )
            ),
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


@pytest.mark.parametrize("type_array", [na.CartesianNdVectorArray])
class TestCartesianNdVectorArrayCreation(
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation,
):
    @pytest.mark.parametrize("like", [None] + _cartesian_nd_arrays())
    class TestFromScalarArray(
        test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation.TestFromScalarArray,
    ):
        pass
