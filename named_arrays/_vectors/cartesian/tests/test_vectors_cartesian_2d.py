from typing import Type, Callable, Sequence
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._vectors.tests.test_vectors
from . import test_vectors_cartesian

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
]

_num_x = test_vectors_cartesian._num_x
_num_y = test_vectors_cartesian._num_y
_num_z = test_vectors_cartesian._num_z
_num_distribution = test_vectors_cartesian._num_distribution


def _cartesian2d_arrays():
    arrays_numeric_x = [
        4,
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(y=_num_y)),
        na.Cartesian2dVectorArray(x=2, y=3)
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
        ),
        na.Cartesian2dVectorArray(x=4, y=8)
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
    arrays_vector_x = [a * unit for a in arrays_vector_x for unit in units]
    arrays_vector_y = [
        8,
        na.random.binomial(10, .65, shape_random=dict(x=_num_x, y=_num_y)),
    ]
    arrays_vector_y = [a * unit for a in arrays_vector_y for unit in units]
    arrays_vector = [na.Cartesian2dVectorArray(x=ax, y=ay) for ax in arrays_vector_x for ay in arrays_vector_y]
    arrays = arrays_scalar + arrays_vector
    return arrays


def _cartesian2d_items() -> list[na.AbstractArray | dict[str, int, slice, na.AbstractArray]]:
    return [
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

class AbstractTestAbstractCartesian2dVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames="axis",
        argvalues=[
            None,
            "y",
            ("y", ),
            ("x", "y"),
            ("y", 'z'),
        ]
    )
    def test_volume_cell(
            self,
            array: na.AbstractVectorArray,
            axis: None | str | Sequence[str],
    ):
        axis_ = na.axis_normalized(array, axis)

        if not set(axis_).issubset(array.axes):
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        if len(axis_) != len(array.components):
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        if len(array.components) != len(array.cartesian_nd.entries):
            with pytest.raises(TypeError):
                array.volume_cell(axis)
            return

        result = array.volume_cell(axis)
        shape_result = na.shape(result)

        for ax in array.shape:
            if ax in shape_result:
                if ax in axis_:
                    assert shape_result[ax] == array.shape[ax] - 1
                else:
                    assert shape_result[ax] == array.shape[ax]

        assert isinstance(na.as_named_array(result), na.AbstractScalar)

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_cartesian2d_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractCartesian2dVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _cartesian2d_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _cartesian2d_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _cartesian2d_arrays_2())
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
                (na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                na.Cartesian2dVectorArray(
                    x=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                    y=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) <= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) <= 0),
                )
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
                na.Cartesian2dVectorArray(x=25 * u.percent, y=35 * u.percent),
                na.Cartesian2dVectorArray(
                    x=na.ScalarLinearSpace(.25, .50, axis='q', num=3, endpoint=True),
                    y=na.ScalarLinearSpace(50 * u.percent, 75 * u.percent, axis='q', num=3, endpoint=True)
                )
            ]
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.
            TestPercentileLikeFunctions,
        ):
            pass


@pytest.mark.parametrize('array', _cartesian2d_arrays())
class TestCartesian2dVectorArray(
    AbstractTestAbstractCartesian2dVectorArray,
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
                y=na.Cartesian2dVectorArray(
                    x=na.ScalarArrayRange(0, _num_y, axis='y'),
                    y=na.ScalarArrayRange(0, _num_y, axis='y'),
                )
            ),
            na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
            np.ones_like(na.Cartesian2dVectorArray(), dtype=bool, shape=dict(y=_num_y)),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.ScalarUniformRandomSample(-5, 5, dict(y=_num_y)),
            na.Cartesian2dVectorUniformRandomSample(-5, 5, shape_random=dict(y=_num_y)),
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


@pytest.mark.parametrize("type_array", [na.Cartesian2dVectorArray])
class TestCartesian2dVectorArrayCreation(
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation,
):
    @pytest.mark.parametrize("like", [None] + _cartesian2d_arrays())
    class TestFromScalarArray(
        test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation.TestFromScalarArray,
    ):
        pass


class AbstractTestAbstractImplicitCartesian2dVectorArray(
    AbstractTestAbstractCartesian2dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractCartesian2dVectorRandomSample(
    AbstractTestAbstractImplicitCartesian2dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorRandomSample,
):
    pass


def _cartesian_2d_uniform_random_samples() -> list[na.Cartesian2dVectorUniformRandomSample]:
    starts = [
        0,
        na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(1, 2, axis='x', num=_num_x)
        )
    ]
    stops = [
        10,
        na.UniformUncertainScalarArray(10, width=1, num_distribution=_num_distribution),
        na.Cartesian2dVectorArray(x=10, y=11),
        na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(10, 11, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(11, 12, axis='x', num=_num_x)
        ),
    ]
    units = [None, u.mm]
    shapes_random = [dict(y=_num_y, z=_num_z)]
    return [
        na.Cartesian2dVectorUniformRandomSample(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            shape_random=shape_random,
        ) for start in starts for stop in stops for unit in units for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _cartesian_2d_uniform_random_samples())
class TestCartesian2dVectorUniformRandomSample(
    AbstractTestAbstractCartesian2dVectorRandomSample,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorUniformRandomSample,
):
    pass


def _cartesian_2d_normal_random_samples() -> list[na.Cartesian2dVectorNormalRandomSample]:
    centers = [
        0,
        na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(1, 2, axis='x', num=_num_x)
        )
    ]
    widths = [
        10,
        na.UniformUncertainScalarArray(10, width=1, num_distribution=_num_distribution),
        na.Cartesian2dVectorArray(x=10, y=11),
        na.Cartesian2dVectorArray(
            x=na.ScalarLinearSpace(10, 11, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(11, 12, axis='x', num=_num_x)
        ),
    ]
    units = [None, u.mm]
    shapes_random = [dict(y=_num_y)]
    return [
        na.Cartesian2dVectorNormalRandomSample(
            center=center << unit if unit is not None else center,
            width=width << unit if unit is not None else width,
            shape_random=shape_random,
        ) for center in centers for width in widths for unit in units for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _cartesian_2d_normal_random_samples())
class TestCartesian2dVectorNormalRandomSample(
    AbstractTestAbstractCartesian2dVectorRandomSample,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedCartesian2dVectorArray(
    AbstractTestAbstractImplicitCartesian2dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass



def _cartesian_2d_vector_array_ranges() -> tuple[na.Cartesian2dVectorArrayRange, ...]:
    starts = (
        0,
        na.Cartesian2dVectorArray(0, 1),
    )
    stops = (
        5,
        na.Cartesian2dVectorArray(5, 5)
    )
    axes = (
        na.Cartesian2dVectorArray('x', 'y'),
    )
    _num = na.Cartesian2dVectorArray(_num_x, _num_y)
    return tuple(
        na.Cartesian2dVectorArrayRange(
            start=start,
            stop=stop,
            axis=axis,
            step=(stop - start) / _num,
        )
        for start in starts
        for stop in stops
        for axis in axes
    )


@pytest.mark.parametrize("array", _cartesian_2d_vector_array_ranges())
class TestCartesian2dVectorArrayRange(
    AbstractTestAbstractParameterizedCartesian2dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArrayRange,
):
    pass


class AbstractTestAbstractCartesian2dVectorSpace(
    AbstractTestAbstractParameterizedCartesian2dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorSpace,
):
    pass


def _cartesian_2d_vector_linear_spaces() -> tuple[na.Cartesian2dVectorLinearSpace, ...]:
    starts = (
        0,
        na.Cartesian2dVectorArray(0, 1),
    )
    stops = (
        5,
        na.Cartesian2dVectorArray(5, 5)
    )
    axes = (
        na.Cartesian2dVectorArray('x', 'y'),
    )
    nums = (
        na.Cartesian2dVectorArray(_num_x, _num_y),
    )
    return tuple(
        na.Cartesian2dVectorLinearSpace(
            start=start,
            stop=stop,
            axis=axis,
            num=num,
        )
        for start in starts
        for stop in stops
        for axis in axes
        for num in nums
    )


@pytest.mark.parametrize("array", _cartesian_2d_vector_linear_spaces())
class TestCartesian2dVectorLinearSpace(
    AbstractTestAbstractCartesian2dVectorSpace,
    AbstractTestAbstractCartesian2dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractArray,
):
    @pytest.mark.parametrize(
        argnames="axis",
        argvalues=[
            None,
            "x",
            "y",
            ("x", "y"),
            ("x", "z"),
            ("x", "y", "z"),
        ]
    )
    def test_volume_cell(
            self,
            array: na.AbstractVectorLinearSpace,
            axis: None | str | Sequence[str],
    ):
        super().test_volume_cell(array=array, axis=axis)

        axis_ = na.axis_normalized(array, axis)
        if len(axis_) != len(array.components):
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        if not set(axis_).issubset(array.shape):
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        assert np.allclose(array.volume_cell(axis), array.explicit.volume_cell(axis))
