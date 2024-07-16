from typing import Type
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._vectors.tests.test_vectors
from . import test_vectors_cartesian
from . import test_vectors_cartesian_2d

__all__ = [
    'AbstractTestAbstractCartesian3dVectorArray',
    'TestCartesian3dVectorArray',
    'TestCartesian3dVectorArrayCreation',
    'AbstractTestAbstractImplicitCartesian3dVectorArray',
    'AbstractTestAbstractCartesian3dVectorRandomSample',
    'TestCartesian3dVectorUniformRandomSample',
    'TestCartesian3dVectorNormalRandomSample',
    'AbstractTestAbstractParameterizedCartesian3dVectorArray',
    'TestCartesian3dVectorArrayRange',
    'AbstractTestAbstractCartesian3dVectorSpace',
    'TestCartesian3dVectorLinearSpace',
]

_num_x = test_vectors_cartesian._num_x
_num_y = test_vectors_cartesian._num_y
_num_z = test_vectors_cartesian._num_z
_num_distribution = test_vectors_cartesian._num_distribution


def _cartesian3d_arrays():
    units = [1, u.mm]
    arrays_numeric_x = [
        4,
    ]

    arrays_numeric_y = [
        5.,
        na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
        na.UniformUncertainScalarArray(
            nominal=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
            width=1,
            num_distribution=_num_distribution
        ),
        na.Cartesian3dVectorArray(x=4, y=8)
    ]

    arrays = [
        na.Cartesian3dVectorArray(x=ax, y=ay) * unit
        for ax in arrays_numeric_x
        for ay in arrays_numeric_y
        for unit in units
    ]
    return arrays


def _cartesian3d_arrays_2():
    units = [1, u.mm]

    arrays_scalar = [
        6,
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
    ]
    arrays_vector_y = [
        8,
        na.ScalarUniformRandomSample(-8, 8, shape_random=dict(y=_num_y, x=_num_x)),
    ]
    arrays_vector = [
        na.Cartesian3dVectorArray(x=ax, y=ay) * unit
        for ax in arrays_vector_x
        for ay in arrays_vector_y
        for unit in units
    ]
    arrays = arrays_scalar + arrays_vector
    return arrays


def _cartesian3d_items() -> list[na.AbstractArray | dict[str, int, slice, na.AbstractArray]]:
    return [
            dict(y=0),
            dict(y=slice(0, 1)),
            dict(y=na.ScalarArrayRange(0, 2, axis='y')),
            dict(
                y=na.Cartesian3dVectorArray(
                    x=na.ScalarArrayRange(0, 2, axis='y'),
                    y=na.ScalarArrayRange(0, 2, axis='y'),
                    z=na.ScalarArrayRange(0, 2, axis='y'),
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
            na.Cartesian3dVectorArray(
                x=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.3,
                y=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
                z=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.6,
            ),
        ]


class AbstractTestAbstractCartesian3dVectorArray(
    test_vectors_cartesian_2d.AbstractTestAbstractCartesian2dVectorArray,
):

    def test_xy(self, array: na.AbstractCartesian3dVectorArray):
        assert isinstance(array.xy, na.Cartesian2dVectorArray)

    def test_solid_area_cell(
        self,
        array: na.AbstractCartesian3dVectorArray,
    ):
        for c in array.components:
            component = na.as_named_array(array.components[c])
            if not isinstance(component, na.AbstractScalar):
                return
        if array.ndim != 2:
            return
        result = array.solid_angle_cell()
        assert isinstance(result, na.AbstractScalar)
        assert np.all(result >= 0)
        assert result.unit.is_equivalent(u.sr)

    @pytest.mark.parametrize("array_2", _cartesian3d_arrays_2())
    def test_cross(
            self,
            array: na.AbstractCartesian3dVectorArray,
            array_2: float | u.Quantity | na.AbstractArray,
    ):
        if not isinstance(array_2, na.AbstractCartesian3dVectorArray):
            with pytest.raises(TypeError):
                array.cross(array_2)
            return

        array_is_not_3d = len(array.cartesian_nd.components) != 3
        array_2_is_not_3d = len(array_2.cartesian_nd.components) != 3

        if array_is_not_3d or array_2_is_not_3d:
            with pytest.raises(ValueError):
                array.cross(array_2)
            return

        assert np.allclose(array.cross(array), 0)

        result = array.cross(array_2)

        assert np.allclose(result @ array, 0)
        assert np.allclose(result@ array_2, 0)

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_cartesian3d_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractCartesian3dVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _cartesian3d_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _cartesian3d_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _cartesian3d_arrays_2())
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
                na.Cartesian3dVectorArray(
                    x=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                    y=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) <= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) <= 0),
                    z=True,
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
                na.Cartesian3dVectorArray(x=25 * u.percent, y=35 * u.percent),
                na.Cartesian3dVectorArray(
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


@pytest.mark.parametrize('array', _cartesian3d_arrays())
class TestCartesian3dVectorArray(
    AbstractTestAbstractCartesian3dVectorArray,
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
                y=na.Cartesian3dVectorArray(
                    x=na.ScalarArrayRange(0, _num_y, axis='y'),
                    y=na.ScalarArrayRange(0, _num_y, axis='y'),
                    z=na.ScalarArrayRange(0, _num_y, axis='y'),
                )
            ),
            na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
            np.ones_like(na.Cartesian3dVectorArray(), dtype=bool, shape=dict(y=_num_y)),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.ScalarUniformRandomSample(-5, 5, dict(y=_num_y)),
            na.Cartesian3dVectorUniformRandomSample(-5, 5, shape_random=dict(y=_num_y)),
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


@pytest.mark.parametrize("type_array", [na.Cartesian3dVectorArray])
class TestCartesian3dVectorArrayCreation(
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation,
):
    @pytest.mark.parametrize("like", [None] + _cartesian3d_arrays())
    class TestFromScalarArray(
        test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArrayCreation.TestFromScalarArray,
    ):
        pass


class AbstractTestAbstractImplicitCartesian3dVectorArray(
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractCartesian3dVectorRandomSample(
    AbstractTestAbstractImplicitCartesian3dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorRandomSample,
):
    pass


def _cartesian_3d_uniform_random_samples() -> list[na.Cartesian3dVectorUniformRandomSample]:
    starts = [
        0,
        na.Cartesian3dVectorArray(
            x=na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(1, 2, axis='x', num=_num_x),
            z=na.ScalarLinearSpace(2, 3, axis='x', num=_num_x)
        )
    ]
    stops = [
        10,
        na.UniformUncertainScalarArray(10, width=1, num_distribution=_num_distribution),
        na.Cartesian3dVectorArray(x=10, y=11, z=12),
        na.Cartesian3dVectorArray(
            x=na.ScalarLinearSpace(10, 11, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(11, 12, axis='x', num=_num_x),
            z=na.ScalarLinearSpace(12, 13, axis='x', num=_num_x),
        ),
    ]
    units = [None, u.mm]
    shapes_random = [dict(y=_num_y)]
    return [
        na.Cartesian3dVectorUniformRandomSample(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            shape_random=shape_random,
        )
        for start in starts
        for stop in stops
        for unit in units
        for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _cartesian_3d_uniform_random_samples())
class TestCartesian3dVectorUniformRandomSample(
    AbstractTestAbstractCartesian3dVectorRandomSample,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorUniformRandomSample,
):
    pass


def _cartesian_3d_normal_random_samples() -> list[na.Cartesian3dVectorNormalRandomSample]:
    centers = [
        0,
        na.Cartesian3dVectorArray(
            x=na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(1, 2, axis='x', num=_num_x),
            z=na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
        )
    ]
    widths = [
        10,
        na.UniformUncertainScalarArray(10, width=1, num_distribution=_num_distribution),
        na.Cartesian3dVectorArray(x=10, y=11, z=12),
        na.Cartesian3dVectorArray(
            x=na.ScalarLinearSpace(10, 11, axis='x', num=_num_x),
            y=na.ScalarLinearSpace(11, 12, axis='x', num=_num_x),
            z=na.ScalarLinearSpace(12, 13, axis='x', num=_num_x),
        ),
    ]
    units = [None, u.mm]
    shapes_random = [dict(y=_num_y)]
    return [
        na.Cartesian3dVectorNormalRandomSample(
            center=center << unit if unit is not None else center,
            width=width << unit if unit is not None else width,
            shape_random=shape_random,
        )
        for center in centers
        for width in widths
        for unit in units
        for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _cartesian_3d_normal_random_samples())
class TestCartesian3dVectorNormalRandomSample(
    AbstractTestAbstractCartesian3dVectorRandomSample,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedCartesian3dVectorArray(
    AbstractTestAbstractImplicitCartesian3dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


def _cartesian_3d_vector_array_ranges() -> tuple[na.Cartesian3dVectorArrayRange, ...]:
    starts = (
        0,
        na.Cartesian3dVectorArray(0, 1, 2),
    )
    stops = (
        5,
        na.Cartesian3dVectorArray(5, 5, 5)
    )
    axes = (
        na.Cartesian3dVectorArray('x', 'y', 'z'),
    )
    _num = na.Cartesian3dVectorArray(_num_x, _num_y, _num_z)
    return tuple(
        na.Cartesian3dVectorArrayRange(
            start=start,
            stop=stop,
            axis=axis,
            step=(stop - start) / _num,
        )
        for start in starts
        for stop in stops
        for axis in axes
    )


@pytest.mark.parametrize("array", _cartesian_3d_vector_array_ranges())
class TestCartesian3dVectorArrayRange(
    AbstractTestAbstractParameterizedCartesian3dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArrayRange,
):
    pass


class AbstractTestAbstractCartesian3dVectorSpace(
    AbstractTestAbstractParameterizedCartesian3dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorSpace,
):
    pass


def _cartesian_3d_vector_linear_spaces() -> tuple[na.Cartesian3dVectorLinearSpace, ...]:
    starts = (
        0,
        na.Cartesian3dVectorArray(0, 1, 2),
    )
    stops = (
        5,
        na.Cartesian3dVectorArray(5, 5, 5)
    )
    axes = (
        na.Cartesian3dVectorArray('x', 'y', 'z'),
    )
    nums = (
        na.Cartesian3dVectorArray(_num_x, _num_y, _num_z),
    )
    return tuple(
        na.Cartesian3dVectorLinearSpace(
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


@pytest.mark.parametrize("array", _cartesian_3d_vector_linear_spaces())
class TestCartesian3dVectorLinearSpace(
    AbstractTestAbstractCartesian3dVectorSpace,
    AbstractTestAbstractCartesian3dVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractArray,
):
    pass
