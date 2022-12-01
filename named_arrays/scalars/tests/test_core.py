from __future__ import annotations
from typing import Type
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from ... import tests

__all__ = [
    'AbstractTestAbstractScalar',
    'AbstractTestAbstractScalarArray',
    'TestScalarArray',
]

_num_x = 11
_num_y = 13


def _scalar_arrays():
    arrays_numeric = [
        na.ScalarArray(4),
        na.ScalarArray(5.),
        na.ScalarArray(10 * (np.random.random((_num_y, )) - 0.5), axes=('y', )),
        na.ScalarArray(10 * (np.random.random((_num_x, _num_y)) - 0.5), axes=('x', 'y')),
    ]
    units = [1, u.mm]
    arrays_numeric = [na.ScalarArray(array.ndarray * unit, array.axes) for array in arrays_numeric for unit in units]
    arrays_bool = [
        na.ScalarArray(np.random.choice([True, False], size=_num_y), axes=('y', )),
        na.ScalarArray(np.random.choice([True, False], size=(_num_x, _num_y)), axes=('x', 'y'))
    ]
    arrays_str = [
        na.ScalarArray('foo'),
        na.ScalarArray(np.random.choice(['foo', 'bar', 'baz'], size=_num_y), axes=('y', )),
    ]
    return arrays_numeric + arrays_bool + arrays_str


def _scalar_arrays_2():
    arrays_numeric = [
        6,
        na.ScalarArray(8),
        na.ScalarArray(10 * (np.random.random((_num_y,)) - 0.5), axes = ('y', )),
        na.ScalarArray(10 * (np.random.random((_num_y, _num_x)) - 0.5), axes=('y', 'x')),
    ]
    units = [1, u.m]
    arrays_numeric = [array * unit for array in arrays_numeric for unit in units]
    arrays_bool = [
        na.ScalarArray(np.random.choice([True, False], size=_num_y), axes=('y', )),
        na.ScalarArray(np.random.choice([True, False], size=(_num_y, _num_x)), axes=('y', 'x'))
    ]
    return arrays_numeric + arrays_bool


class AbstractTestAbstractScalar(
    tests.test_core.AbstractTestAbstractArray,
):
    pass


class AbstractTestAbstractScalarArray(
    AbstractTestAbstractScalar,
):

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=slice(0,1)),
            dict(y=na.ScalarArray(np.array([0, 1]), axes=('y', ))),
            na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
        ]
    )
    def test__getitem__(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array, item)

    @pytest.mark.parametrize('array_2', _scalar_arrays_2())
    class TestUfuncBinary(
        AbstractTestAbstractScalar.TestUfuncBinary,
    ):
        pass

    @pytest.mark.parametrize('axis', [None, 'y', 'x', ('x', 'y')])
    class TestReductionFunctions(
        AbstractTestAbstractScalar.TestReductionFunctions
    ):
        pass


@pytest.mark.parametrize('array', _scalar_arrays())
class TestScalarArray(
    AbstractTestAbstractScalarArray,
    tests.test_core.AbstractTestArrayBase,
):
    @pytest.mark.parametrize('index', [1, ~0])
    def test_change_axis_index(self, array: na.ScalarArray, index: int):
        axis = 'x'
        if axis in array.axes:
            result = array.change_axis_index(axis, index)
            assert result.axes.index(axis) == (index % array.ndim)
        else:
            with pytest.raises(KeyError):
                array.change_axis_index(axis, index)


class TestScalarArrayCreation(
    tests.test_core.AbstractTestArrayBaseCreation
):

    @property
    def type_array(self) -> Type[na.ScalarArray]:
        return na.ScalarArray


class AbstractTestAbstractScalarParameterizedArray(
    AbstractTestAbstractScalarArray,
    tests.test_core.AbstractTestAbstractParameterizedArray,
):
    pass


class AbstractTestAbstractScalarRange(
    AbstractTestAbstractScalarParameterizedArray,
    tests.test_core.AbstractTestAbstractRange,
):
    pass


class AbstractTestAbstractScalarSymmetricRange(
    AbstractTestAbstractScalarParameterizedArray,
    tests.test_core.AbstractTestAbstractSymmetricRange,
):
    pass


def _scalar_uniform_random_samples() -> list[na.ScalarUniformRandomSample]:
    starts = [
        0,
        na.ScalarArray(np.random.random(_num_x), axes=('x', )),
    ]
    stops = [
        10,
        na.ScalarArray(10 * np.random.random(_num_x) + 1, axes=('x', )),
    ]
    units = [None, u.mm]
    return [
        na.ScalarUniformRandomSample(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            axis='y',
            num=_num_y
        ) for start in starts for stop in stops for unit in units
    ]


@pytest.mark.parametrize('array', _scalar_uniform_random_samples())
class TestScalarUniformRandomSample(
    AbstractTestAbstractScalarRange,
    tests.test_core.AbstractTestAbstractUniformRandomSample,
):
    pass


def _scalar_normal_random_samples() -> list[na.ScalarNormalRandomSample]:
    centers = [
        0,
        na.ScalarArray(np.random.random(_num_x), axes=('x', )),
    ]
    widths = [
        10,
        na.ScalarArray(10 * np.random.random(_num_x) + 1, axes=('x', )),
    ]
    units = [None, u.mm]
    return [
        na.ScalarNormalRandomSample(
            center=center << unit if unit is not None else center,
            width=width << unit if unit is not None else width,
            axis='y',
            num=_num_y
        ) for center in centers for width in widths for unit in units
    ]


@pytest.mark.parametrize('array', _scalar_normal_random_samples())
class TestScalarNormalRandomSample(
    AbstractTestAbstractScalarRange,
    tests.test_core.AbstractTestAbstractNormalRandomSample,
):
    pass


def _scalar_array_ranges() -> list[na.ScalarArrayRange]:
    starts = [0, ]
    steps = [1, 2.5, ]
    return [
        na.ScalarArrayRange(
            start=start,
            stop=start + step * _num_y,
            axis='y',
            step=step,
        ) for start in starts for step in steps
    ]


@pytest.mark.parametrize('array', _scalar_array_ranges())
class TestScalarArrayRange(
    AbstractTestAbstractScalarRange,
    tests.test_core.AbstractTestAbstractArrayRange,
):
    pass


class AbstractTestAbstractScalarSpace(
    AbstractTestAbstractScalarRange,
    tests.test_core.AbstractTestAbstractSpace,
):
    pass


def _scalar_linear_spaces():
    starts = [
        0,
        na.ScalarArray(np.random.random(_num_x), axes=('x', )),
    ]
    stops = [
        10,
        na.ScalarArray(10 * np.random.random(_num_x) + 1, axes=('x', )),
    ]
    units = [None, u.mm]
    endpoints = [
        False,
        True,
    ]
    return [
        na.ScalarLinearSpace(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            axis='y',
            num=_num_y,
            endpoint=endpoint
        ) for start in starts for stop in stops for unit in units for endpoint in endpoints
    ]


@pytest.mark.parametrize('array', _scalar_linear_spaces())
class TestScalarLinearSpace(
    AbstractTestAbstractScalarSpace,
    tests.test_core.AbstractTestAbstractLinearSpace,
):
    pass


# class OldTestScalarArray:
#     def test__post_init__(self):
#         with pytest.raises(ValueError):
#             na.ScalarArray(ndarray=np.empty((2, 3)) * u.dimensionless_unscaled, axes=['x'])
#
#     def test_shape(self):
#         shape = dict(x=2, y=3)
#         a = na.ScalarArray(
#             ndarray=np.random.random(tuple(shape.values())) * u.dimensionless_unscaled,
#             axes=['x', 'y'],
#         )
#         assert a.shape == shape
#
#     def test_shape_broadcasted(self):
#         shape = dict(x=5, y=6)
#         d1 = na.ScalarArray.empty(dict(x=shape['x'], y=1))
#         d2 = na.ScalarArray.empty(dict(y=shape['y'], x=1))
#         assert d1.shape_broadcasted(d2) == shape
#
#     def test_ndarray_aligned(self):
#         shape = dict(x=5, y=6, z=7)
#         d = na.ScalarArray.empty(dict(z=shape['z']))
#         assert d.ndarray_aligned(shape).shape == (1, 1, shape['z'])
#
#     def test_combine_axes(self):
#         shape = dict(x=5, y=6, z=7)
#         a = na.ScalarArray.zeros(shape).combine_axes(['x', 'y'])
#         assert a.shape == dict(z=shape['z'], xy=shape['x'] * shape['y'])
#
#     def test__array_ufunc__(self):
#         shape = dict(x=100, y=101)
#         a = na.ScalarArray(
#             ndarray=np.random.random(shape['x']),
#             axes=['x'],
#         )
#         b = na.ScalarArray(
#             ndarray=np.random.random(shape['y']),
#             axes=['y'],
#         )
#         c = a + b
#         assert c.shape == shape
#         assert (c.ndarray == a.ndarray[..., np.newaxis] + b.ndarray).all()
#
#     def test__array_ufunc__incompatible_dims(self):
#         a = na.ScalarArray(
#             ndarray=np.random.random(10),
#             axes=['x'],
#         )
#         b = na.ScalarArray(
#             ndarray=np.random.random(11),
#             axes=['x'],
#         )
#         with pytest.raises(ValueError):
#             a + b
#
#     @pytest.mark.parametrize(
#         argnames='a,b',
#         argvalues=[
#             (na.ScalarArray(5), 6),
#             (na.ScalarArray(5 * u.mm), 6 * u.mm),
#
#             (na.ScalarLinearSpace(0, 1, num=11, axis='x'), 6),
#             (na.ScalarLinearSpace(0, 1, num=11, axis='x') * u.mm, 6 * u.mm),
#             (na.ScalarLinearSpace(0, 1, num=11, axis='x') * u.mm, na.ScalarLinearSpace(0, 1, num=11, axis='x') * u.mm),
#         ],
#     )
#     def test__add__(self, a: na.ScalarLike, b: na.ScalarLike):
#         c = a + b
#         d = b + a
#         b_normalized = b
#         if not isinstance(b, na.AbstractArray):
#             b_normalized = na.ScalarArray(b)
#         assert isinstance(c, na.AbstractArray)
#         assert isinstance(d, na.AbstractArray)
#         assert np.all(c.ndarray == a.ndarray + b_normalized.ndarray)
#         assert np.all(d.ndarray == b_normalized.ndarray + a.ndarray)
#         assert np.all(c == d)
#
#     def test__mul__unit(self):
#         a = na.ScalarUniformRandomSpace(0, 1, axis='x', num=10,) * u.mm
#         assert isinstance(a, na.AbstractArray)
#         assert isinstance(a.ndarray, u.Quantity)
#
#     def test__mul__float(self):
#         a = na.ScalarUniformRandomSpace(0, 1, axis='x', num=10,)
#         b = 2.
#         c = a * b
#         assert isinstance(c, na.Array)
#         assert c.ndarray.mean() > a.ndarray.mean()
#
#     def test__mul__ndarray(self):
#         shape = dict(x=10)
#         a = na.ScalarUniformRandomSpace(0, 1, axis='x', num=shape['x'])
#         b = np.ones(shape['x'])
#         with pytest.raises(ValueError):
#             a * b
#
#     def test__array_function__broadcast_to(self):
#         shape = dict(x=5, y=6)
#         a = na.ScalarLinearSpace(1, 5, axis='y', num=shape['y'])
#         b = np.broadcast_to(a, shape)
#         c = np.broadcast_to(a, shape=shape)
#         d = np.broadcast_to(array=a, shape=shape)
#         assert np.all(b == c)
#         assert np.all(b == d)
#
#     def test__array_function__stack(self):
#         a = na.ScalarLinearSpace(0, 1, num=11, axis='x')
#         b = na.ScalarLinearSpace(2, 3, num=11, axis='x')
#         c = na.ScalarLinearSpace(3, 4, num=11, axis='x')
#         result = np.stack([a, b, c], axis='y')
#         assert np.all(result.ndarray == np.stack([a.ndarray, b.ndarray, c.ndarray]))
#
#     def test__array_function__sum(self):
#         shape = dict(x=4, y=7)
#         a = np.sum(na.ScalarArray.ones(shape))
#         assert a.ndarray == shape['x'] * shape['y']
#         assert a.shape == dict()
#
#     def test__array_function__sum_axis(self):
#         shape = dict(x=4, y=7)
#         a = np.sum(na.ScalarArray.ones(shape), axis='x')
#         assert (a.ndarray == shape['x']).all()
#         assert a.shape == dict(y=shape['y'])
#
#     def test__array_function__sum_keepdims(self):
#         shape = dict(x=4, y=7)
#         a = np.sum(na.ScalarArray.ones(shape), keepdims=True)
#         assert a.ndarray[0, 0] == shape['x'] * shape['y']
#         assert a.shape == dict(x=1, y=1)
#
#     @pytest.mark.parametrize(
#         argnames='a, shift',
#         argvalues=[
#             (na.ScalarLinearSpace(0, 1, num=11, axis='x'), 1),
#             (na.ScalarLinearSpace(0, 1, num=11, axis='x') * na.ScalarLinearSpace(0, 1, num=11, axis='y'), 1),
#         ],
#     )
#     def test__array_function__roll(self, a: na.AbstractScalarArray, shift: int):
#         b = np.roll(a, shift, axis='x')
#         assert np.all(b.ndarray == np.roll(a.ndarray, shift, axis=0))
#
#     def test__getitem__int(self):
#         a = na.ScalarRange(stop=10, axis='x')
#         b = na.ScalarRange(stop=11, axis='y')
#         c = na.ScalarRange(stop=5, axis='z')
#         d = a * b * c
#         index = dict(x=1, y=1)
#         assert (d[index].ndarray == c.ndarray).all()
#         assert d[index].shape == c.shape
#
#     def test__getitem__slice(self):
#         a = na.ScalarRange(stop=10, axis='x')
#         b = na.ScalarRange(stop=11, axis='y')
#         c = na.ScalarRange(stop=5, axis='z')
#         d = a * b * c
#         index = dict(x=slice(1, 2), y=slice(1, 2))
#         assert (d[index].ndarray == c.ndarray).all()
#         assert d[index].shape == dict(x=1, y=1, z=d.shape['z'])
#
#     def test__getitem__advanced_bool(self):
#         a = na.ScalarRange(stop=10, axis='x')
#         b = na.ScalarRange(stop=11, axis='y')
#         c = na.ScalarRange(stop=5, axis='z')
#         d = a * b * c
#         assert d[a > 5].shape == {**b.shape, **c.shape, **d[a > 5].shape}
#
#     def test_ndindex(self):
#         shape = dict(x=2, y=2)
#         result_expected = [{'x': 0, 'y': 0}, {'x': 0, 'y': 1}, {'x': 1, 'y': 0}, {'x': 1, 'y': 1}]
#         a = na.ScalarArray.empty(shape)
#         assert list(a.ndindex()) == result_expected
#
#     # def test_index_nearest_brute(self):
#     #
#     #     x = kgpy.labeled.LinearSpace(0, 1, num=5, axis='x')
#     #     y = kgpy.labeled.LinearSpace(0, 1, num=5, axis='y')
#     #     z = kgpy.labeled.LinearSpace(0, 1, num=5, axis='z')
#     #     a = x + 10 * y + 100 * z
#     #     index_nearest = a.index_nearest_brute(a, axis=('x', 'y'))
#     #     indices = a.indices
#     #
#     #     for ax in indices:
#     #         assert np.all(index_nearest[ax] == indices[ax])
#     #
#     # def test_index_nearest_secant(self):
#     #
#     #     x = na.ScalarLinearSpace(0, 1, num=5, axis='x')
#     #     y = na.ScalarLinearSpace(0, 1, num=5, axis='y')
#     #     # z = kgpy.labeled.LinearSpace(0, 1, num=5, axis='z')
#     #     a = x + 10*y
#     #     index_nearest = a.index_nearest_secant(a, axis='x')
#     #     indices = a.indices
#     #
#     #     for ax in indices:
#     #         assert np.all(index_nearest[ax] == indices[ax])

