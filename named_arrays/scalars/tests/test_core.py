from __future__ import annotations
from typing import Type, Sequence
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from ... import tests

__all__ = [
    'AbstractTestAbstractScalar',
    'AbstractTestAbstractScalarArray',
    'AbstractTestScalarArray',
    'TestNumericScalarArray',
]


class AbstractTestAbstractScalar(
    tests.test_core.AbstractTestAbstractArray,
):
    pass


class AbstractTestAbstractScalarArray(
    AbstractTestAbstractScalar,
):
    pass


class AbstractTestScalarArray(
    AbstractTestAbstractScalarArray,
):

    def _array(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        if unit is not None:
            ndarray = ndarray << unit
        return na.ScalarArray(ndarray=ndarray, axes=axes)

    def test_copy(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_copy(self._array(ndarray, axes, unit))

    def test_copy_shallow(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_copy_shallow(self._array(ndarray, axes, unit))

    def test__neg__(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test__neg__(array)

    def test__pos__(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test__pos__(array)

    def test__abs__(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test__abs__(array)

    def test__invert__(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test__invert__(array)

    def test_ndarray(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_ndarray(array)
        if unit is not None:
            ndarray = ndarray << unit
        assert np.all(array.ndarray == ndarray)

    def test_axes(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_axes(array)
        assert array.axes == axes

    def test_shape(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_shape(array)
        assert array.shape == {axis: shape_axis for axis, shape_axis in zip(axes, np.shape(ndarray))}

    def test_ndim(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_ndim(array)
        assert array.ndim == np.ndim(ndarray)

    def test_dtype(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_dtype(self._array(ndarray, axes, unit))

    def test_unit(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_unit(array)
        if unit is None:
            unit = 1
        assert array.unit == unit

    def test_array(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_array(array)
        assert np.all(array.array == array)

    def test_scalar(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        array = self._array(ndarray, axes, unit)
        super().test_scalar(array)

    def test_components(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_components(self._array(ndarray, axes, unit))

    def test_nominal(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_nominal(self._array(ndarray, axes, unit))

    def test_distribution(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_distribution(self._array(ndarray, axes, unit))

    def test_centers(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_centers(self._array(ndarray, axes, unit))

    @pytest.mark.parametrize('dtype', [int, float])
    def test_astype(
            self,
            ndarray: int | float | complex | np.ndarray,
            axes: list[str],
            unit: None | u.UnitBase,
            dtype: Type,
    ):
        super().test_astype(self._array(ndarray, axes, unit), dtype)

    def test_to(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        unit_new = u.m
        super().test_to(self._array(ndarray, axes, unit), unit_new)

    def test_broadcasted(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_broadcasted(self._array(ndarray, axes, unit))

    def test_length(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_length(self._array(ndarray, axes, unit))

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(x=0),
            dict(x=slice(0,1)),
            dict(x=na.ScalarArray(np.array([0, 1]), axes=['x'])),
        ],
    )
    def test__getitem__(
            self,
            ndarray: int | float | complex | np.ndarray,
            axes: list[str],
            unit: None | u.UnitBase,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        super().test__getitem__(self._array(ndarray, axes, unit), item)

    def test_indices(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_indices(self._array(ndarray, axes, unit))

    def test_ndindex(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        super().test_ndindex(self._array(ndarray, axes, unit))

    @pytest.mark.parametrize('axes_new', ['x0', ('x0', 'y0')])
    def test_add_axes(
            self,
            ndarray: int | float | complex | np.ndarray,
            axes: list[str],
            unit: None | u.UnitBase,
            axes_new: str | Sequence[str],
    ):
        super().test_add_axes(self._array(ndarray, axes, unit), axes_new)

    def test_combine_axes(self, ndarray: int | float | complex | np.ndarray, axes: list[str], unit: None | u.UnitBase):
        axes_old = ['x', 'y']
        super().test_combine_axes(self._array(ndarray, axes, unit), axes=axes_old)


@pytest.mark.parametrize(
    argnames=['ndarray', 'axes'],
    argvalues=[
        (4, []),
        (5., []),
        (np.linspace(0, 1, num=11), ['x']),
        (np.random.random((11, 13)), ['x', 'y']),
    ]
)
@pytest.mark.parametrize(
    argnames='unit',
    argvalues=[
        None,
        u.mm,
    ]
)
class TestNumericScalarArray(
    AbstractTestScalarArray,
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

