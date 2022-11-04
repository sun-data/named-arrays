from __future__ import annotations
from typing import Sequence, Type
import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na
from . import test_mixins

num_x = 1
num_y = 2
num_z = 3


def _normalize_shape(shape: dict[str, None | int]) -> dict[str, int]:
    return {axis: shape[axis] for axis in shape if shape[axis] is not None}


@pytest.mark.parametrize(argnames='shape_1_x', argvalues=[None, num_x], )
@pytest.mark.parametrize(argnames='shape_1_y', argvalues=[None, num_y], )
@pytest.mark.parametrize(argnames='shape_2_x', argvalues=[None, num_x], )
@pytest.mark.parametrize(argnames='shape_2_y', argvalues=[None, num_y], )
class TestBroadcastingFunctions:

    def _shapes(self, shape_1_x: int, shape_1_y: int, shape_2_x: int, shape_2_y: int) -> tuple[dict[str, int], dict[str, int]]:
        shape_1 = _normalize_shape(dict(x=shape_1_x, y=shape_1_y))
        shape_2 = _normalize_shape(dict(x=shape_2_x, y=shape_2_y))
        return shape_1, shape_2

    def _shape_expected(self, shape_1_x: int, shape_1_y: int, shape_2_x: int, shape_2_y: int):

        shape_expected = dict(x=num_x, y=num_y)
        if shape_1_x is None and shape_2_x is None:
            del shape_expected['x']
        if shape_1_y is None and shape_2_y is None:
            del shape_expected['y']

        return shape_expected

    def test_broadcast_shapes(self, shape_1_x: int, shape_1_y: int, shape_2_x: int, shape_2_y: int):

        shape_1, shape_2 = self._shapes(shape_1_x, shape_1_y, shape_2_x, shape_2_y)

        shape_broadcasted = na.broadcast_shapes(shape_1, shape_2)

        assert shape_broadcasted == self._shape_expected(shape_1_x, shape_1_y, shape_2_x, shape_2_y)

    def test_shape_broadcasted(self, shape_1_x: int, shape_1_y: int, shape_2_x: int, shape_2_y: int):

        shape_1, shape_2 = self._shapes(shape_1_x, shape_1_y, shape_2_x, shape_2_y)

        array_1 = na.ScalarArray.empty(shape_1)
        array_2 = na.ScalarArray.empty(shape_2)

        shape_broadcasted = na.shape_broadcasted(array_1, array_2)

        assert shape_broadcasted == self._shape_expected(shape_1_x, shape_1_y, shape_2_x, shape_2_y)


def are_units_equivalent(
        array_1: bool | int | float | complex | str | np.ndarray | na.AbstractArray,
        array_2: bool | int | float | complex | str | np.ndarray | na.AbstractArray,
):
    unit_1 = na.get_unit(array_1)
    unit_2 = na.get_unit(array_2)

    if unit_1 is None:
        if unit_2 is None:
            return True
        else:
            return False
    else:
        if unit_2 is None:
            return False
        else:
            return unit_1.is_equivalent(unit_2)


@pytest.mark.parametrize('shape_x', [None, num_x])
@pytest.mark.parametrize('shape_y', [None, num_y])
@pytest.mark.parametrize('shape_z', [None, num_z])
class TestIndexingFunctions:

    def _shape(self, shape_x: None | int, shape_y: None | int, shape_z: None | int) -> dict[str, int]:
        return _normalize_shape(dict(x=shape_x, y=shape_y, z=shape_z))

    @pytest.mark.parametrize('ignore_x', [False, True])
    @pytest.mark.parametrize('ignore_y', [False, True])
    @pytest.mark.parametrize('ignore_z', [False, True])
    def test_ndindex(
            self,
            shape_x: None | int,
            shape_y: None | int,
            shape_z: None | int,
            ignore_x: bool,
            ignore_y: bool,
            ignore_z: bool,
    ):
        shape = self._shape(shape_x, shape_y, shape_z)

        axis_ignored_normalized = []
        if ignore_x:
            axis_ignored_normalized.append('x')
        if ignore_y:
            axis_ignored_normalized.append('y')
        if ignore_z:
            axis_ignored_normalized.append('z')

        if not axis_ignored_normalized:
            axis_ignored = None
        elif len(axis_ignored_normalized) == 1:
            axis_ignored = axis_ignored_normalized[0]
        else:
            axis_ignored = axis_ignored_normalized

        ndindex = list(na.ndindex(shape, axis_ignored=axis_ignored))

        shape_not_ignored = shape.copy()
        for axis in axis_ignored_normalized:
            if axis in shape:
                shape_not_ignored.pop(axis)

        if shape_not_ignored:
            assert len(ndindex) == np.array(list(shape_not_ignored.values())).prod()
        else:
            assert len(ndindex) == 1

        assert {axis: 0 for axis in shape_not_ignored} == ndindex[0]
        assert {axis: shape_not_ignored[axis] - 1 for axis in shape_not_ignored} == ndindex[~0]

    def test_indices(self, shape_x: None | int, shape_y: None | int, shape_z: None | int):

        shape = self._shape(shape_x, shape_y, shape_z)

        indices = na.indices(shape)

        assert len(indices) == len(shape)
        for axis in shape:
            assert indices[axis].shape[axis] == shape[axis]
            assert indices[axis][{axis: 0}] == 0
            assert indices[axis][{axis: ~0}] == shape[axis] - 1


class AbstractTestAbstractArray(
    test_mixins.AbstractTestCopyable,
):
    def test__neg__(self, array: na.AbstractArray):
        if np.issubdtype(array.dtype, np.number):
            assert np.all((-array).ndarray == -(array.ndarray))
        else:
            with pytest.raises(TypeError):
                -array

    def test__pos__(self, array: na.AbstractArray):
        if np.issubdtype(array.dtype, np.number):
            assert np.all((+array).ndarray == +(array.ndarray))
        else:
            with pytest.raises(TypeError):
                +array

    def test__abs__(self, array: na.AbstractArray):
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, bool):
            assert np.all(abs(array).ndarray == abs(array.ndarray))
        else:
            with pytest.raises(TypeError):
                abs(array)

    def test__invert__(self, array: na.AbstractArray):
        if np.issubdtype(array.dtype, bool) or np.issubdtype(array.dtype, np.int):
            assert np.all((~array).ndarray == ~(array.ndarray))
        else:
            with pytest.raises(TypeError):
                ~array

    def test_ndarray(self, array: na.AbstractArray):
        assert isinstance(array.ndarray, (int, float, complex, str, np.ndarray))

    def test_axes(self, array: na.AbstractArray):
        axes = array.axes
        assert isinstance(axes, list)
        assert len(axes) == np.ndim(array.ndarray)
        for axis in axes:
            assert isinstance(axis, str)

    def test_shape(self, array: na.AbstractArray):
        shape = array.shape
        assert isinstance(shape, dict)
        for axis in shape:
            assert isinstance(axis, str)
            assert isinstance(shape[axis], int)

    def test_ndim(self, array: na.AbstractArray):
        assert isinstance(array.ndim, int)

    def test_dtype(self, array: na.AbstractArray):
        assert array.dtype is not None

    def test_unit(self, array: na.AbstractArray):
        unit = array.unit
        if unit is not None:
            assert isinstance(array.unit, u.UnitBase)

    def test_array(self, array: na.AbstractArray):
        assert isinstance(array.array, na.ArrayBase)

    def test_type_array(self, array: na.AbstractArray):
        assert issubclass(array.type_array, na.ArrayBase)

    def test_scalar(self, array: na.AbstractArray):
        assert isinstance(array.scalar, na.AbstractScalar)

    def test_components(self, array: na.AbstractArray):
        components = array.components
        assert isinstance(components, dict)
        for component in components:
            assert isinstance(component, str)
            assert isinstance(components[component], (int, float, complex, np.ndarray, na.AbstractArray))

    def test_nominal(self, array: na.AbstractArray):
        assert isinstance(array.nominal, na.AbstractArray)

    def test_distribution(self, array: na.AbstractArray):
        assert isinstance(array.distribution, na.AbstractArray) or array.distribution is None

    def test_centers(self, array: na.AbstractArray):
        assert isinstance(array.centers, na.AbstractArray)

    @pytest.mark.parametrize('dtype', [int, float])
    def test_astype(self, array: na.AbstractArray, dtype: type):
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, bool):
            array_new = array.astype(dtype)
            assert array_new.dtype == dtype
        else:
            with pytest.raises(ValueError):
                array.astype(dtype)

    @pytest.mark.parametrize('unit', [u.m, u.s])
    def test_to(self, array: na.AbstractArray, unit: None | u.UnitBase):
        if isinstance(array.unit, u.UnitBase) and array.unit.is_equivalent(unit):
            array_new = array.to(unit)
            assert array_new.unit == unit
        elif np.issubdtype(array.dtype, str):
            with pytest.raises(TypeError):
                array.to(unit)
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test_broadcasted(self, array: na.AbstractArray):
        array_broadcasted = array.broadcasted
        shape = array.shape
        components = array_broadcasted.components
        for component in components:
            assert components[component].shape == shape

    def test_length(self, array: na.AbstractArray):
        if np.issubdtype(array.dtype, np.number):
            assert isinstance(array.length, na.AbstractScalar)
            assert np.all(array.length >= 0)
        else:
            with pytest.raises(ValueError):
                array.length

    def test__getitem__(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        if array.shape:
            array_item = array[item]
            assert isinstance(array_item, na.AbstractArray)
            if np.issubdtype(array.dtype, np.number):
                assert np.all(array_item > 0)
        else:
            with pytest.raises(ValueError):
                array[item]

    def test_indices(self, array: na.AbstractArray):

        indices = array.indices
        indices_expected = na.indices(array.shape)

        for axis in indices_expected:
            assert np.all(indices[axis] == indices_expected[axis])

    def test_ndindex(self, array: na.AbstractArray):
        assert list(array.ndindex()) == list(na.ndindex(array.shape))

    @pytest.mark.parametrize('axes', ['x0', ('x0', 'y0')])
    def test_add_axes(self, array: na.AbstractArray, axes: str | Sequence[str]):
        array_new = array.add_axes(axes)

        if isinstance(axes, str):
            axes = [axes]

        for axis in axes:
            assert axis in array_new.axes
            assert array_new.shape[axis] == 1

    @pytest.mark.parametrize('axes', [('x', 'y'), ('x', 'y', 'z')])
    def test_combine_axes(self, array: na.AbstractArray, axes: Sequence[str]):
        axis_new = 'new_test_axis'
        if set(axes).issubset(array.axes):
            array_new = array.combine_axes(axes=axes, axis_new=axis_new)
            assert axis_new in array_new.axes
            assert array_new.shape[axis_new] == np.array([array.shape[axis] for axis in axes]).prod()
            for axis in axes:
                assert axis not in array_new.axes
        else:
            with pytest.raises(ValueError):
                array.combine_axes(axes=axes, axis_new=axis_new)

    class TestBinaryOperators(
        abc.ABC,
    ):

        @classmethod
        def _to_ndarray(
                cls,
                array,
                shape,
        ):
            if isinstance(array, na.AbstractArray):
                result = array.broadcast_to(shape).ndarray
            else:
                result = array
            return result

        def test__lt__(
                self,
                array: bool | int | float | complex | u.Quantity | na.AbstractArray,
                array_2: bool | int | float | complex | u.Quantity | na.AbstractArray,
        ):
            array_dtype = na.get_dtype(array)
            array_2_dtype = na.get_dtype(array_2)

            is_array_numeric_or_bool = np.issubdtype(array_dtype, np.number) or np.issubdtype(array_dtype, bool)
            is_array_2_numeric_or_bool = np.issubdtype(array_2_dtype, np.number) or np.issubdtype(array_2_dtype, bool)
            units_are_equivalent = are_units_equivalent(array, array_2)

            if is_array_numeric_or_bool and is_array_2_numeric_or_bool and units_are_equivalent:
                shape = na.shape_broadcasted(array, array_2)

                ndarray_1 = self._to_ndarray(array, shape)
                ndarray_2 = self._to_ndarray(array_2, shape)

                result = array < array_2
                ndarray_expected = ndarray_1 < ndarray_2

                assert np.all(result.ndarray == ndarray_expected)
                assert result.shape == shape

            else:
                with pytest.raises((TypeError, u.UnitConversionError)):
                    array < array_2

        def test__lt__reversed(
                self,
                array: bool | int | float | complex | u.Quantity | na.AbstractArray,
                array_2: bool | int | float | complex | u.Quantity | na.AbstractArray,
        ):
            self.test__lt__(array_2, array)


class AbstractTestArrayBase(
    AbstractTestAbstractArray,
):
    pass


@pytest.mark.parametrize('shape', [dict(x=3), dict(x=4, y=5)])
@pytest.mark.parametrize('dtype', [int, float, complex])
class AbstractTestArrayBaseCreation(abc.ABC):

    @property
    @abc.abstractmethod
    def type_array(self) -> Type[na.ArrayBase]:
        pass

    def test_empty(self, shape: dict[str, int], dtype: Type):
        result = self.type_array.empty(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == dtype

    def test_zeros(self, shape: dict[str, int], dtype: Type):
        result = self.type_array.zeros(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == dtype
        assert np.all(result == 0)

    def test_ones(self, shape: dict[str, int], dtype: Type):
        result = self.type_array.ones(shape, dtype=dtype)
        assert result.shape == shape
        assert result.dtype == dtype
        assert np.all(result == 1)


class AbstractTestAbstractParameterizedArray(
    AbstractTestAbstractArray,
):

    def test_axis(self, array: na.AbstractParameterizedArray):
        assert isinstance(array.axis, (str, na.AbstractArray))

    def test_num(self, array: na.AbstractParameterizedArray):
        assert isinstance(array.num, (int, na.AbstractArray))


class AbstractTestAbstractRandomMixin(
    abc.ABC,
):

    def test_seed(self, array: na.AbstractRandomMixin):
        assert isinstance(array.seed, int)


class AbstractTestRandomMixin(
    AbstractTestAbstractRandomMixin,
):
    pass


class AbstractTestAbstractRange(
    AbstractTestAbstractParameterizedArray,
):

    def test_start(self, array: na.AbstractRange):
        assert isinstance(array.start, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_stop(self, array: na.AbstractRange):
        assert isinstance(array.stop, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_range(self, array: na.AbstractRange):
        assert np.abs(array.range) > 0


class AbstractTestAbstractSymmetricRange(
    AbstractTestAbstractRange,
):
    def test_center(self, array: na.AbstractSymmetricRange):
        assert isinstance(array.center, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_width(self, array: na.AbstractSymmetricRange):
        assert isinstance(array.width, (int, float, complex, u.Quantity, na.AbstractArray))
        assert np.all(array.width > 0)


class AbstractTestAbstractLinearParametrizedArrayMixin:

    def test_step(self, array: na.AbstractLinearParameterizedArrayMixin):
        assert isinstance(array.step, (int, float, complex, u.Quantity, na.AbstractArray))
        assert np.all(np.abs(array.step) > 0)


class AbstractTestAbstractArrayRange(
    AbstractTestAbstractLinearParametrizedArrayMixin,
    AbstractTestAbstractRange,
):
    pass


class AbstractTestAbstractSpace(
    AbstractTestAbstractRange,
):

    def test_endpoint(self, array: na.AbstractSpace):
        assert isinstance(array.endpoint, bool)


class AbstractTestAbstractLinearSpace(
    AbstractTestAbstractLinearParametrizedArrayMixin,
    AbstractTestAbstractSpace,
):
    pass


class AbstractTestAbstractStratifiedRandomSpace(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractLinearSpace,
):
    pass


class AbstractTestAbstractLogarithmicSpace(
    AbstractTestAbstractSpace,
):

    def test_start_exponent(self, array: na.AbstractLogarithmicSpace):
        assert isinstance(array.start_exponent, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_stop_exponent(self, array: na.AbstractLogarithmicSpace):
        assert isinstance(array.stop_exponent, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_base(self, array: na.AbstractLogarithmicSpace):
        assert isinstance(array.base, (int, float, complex, u.Quantity, na.AbstractArray))


class AbstractTestAbstractGeometricSpace(
    AbstractTestAbstractSpace,
):
    pass


class AbstractTestAbstractUniformRandomSample(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractRange,
):
    pass


class AbstractTestAbstractNormalRandomSample(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractSymmetricRange,
):
    pass