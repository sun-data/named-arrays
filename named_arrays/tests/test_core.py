from __future__ import annotations
from typing import Sequence, Type, Callable
import pytest
import abc
import numpy as np
import astropy.units as u
import astropy.units.quantity_helper.helpers as quantity_helpers
import named_arrays as na
from . import test_mixins

num_x = 1
num_y = 2
num_z = 3


def _normalize_shape(shape: dict[str, None | int]) -> dict[str, int]:
    return {axis: shape[axis] for axis in shape if shape[axis] is not None}


@pytest.mark.parametrize(argnames='shape_1_x', argvalues=[num_x], )
@pytest.mark.parametrize(argnames='shape_1_y', argvalues=[num_y], )
@pytest.mark.parametrize(argnames='shape_2_x', argvalues=[None, 1, num_x], )
@pytest.mark.parametrize(argnames='shape_2_y', argvalues=[None, 1, num_y], )
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

        assert na.broadcast_shapes(shape_1, shape_2) == self._shape_expected(shape_1_x, shape_1_y, shape_2_x, shape_2_y)
        assert na.broadcast_shapes(shape_2, shape_1) == self._shape_expected(shape_1_x, shape_1_y, shape_2_x, shape_2_y)

    def test_broadcast_shapes_invalid(self, shape_1_x: int, shape_1_y: int, shape_2_x: int, shape_2_y: int):
        shape_1, shape_2 = self._shapes(shape_1_x, shape_1_y, shape_2_x, shape_2_y)
        shape_3 = dict(x=num_x + 1, y=num_y + 1)
        with pytest.raises(ValueError, match="shapes .* are not compatible"):
            na.broadcast_shapes(shape_1, shape_2, shape_3)

    def test_shape_broadcasted(self, shape_1_x: int, shape_1_y: int, shape_2_x: int, shape_2_y: int):

        shape_1, shape_2 = self._shapes(shape_1_x, shape_1_y, shape_2_x, shape_2_y)

        array_1 = na.ScalarArray.empty(shape_1)
        array_2 = na.ScalarArray.empty(shape_2)

        shape_broadcasted = na.shape_broadcasted(array_1, array_2)

        assert shape_broadcasted == self._shape_expected(shape_1_x, shape_1_y, shape_2_x, shape_2_y)


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

    def test_ndarray(self, array: na.AbstractArray):
        assert isinstance(array.ndarray, (int, float, complex, str, np.ndarray))

    def test_axes(self, array: na.AbstractArray):
        axes = array.axes
        assert isinstance(axes, tuple)
        assert len(axes) == np.ndim(array.ndarray)
        for axis in axes:
            assert isinstance(axis, str)

    def test_axes_flattened(self, array: na.AbstractArray):
        axes = array.axes_flattened
        assert isinstance(axes, str)
        for ax in array.axes:
            assert ax in axes

    def test_shape(self, array: na.AbstractArray):
        shape = array.shape
        assert isinstance(shape, dict)
        for axis in shape:
            assert isinstance(axis, str)
            assert isinstance(shape[axis], int)

    def test_ndim(self, array: na.AbstractArray):
        assert isinstance(array.ndim, int)

    def test_size(self, array: na.AbstractArray):
        size = array.size
        assert isinstance(size, int)
        assert size == np.size(array.ndarray)

    def test_dtype(self, array: na.AbstractArray):
        assert array.dtype is not None

    def test_unit(self, array: na.AbstractArray):
        unit = array.unit
        if unit is not None:
            assert isinstance(array.unit, u.UnitBase)

    def test_unit_normalized(self, array: na.AbstractArray):
        if array.unit is None:
            assert array.unit_normalized == u.dimensionless_unscaled
        else:
            assert array.unit_normalized == array.unit

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
            result = array[item]
            assert isinstance(result, na.AbstractArray)

            if isinstance(item, dict):
                item_expected = [Ellipsis]
                for axis in item:
                    if isinstance(item[axis], na.AbstractArray):
                        item_expected.append(item[axis].ndarray)
                    else:
                        item_expected.append(item[axis])
                item_expected = tuple(item_expected)
            else:
                item_expected = (Ellipsis, item.ndarray)

            result_expected = array.ndarray[item_expected]
            if 'y' in result.axes:
                result = result.change_axis_index('y', ~0)
            assert np.all(result.ndarray == result_expected)

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

    @pytest.mark.parametrize(
        argnames='ufunc',
        argvalues=[
            np.negative,
            np.positive,
            np.absolute,
            np.fabs,
            np.rint,
            np.sign,
            np.conj,
            np.conjugate,
            np.exp,
            np.exp2,
            np.log,
            np.log2,
            np.log10,
            np.expm1,
            np.log1p,
            np.sqrt,
            np.square,
            np.cbrt,
            np.reciprocal,
            np.sin,
            np.cos,
            np.tan,
            np.arcsin,
            np.arccos,
            np.arctan,
            np.sinh,
            np.cosh,
            np.tanh,
            np.arcsinh,
            np.arccosh,
            np.arctanh,
            np.degrees,
            np.radians,
            np.deg2rad,
            np.rad2deg,
            np.invert,
            np.logical_not,
            np.isfinite,
            np.isinf,
            np.isnan,
            np.isnat,
            np.signbit,
            np.spacing,
            np.modf,
            np.frexp,
            np.floor,
            np.ceil,
            np.trunc,
        ]
    )
    @pytest.mark.parametrize('out', [False, True])
    def test_ufunc_unary(
            self,
            ufunc: np.ufunc,
            array: na.AbstractArray,
            out: bool,
    ):
        dtypes = dict()
        for types in ufunc.types:
            dtype_inputs, dtype_outputs = types.split('->')
            if len(dtype_inputs) != 1:
                raise ValueError('This test is only valid for unary ufuncs')
            dtype_inputs = np.dtype(dtype_inputs)
            dtype_outputs = tuple(np.dtype(c) for c in dtype_outputs)
            dtypes[dtype_inputs] = dtype_outputs

        if array.dtype not in dtypes:
            with pytest.raises(TypeError):
                ufunc(array, casting ='no')
            return

        if out:
            type_array = array.type_array
            out = tuple(type_array.empty(array.shape, dtype=dtypes[array.dtype][i]) for i in range(ufunc.nout))
            out_ndarray = tuple(type_array.empty(array.shape, dtype=dtypes[array.dtype][i]).ndarray for i in range(ufunc.nout))
            if array.unit is not None and ufunc not in quantity_helpers.onearg_test_ufuncs:
                out = tuple(o << array.unit for o in out)
                out_ndarray = tuple(o << array.unit for o in out_ndarray)
        else:
            out = (None,) * ufunc.nout
            out_ndarray = (None,) * ufunc.nout
        out = out[0] if len(out) == 1 else out
        out_ndarray = out_ndarray[0] if len(out_ndarray) == 1 else out_ndarray

        ignored_ufuncs = tuple()
        ignored_ufuncs = ignored_ufuncs + quantity_helpers.dimensionless_to_dimensionless_ufuncs
        ignored_ufuncs = ignored_ufuncs + quantity_helpers.radian_to_dimensionless_ufuncs
        ignored_ufuncs = ignored_ufuncs + quantity_helpers.dimensionless_to_radian_ufuncs
        ignored_ufuncs = ignored_ufuncs + quantity_helpers.degree_to_radian_ufuncs
        ignored_ufuncs = ignored_ufuncs + quantity_helpers.radian_to_degree_ufuncs
        ignored_ufuncs = ignored_ufuncs + tuple(quantity_helpers.UNSUPPORTED_UFUNCS)
        ignored_ufuncs = ignored_ufuncs + (np.modf, np.frexp)

        if array.unit is not None and ufunc in ignored_ufuncs:
            with pytest.raises(TypeError):
                ufunc(array, out=out, casting='no')
            return

        if ufunc in [np.log, np.log2, np.log10, np.sqrt]:
            where = array > 0
        elif ufunc in [np.log1p]:
            where = array >= -1
        elif ufunc in [np.arcsin, np.arccos, np.arctanh]:
            where = (-1 <= array) & (array <= 1)
        elif ufunc in [np.arccosh]:
            where = array >= 1
        elif ufunc in [np.reciprocal]:
            where = array != 0
        else:
            where = na.ScalarArray(True)

        result = ufunc(array, out=out, where=where)
        result_ndarray = ufunc(
            array.ndarray,
            out=out_ndarray,
            casting='no',
            where=where.ndarray,
        )

        if ufunc.nout == 1:
            result = (result, )
            result_ndarray = (result_ndarray, )

        for i in range(ufunc.nout):
            assert np.all(result[i].ndarray == result_ndarray[i], where=where.ndarray)

    @pytest.mark.parametrize(
        argnames='ufunc',
        argvalues=[
            np.add,
            np.subtract,
            np.matmul,
            np.divide,
            np.logaddexp,
            np.logaddexp2,
            np.true_divide,
            np.floor_divide,
            np.power,
            np.float_power,
            np.remainder,
            np.mod,
            np.fmod,
            np.divmod,
            np.heaviside,
            np.gcd,
            np.lcm,
            np.arctan2,
            np.hypot,
            np.bitwise_and,
            np.bitwise_or,
            np.bitwise_xor,
            np.left_shift,
            np.right_shift,
            np.greater,
            np.greater_equal,
            np.less,
            np.less_equal,
            np.not_equal,
            np.equal,
            np.logical_and,
            np.logical_or,
            np.logical_xor,
            np.maximum,
            np.minimum,
            np.fmax,
            np.fmin,
            np.copysign,
            np.nextafter,
            # np.ldexp,
            np.fmod,
        ]
    )
    @pytest.mark.parametrize('out', [False, True])
    class TestUfuncBinary(
        abc.ABC,
    ):
        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):

            if ufunc is np.matmul:
                with pytest.raises(ValueError, match='np.matmul not supported*'):
                    ufunc(array, array_2, casting='no')
                return

            dtypes = dict()
            for types in ufunc.types:
                dtype_inputs, dtype_outputs = types.split('->')
                if not len(dtype_inputs) == 2:
                    raise TypeError('This test is only valid for binary ufuncs')
                dtype_inputs = tuple(np.dtype(c) for c in dtype_inputs)
                dtype_outputs = tuple(np.dtype(c) for c in dtype_outputs)
                dtypes[dtype_inputs] = dtype_outputs

            if (na.get_dtype(array), na.get_dtype(array_2)) not in dtypes:
                with pytest.raises(TypeError):
                    ufunc(array, casting='no')
                return

            shape = na.shape_broadcasted(array, array_2)
            shape = {k: shape[k] for k in sorted(shape)}

            unit = na.unit(array)
            unit_2 = na.unit(array_2)

            if out:
                type_array = na.type_array(array, array_2)
                dtype_in = (na.get_dtype(array), na.get_dtype(array_2))
                out = tuple(type_array.empty(shape, dtype=dtypes[dtype_in][i]) for i in range(ufunc.nout))
                out_ndarray = tuple(type_array.empty(shape, dtype=dtypes[dtype_in][i]).ndarray for i in range(ufunc.nout))
                if ufunc is np.copysign:
                    if unit is not None:
                        out = tuple(o << unit for o in out)
                        out_ndarray = tuple(o << unit for o in out_ndarray)
                elif unit is not None or unit_2 is not None:
                    if ufunc not in quantity_helpers.twoarg_comparison_ufuncs:
                        out = tuple(o << u.dimensionless_unscaled for o in out)
                        out_ndarray = tuple(o << u.dimensionless_unscaled for o in out_ndarray)
            else:
                out = (None,) * ufunc.nout
                out_ndarray = (None,) * ufunc.nout
            out = out[0] if len(out) == 1 else out
            out_ndarray = out_ndarray[0] if len(out_ndarray) == 1 else out_ndarray

            ignored_ufuncs = tuple()
            ignored_ufuncs = ignored_ufuncs + quantity_helpers.twoarg_invariant_ufuncs
            ignored_ufuncs = ignored_ufuncs + (np.floor_divide, np.divmod, )
            ignored_ufuncs = ignored_ufuncs + quantity_helpers.twoarg_invtrig_ufuncs
            ignored_ufuncs = ignored_ufuncs + quantity_helpers.twoarg_comparison_ufuncs

            if ufunc in quantity_helpers.UNSUPPORTED_UFUNCS:
                if unit is not None or unit_2 is not None:
                    with pytest.raises(TypeError):
                        ufunc(array, array_2, out=out, casting='no')
                    return

            if unit is not None:
                if unit_2 is None or not unit.is_equivalent(unit_2):
                    if ufunc in ignored_ufuncs:
                        with pytest.raises(u.UnitConversionError):
                            ufunc(array, array_2, out=out, casting='no')
                        return

            if unit_2 is not None:
                if unit is None or not unit_2.is_equivalent(unit):
                    if ufunc in ignored_ufuncs:
                        with pytest.raises(u.UnitConversionError):
                            ufunc(array, array_2, out=out, casting='no')
                        return

            if unit is not None and not unit.is_equivalent(u.dimensionless_unscaled):
                if ufunc in quantity_helpers.two_arg_dimensionless_ufuncs:
                    with pytest.raises(u.UnitTypeError):
                        ufunc(array, array_2, out=out)
                    return

            if unit_2 is not None and not unit_2.is_equivalent(u.dimensionless_unscaled):
                if ufunc in quantity_helpers.two_arg_dimensionless_ufuncs + (np.power, np.float_power, np.heaviside):
                    with pytest.raises(u.UnitTypeError):
                        ufunc(array, array_2, out=out, casting='no')
                    return

            if unit is not None and not unit.is_equivalent(u.dimensionless_unscaled):
                if ufunc in (np.power, np.float_power):
                    if array_2.shape:
                        with pytest.raises(
                                ValueError,
                                match="Quantities and Units may only be raised to a scalar power"
                        ):
                            ufunc(array, array_2, out=out, casting='no')
                    return

            if ufunc in [np.power, np.float_power]:
                where = (array_2 >= 1) & (array >= 0)
            elif ufunc in [np.divide, np.floor_divide, np.remainder, np.fmod, np.divmod]:
                where = array_2 != 0
                where = na.ScalarArray(where) if not isinstance(where, na.AbstractArray) else where
            else:
                where = na.ScalarArray(True)
            where = where.broadcast_to(shape)

            result = ufunc(array, array_2, out=out, where=where)
            result_ndarray = ufunc(
                array.ndarray if isinstance(array, na.AbstractArray) else array,
                np.transpose(array_2.ndarray) if isinstance(array_2, na.AbstractArray) else np.transpose(array_2),
                out=out_ndarray,
                casting='no',
                where=where.ndarray
            )

            if ufunc.nout == 1:
                result = (result,)
                result_ndarray = (result_ndarray,)

            for i in range(ufunc.nout):
                result_i = result[i].broadcast_to(shape)
                assert np.all(
                    result_i.ndarray == result_ndarray[i],
                    where=where.ndarray,
                )

        def test_ufunc_binary_reversed(
                self,
                ufunc: np.ufunc,
                array: na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):
            array = np.transpose(array)
            if array_2 is not None:
                array_2 = np.transpose(array_2)
            self.test_ufunc_binary(ufunc, array_2, array, out=out)

    class TestArrayFunctions:

        def test_broadcast_to(self, array: na.AbstractArray, shape: dict[str, int]):
            result = np.broadcast_to(array, shape=shape)
            assert result.shape == shape

        def test_shape(self, array: na.AbstractArray):
            assert np.shape(array) == array.shape

        def test_transpose(self, array: na.AbstractArray, axes: None | Sequence[str]):
            axes_normalized = tuple(reversed(array.axes)) if axes is None else axes
            result = np.transpose(
                a=array,
                axes=axes
            )
            assert result.axes == axes_normalized
            assert {ax: result.shape[ax] for ax in result.shape if ax in array.axes} == array.shape

        def test_moveaxis(
                self,
                array: na.AbstractArray,
                source: str | Sequence[str],
                destination: str | Sequence[str],
        ):
            source_normalized = (source, ) if isinstance(source, str) else source
            destination_normalized = (destination, ) if isinstance(destination, str) else destination

            if any(ax not in array.axes for ax in source_normalized):
                with pytest.raises(ValueError, match=r"source axis .* not in array axes .*"):
                    np.moveaxis(a=array, source=source, destination=destination)
                return

            result = np.moveaxis(a=array, source=source, destination=destination)

            assert np.all(array.ndarray == result.ndarray)
            assert len(array.axes) == len(result.axes)
            assert not any(ax in result.axes for ax in source_normalized)
            assert all(ax in result.axes for ax in destination_normalized)

        def test_array_equal(self, array: na.AbstractArray, array_2: None | na.AbstractArray):
            if array_2 is None:
                array_2 = array.copy()
                assert np.array_equal(array, array_2)
                return

            if not array.unit_normalized.is_equivalent(array_2.unit_normalized):
                with pytest.raises(u.UnitConversionError):
                    np.array_equal(array, array_2)
                return

            if array.shape and np.issubdtype(array.dtype, str) and not np.issubdtype(array_2, str):
                with pytest.raises(FutureWarning, match="elementwise comparison failed"):
                    np.array_equal(array, array_2)

            else:
                if array.unit_normalized.is_equivalent(array_2.unit_normalized):
                    assert not np.array_equal(array, array_2)
                else:
                    with pytest.raises(u.UnitConversionError):
                        np.array_equal(array, array_2)

        @pytest.mark.parametrize(
            argnames='func',
            argvalues=[
                np.all,
                np.any,
                np.max,
                np.nanmax,
                np.min,
                np.nanmin,
                np.sum,
                np.nansum,
                np.prod,
                np.nanprod,
                np.mean,
                np.nanmean,
                np.std,
                np.nanstd,
                np.var,
                np.nanvar,
                np.median,
                np.nanmedian,
            ]
        )
        @pytest.mark.parametrize('dtype', [None, float])
        @pytest.mark.parametrize('out', [False, True])
        @pytest.mark.parametrize('keepdims', [False, True])
        @pytest.mark.parametrize('where', [False, True])
        class TestReductionFunctions:
            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: None | str | Sequence[str],
                    dtype: Type,
                    out: bool,
                    keepdims: bool,
                    where: bool,
            ):
                kwargs = dict()
                kwargs_ndarray = dict()

                axis_normalized = axis if axis is not None else array.axes
                axis_normalized = (axis_normalized, ) if isinstance(axis_normalized, str) else axis_normalized
                shape_result = {ax: 1 if ax in axis_normalized else array.shape[ax] for ax in reversed(array.shape)}

                if dtype is not None:
                    kwargs['dtype'] = dtype
                    kwargs_ndarray['dtype'] = dtype

                if keepdims:
                    kwargs['keepdims'] = keepdims
                    kwargs_ndarray['keepdims'] = keepdims
                else:
                    for ax in axis_normalized:
                        if ax in shape_result:
                            shape_result.pop(ax)

                if out:
                    kwargs['out'] = array.type_array.empty(shape_result)
                    kwargs_ndarray['out'] = array.type_array.empty(shape_result).ndarray.transpose()
                    if array.unit is not None:
                        kwargs['out'] = kwargs['out'] << array.unit
                        kwargs_ndarray['out'] = kwargs_ndarray['out'] << array.unit

                if where:
                    if array.shape:
                        kwargs['where'] = na.ScalarArray(
                            ndarray=np.random.choice([False, True], size=np.shape(array.ndarray)),
                            axes=array.axes,
                        )
                        kwargs['where'][{ax: 0 for ax in axis_normalized if ax in kwargs['where'].axes}] = True
                    else:
                        kwargs['where'] = na.ScalarArray(True)
                    kwargs_ndarray['where'] = kwargs['where'].ndarray
                    if func in [np.min, np.nanmin, np.max, np.nanmax]:
                        kwargs['initial'] = 0
                        kwargs_ndarray['initial'] = kwargs['initial']

                if dtype is not None:
                    if func in [np.all, np.any, np.max, np.nanmax, np.min, np.nanmin, np.median, np.nanmedian]:
                        with pytest.raises(
                                expected_exception=TypeError,
                                match=r".* got an unexpected keyword argument .*"
                        ):
                            func(array, axis=axis, **kwargs, )
                        return

                if array.unit is not None:
                    if func in [np.all, np.any]:
                        if 'where' in kwargs:
                            kwargs.pop('where')
                        with pytest.raises(
                                expected_exception=TypeError,
                                match=r"(no implementation found for *)|(cannot evaluate truth value of quantities. *)"
                        ):
                            func(array, axis=axis, **kwargs, )
                        return

                    if func in [np.prod, np.nanprod]:
                        with pytest.raises(u.UnitsError):
                            func(array, axis=axis, **kwargs, )
                        return

                if func in [np.median, np.nanmedian]:
                    if where:
                        with pytest.raises(TypeError, match=r".* got an unexpected keyword argument \'where\'"):
                            func(array, axis=axis, **kwargs, )
                        return

                if np.issubdtype(array.dtype, str):
                    if dtype is None:
                        with pytest.raises(
                                expected_exception=TypeError,
                                match=r"(ufunc .* did not contain a loop with signature matching types .*)|"
                                      r"(ufunc .* not supported for the input types, *)"
                        ):
                            func(array, axis=axis, **kwargs, )
                        return

                    else:
                        with pytest.raises(
                            expected_exception=ValueError,
                            match=r"could not convert string to .*"
                        ):
                            func(array, axis=axis, **kwargs)
                        return

                if func in [np.median, np.nanmedian]:
                    if out and keepdims and np.ndim(array) != 0 and (set(axis_normalized) & set(array.axes)):
                        shape_result_2 = {ax: shape_result[ax] for ax in shape_result if ax not in axis_normalized}
                        try:
                            np.empty(tuple(shape_result.values())).T[...] = np.empty(tuple(shape_result_2.values())).T
                            broadcastable = True
                        except ValueError:
                            broadcastable = False

                        if func in [np.nanmedian] and broadcastable:
                            pass
                        else:
                            with pytest.raises(
                                    expected_exception=ValueError,
                                    match=r"(output parameter for reduction operation add has the wrong number of *)|"
                                          r"(could not broadcast input array from shape .* into shape .*)"
                            ):
                                func(array, axis=axis, **kwargs, )
                            return

                result = func(array, axis=axis, **kwargs, )
                result_ndarray = func(
                    array.ndarray,
                    axis=tuple(array.axes.index(ax) for ax in axis_normalized if ax in array.axes),
                    **kwargs_ndarray,
                )
                assert np.all(result.ndarray == result_ndarray)

        @pytest.mark.parametrize(
            argnames='func',
            argvalues=[
                np.argmin,
                np.nanargmin,
                np.argmax,
                np.nanargmax,
            ]
        )
        @pytest.mark.parametrize('out', [False, True])
        @pytest.mark.parametrize('keepdims', [False, True])
        class TestArgReductionFunctions:
            def test_arg_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: None | str,
                    out: bool,
                    keepdims: bool,
            ):
                kwargs = dict()
                kwargs_ndarray = dict()

                axis_normalized = array.axes_flattened if axis is None else axis

                if axis is not None:
                    shape_result = {ax: 1 if ax == axis else array.shape[ax] for ax in reversed(array.shape)}
                    if not keepdims:
                        shape_result.pop(axis, None)
                else:
                    if keepdims:
                        shape_result = {ax: 1 for ax in reversed(array.shape)}
                    else:
                        shape_result = dict()

                if keepdims:
                    kwargs['keepdims'] = keepdims
                    kwargs_ndarray['keepdims'] = keepdims

                if out:
                    kwargs['out'] = {axis_normalized: array.type_array.empty(shape_result, dtype=int)}
                    kwargs_ndarray['out'] = array.type_array.empty(shape_result, dtype=int).ndarray.transpose()

                if axis is not None:
                    if axis not in array.axes:
                        with pytest.raises(ValueError, match='Reduction axis .* not in array with axes .*'):
                            func(array, axis=axis, **kwargs)
                        return

                result = func(array, axis=axis, **kwargs)
                result_ndarray = func(
                    array.ndarray,
                    axis=array.axes.index(axis) if axis is not None else axis,
                    **kwargs_ndarray
                )

                assert len(result) == 1
                assert np.all(result[axis_normalized].ndarray == result_ndarray)


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
        assert array.num == array.shape[array.axis]


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
        assert np.all(np.abs(array.range) > 0)


class AbstractTestAbstractSymmetricRange(
    AbstractTestAbstractRange,
):
    def test_center(self, array: na.AbstractSymmetricRange):
        assert isinstance(array.center, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_width(self, array: na.AbstractSymmetricRange):
        assert isinstance(array.width, (int, float, complex, u.Quantity, na.AbstractArray))
        assert np.all(array.width > 0)


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
