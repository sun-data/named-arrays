from __future__ import annotations
from typing import Type, Sequence, Callable
import pytest
import numpy as np
import astropy.units as u
import astropy.units.quantity_helper.helpers as quantity_helpers
import named_arrays as na
from ... import tests

__all__ = [
    'AbstractTestAbstractScalar',
    'AbstractTestAbstractScalarArray',
    'TestScalarArray',
    'AbstractTestAbstractParameterizedScalarArray',
]

_num_x = tests.test_core.num_x
_num_y = tests.test_core.num_y


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
    return arrays_numeric + arrays_bool


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
    return [None] + arrays_numeric + arrays_bool


@pytest.mark.parametrize('value', _scalar_arrays_2())
def test_as_named_array(value: bool | int | float | complex | str | u.Quantity | na.AbstractArray):
    result = na.as_named_array(value)
    assert isinstance(result, na.AbstractArray)


class AbstractTestAbstractScalar(
    tests.test_core.AbstractTestAbstractArray,
):

    def test_ndarray(self, array: na.AbstractScalar):
        assert isinstance(array.ndarray, (int, float, complex, str, np.ndarray))

    def test_dtype(self, array: na.AbstractScalar):
        super().test_dtype(array)
        assert isinstance(array.dtype, np.dtype)

    def test_unit(self, array: na.AbstractScalar):
        assert array.unit == array.array.unit
        unit = array.unit
        if unit is not None:
            assert isinstance(unit, u.UnitBase)

    def test_unit_normalized(self, array: na.AbstractScalar):
        assert array.unit_normalized == array.array.unit_normalized
        if array.unit is None:
            assert array.unit_normalized == u.dimensionless_unscaled
        else:
            assert array.unit_normalized == array.unit

    @pytest.mark.parametrize('unit', [u.m, u.s])
    def test_to(self, array: na.AbstractScalar, unit: None | u.UnitBase):
        super().test_to(array=array, unit=unit)
        if isinstance(array.unit, u.UnitBase) and array.unit.is_equivalent(unit):
            array_new = array.to(unit)
            assert array_new.unit == unit
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test__getitem__(
            self,
            array: na.AbstractScalar,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

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

    class TestUfuncUnary(
        tests.test_core.AbstractTestAbstractArray.TestUfuncUnary,
    ):

        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractArray,
                out: bool,
        ):
            super().test_ufunc_unary(ufunc=ufunc, array=array, out=out)

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
                out = (out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].ndarray == result_ndarray[i], where=where.ndarray)

                if out[i] is not None:
                    assert result[i] is out[i]

    class TestUfuncBinary(
        tests.test_core.AbstractTestAbstractArray.TestUfuncBinary,
    ):
        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):

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
                out = (out, )

            for i in range(ufunc.nout):
                result_i = result[i].broadcast_to(shape)
                assert np.all(
                    result_i.ndarray == result_ndarray[i],
                    where=where.ndarray,
                )

                if out[i] is not None:
                    assert result[i] is out[i]

    class TestMatmul(
        tests.test_core.AbstractTestAbstractArray.TestMatmul,
    ):

        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):
            shape = na.shape_broadcasted(array, array_2)
            shape = {k: shape[k] for k in sorted(shape)}

            if out:
                type_array = na.type_array(array, array_2)
                out = type_array.empty(shape)
                out_expected = type_array.empty(shape).ndarray
                unit, unit_2 = na.unit(array), na.unit(array_2)
                if unit is not None:
                    out = out << unit
                    out_expected = out_expected << unit
                if unit_2 is not None:
                    out = out * unit_2
                    out_expected = out_expected * unit_2
            else:
                out = None
                out_expected = None

            result = np.matmul(array, array_2, out=out)
            if array is None or array_2 is None:
                assert result is None
            else:
                result = result.broadcast_to(shape)
                result_expected = np.multiply(
                    array.ndarray if isinstance(array, na.AbstractArray) else array,
                    np.transpose(array_2.ndarray) if isinstance(array_2, na.AbstractArray) else array_2,
                    out=out_expected,
                )

                assert np.all(result.ndarray == result_expected)

    class TestArrayFunctions(
        tests.test_core.AbstractTestAbstractArray.TestArrayFunctions
    ):

        class TestReductionFunctions(
            tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestReductionFunctions
        ):
            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractScalar,
                    axis: None | str | Sequence[str],
                    dtype: Type,
                    out: bool,
                    keepdims: bool,
                    where: bool,
            ):
                super().test_reduction_functions(
                    func=func,
                    array=array,
                    axis=axis,
                    dtype=dtype,
                    out=out,
                    keepdims=keepdims,
                    where=where,
                )

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

                result = func(array, axis=axis, **kwargs, )
                result_ndarray = func(
                    array.ndarray,
                    axis=tuple(array.axes.index(ax) for ax in axis_normalized if ax in array.axes),
                    **kwargs_ndarray,
                )
                assert np.all(result.ndarray == result_ndarray)

        class TestPercentileLikeFunctions(
            tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestPercentileLikeFunctions
        ):

            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    out: bool,
                    keepdims: bool,
            ):
                super().test_percentile_like_functions(
                    func=func,
                    array=array.array,
                    q=q,
                    axis=axis,
                    out=out,
                    keepdims=keepdims,
                )

                kwargs = dict()
                kwargs_ndarray = dict()

                q_normalized = q if isinstance(q, na.AbstractArray) else na.ScalarArray(q)

                axis_normalized = axis if axis is not None else array.axes
                axis_normalized = (axis_normalized, ) if isinstance(axis_normalized, str) else axis_normalized
                shape_result = q_normalized.shape
                shape_result |= {ax: 1 if ax in axis_normalized else array.shape[ax] for ax in array.shape}

                if keepdims:
                    kwargs['keepdims'] = keepdims
                    kwargs_ndarray['keepdims'] = keepdims
                else:
                    for ax in axis_normalized:
                        if ax in shape_result:
                            shape_result.pop(ax)

                if out:
                    out_dtype = na.get_dtype(array)
                    kwargs['out'] = array.type_array.empty(shape_result, dtype=out_dtype)
                    kwargs_ndarray['out'] = array.type_array.empty(shape_result, dtype=out_dtype).ndarray
                    if array.unit is not None:
                        kwargs['out'] = kwargs['out'] << array.unit
                        kwargs_ndarray['out'] = kwargs_ndarray['out'] << array.unit
                    elif q_normalized.unit is not None:
                        kwargs['out'] = kwargs['out'] << u.dimensionless_unscaled
                        kwargs_ndarray['out'] = kwargs_ndarray['out'] << u.dimensionless_unscaled

                kwargs['method'] = 'closest_observation'
                kwargs_ndarray['method'] = kwargs['method']

                result = func(array, q, axis=axis, **kwargs, )
                result_ndarray = func(
                    array.ndarray,
                    q_normalized.ndarray,
                    axis=tuple(array.axes.index(ax) for ax in axis_normalized if ax in array.axes),
                    **kwargs_ndarray,
                )
                assert np.all(result.ndarray == result_ndarray)

        class TestFFTLikeFunctions(
            tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTLikeFunctions
        ):

            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractScalar,
                    axis: str,
            ):
                super().test_fft_like_functions(
                    func=func,
                    array=array,
                    axis=axis,
                )

                if axis[0] not in array.axes:
                    with pytest.raises(ValueError):
                        func(a=array, axis=axis)
                    return

                result = func(
                    a=array,
                    axis=axis,
                )
                result_expected = func(
                    a=array.ndarray,
                    axis=array.axes.index(axis[0])
                )

                assert isinstance(result, na.AbstractArray)
                assert axis[1] in result.axes
                assert not axis[0] in result.axes

                assert np.all(result.ndarray == result_expected)

        class TestFFTNLikeFunctions(
            tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTNLikeFunctions
        ):

            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axes: dict[str, str],
                    s: None | dict[str, int],
            ):
                super().test_fftn_like_functions(
                    func=func,
                    array=array,
                    axes=axes,
                    s=s,
                )

                shape = array.shape

                if not all(ax in shape for ax in axes):
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                if func in [np.fft.rfft2, np.fft.irfft2, np.fft.rfftn, np.fft.irfftn]:
                    if not shape:
                        with pytest.raises(IndexError):
                            func(a=array, axes=axes, s=s)
                        return

                if s is not None and axes.keys() != s.keys():
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                if s is None:
                    s_normalized = {ax: shape[ax] for ax in axes}
                else:
                    s_normalized = {ax: s[ax] for ax in axes}

                result = func(
                    a=array,
                    axes=axes,
                    s=s,
                )
                result_expected = func(
                    a=array.ndarray,
                    s=s_normalized.values(),
                    axes=[array.axes.index(ax) for ax in axes],
                )

                assert isinstance(result, na.AbstractArray)
                assert all(axes[ax] in result.axes for ax in axes)
                assert all(ax not in result.axes for ax in axes)

                assert np.all(result.ndarray == result_expected)

        @pytest.mark.parametrize('axis', [None, 'x', 'y'])
        def test_sort(self, array: na.AbstractScalar, axis: None | str):

            super().test_sort(array=array, axis=axis)

            if axis is not None and axis not in array.axes:
                with pytest.raises(ValueError, match="axis .* not in input array with axes .*"):
                    np.sort(a=array, axis=axis)
                return

            result = np.sort(a=array, axis=axis)
            result_ndarray = np.sort(
                a=array.ndarray,
                axis=array.axes.index(axis) if axis is not None else axis,
            )

            assert np.all(result.ndarray == result_ndarray)

        def test_nonzero(self, array: na.AbstractArray):
            if not array.shape:
                with pytest.raises(DeprecationWarning, match="Calling nonzero on 0d arrays is deprecated, .*"):
                    np.nonzero(array)
                return

            result = np.nonzero(array)
            expected = np.nonzero(array.ndarray)

            for i, ax in enumerate(array.axes):
                assert np.all(result[ax].ndarray == expected[i])
                assert len(result[ax].axes) == 1
                assert result[ax].axes[0] == f"{array.axes_flattened}_nonzero"

        @pytest.mark.parametrize('copy', [False, True])
        def test_nan_to_num(self, array: na.AbstractArray, copy: bool):

            super().test_nan_to_num(array=array, copy=copy)

            if not copy and not isinstance(array, na.AbstractExplicitArray):
                with pytest.raises(ValueError, match="can't write to an array .*"):
                    np.nan_to_num(array, copy=copy)
                return

            result = np.nan_to_num(array, copy=copy)
            expected = np.nan_to_num(array.ndarray, copy=copy)

            if not copy:
                assert result is array

            assert np.all(result.ndarray == expected)


class AbstractTestAbstractScalarArray(
    AbstractTestAbstractScalar,
):
    def test_axes(self, array: na.AbstractArray):
        super().test_axes(array)
        assert len(array.axes) == np.ndim(array.ndarray)

    def test_size(self, array: na.AbstractScalarArray):
        super().test_size(array)
        assert array.size == np.size(array.ndarray)

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

    @pytest.mark.parametrize('array_2', _scalar_arrays_2())
    class TestMatmul(
        AbstractTestAbstractScalar.TestMatmul,
    ):
        pass

    class TestArrayFunctions(
        AbstractTestAbstractScalar.TestArrayFunctions,
    ):

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                .25,
                25 * u.percent,
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
            ]
        )
        class TestPercentileLikeFunctions(
            AbstractTestAbstractScalar.TestArrayFunctions.TestPercentileLikeFunctions
        ):
            pass


@pytest.mark.parametrize('array', _scalar_arrays())
class TestScalarArray(
    AbstractTestAbstractScalarArray,
    tests.test_core.AbstractTestAbstractExplicitArray,
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
    tests.test_core.AbstractTestAbstractExplicitArrayCreation
):

    @property
    def type_array(self) -> Type[na.ScalarArray]:
        return na.ScalarArray


class AbstractTestAbstractImplicitScalarArray(
    AbstractTestAbstractScalarArray,
    tests.test_core.AbstractTestAbstractImplicitArray,
):
    pass


class AbstractTestAbstractScalarRandomSample(
    AbstractTestAbstractImplicitScalarArray,
    tests.test_core.AbstractTestAbstractRandomSample,
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
    shapes_random = [dict(y=_num_y)]
    return [
        na.ScalarUniformRandomSample(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            shape_random=shape_random,
        ) for start in starts for stop in stops for unit in units for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _scalar_uniform_random_samples())
class TestScalarUniformRandomSample(
    AbstractTestAbstractScalarRandomSample,
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
    shapes_random = [dict(y=_num_y)]
    return [
        na.ScalarNormalRandomSample(
            center=center << unit if unit is not None else center,
            width=width << unit if unit is not None else width,
            shape_random=shape_random,
        ) for center in centers for width in widths for unit in units for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _scalar_normal_random_samples())
class TestScalarNormalRandomSample(
    AbstractTestAbstractScalarRandomSample,
    tests.test_core.AbstractTestAbstractNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedScalarArray(
    AbstractTestAbstractImplicitScalarArray,
    tests.test_core.AbstractTestAbstractParameterizedArray,
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
    AbstractTestAbstractParameterizedScalarArray,
    tests.test_core.AbstractTestAbstractArrayRange,
):
    pass


class AbstractTestAbstractScalarSpace(
    AbstractTestAbstractParameterizedScalarArray,
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

