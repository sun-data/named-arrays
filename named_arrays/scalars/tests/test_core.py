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
_num_distribution = tests.test_core.num_distribution


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
    arrays_bool[0][dict(y=0)] = True
    arrays_bool[1][dict(x=0, y=0)] = True
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

    def test_dtype(self, array: na.AbstractScalar):
        assert array.dtype == array.array.dtype
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

    @pytest.mark.parametrize('dtype', [int, float])
    def test_astype(self, array: na.AbstractScalar, dtype: Type):
        super().test_astype(array=array, dtype=dtype)
        array_new = array.astype(dtype)
        assert array_new.dtype == dtype

    @pytest.mark.parametrize('unit', [u.m, u.s])
    def test_to(self, array: na.AbstractScalar, unit: None | u.UnitBase):
        super().test_to(array=array, unit=unit)
        if isinstance(array.unit, u.UnitBase) and array.unit.is_equivalent(unit):
            array_new = array.to(unit)
            assert array_new.unit == unit
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test_length(self, array: na.AbstractScalar):
        super().test_length(array=array)
        if not np.issubdtype(array.dtype, np.number):
            with pytest.raises(ValueError):
                array.length
            return

        length = array.length
        assert isinstance(length, (int, float, np.ndarray, na.AbstractScalar))
        assert np.all(length >= 0)

    def test__bool__(self, array: na.AbstractScalarArray):
        if array.shape or array.unit is not None:
            with pytest.raises(
                expected_exception=ValueError,
                match=r"(Quantity truthiness is ambiguous, .*)"
                      r"|(The truth value of an array with more than one element is ambiguous. .*)"
            ):
                bool(array)
            return

        result = bool(array)
        assert isinstance(result, bool)

    class TestMatmul(
        tests.test_core.AbstractTestAbstractArray.TestMatmul,
    ):

        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            result = np.matmul(array, array_2)

            if array is None or array_2 is None:
                assert result is None
                return

            result_expected = np.multiply(array, array_2)

            out = 0 * result
            result_out = np.matmul(array, array_2, out=out)

            assert np.all(result == result_expected)
            assert np.all(result == result_out)
            assert result_out is out

    def test_ptp(
            self,
            array: na.AbstractScalar,
    ):
        super().test_ptp(array=array)
        if np.issubdtype(array.dtype, bool):
            with pytest.raises(TypeError, match='numpy boolean subtract, .*'):
                array.ptp()
            return

        assert np.all(array.ptp() == np.ptp(array))

    def test_all(
            self,
            array: na.AbstractScalar,
    ):
        super().test_all(array=array)
        if array.unit is not None:
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.all()
            return

        assert np.all(array.all() == np.all(array))

    def test_any(
            self,
            array: na.AbstractScalar,
    ):
        super().test_any(array=array)
        if array.unit is not None:
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.any()
            return

        assert np.all(array.any() == np.any(array))


class AbstractTestAbstractScalarArray(
    AbstractTestAbstractScalar,
):

    def test_ndarray(self, array: na.AbstractScalarArray):
        assert isinstance(array.ndarray, (int, float, complex, str, np.ndarray))

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
            array: na.AbstractScalarArray,
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

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
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
            na.UniformUncertainScalarArray(
                nominal=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                width=.1,
                num_distribution=_num_distribution,
            ) > 0.5,
        ]
    )
    def test__getitem__uncertain(
            self,
            array: na.AbstractScalarArray,
            item: dict[str, na.UncertainScalarArray] | na.UncertainScalarArray
    ):

        if isinstance(item, dict):

            if not (set(item) - set((na.UncertainScalarArray.axis_distribution, ))).issubset(array.shape):
                with pytest.raises(
                    expected_exception=ValueError,
                    match="the axes in item, .*, must be a subset of the axes in the array, .*"
                ):
                    array[item]
                return

            num_distribution = item[na.UncertainScalarArray.axis_distribution].distribution.max().ndarray + 1
            shape_distribution = na.broadcast_shapes(
                array.shape,
                {na.UncertainScalarArray.axis_distribution: num_distribution}
            )

            result_expected = na.UncertainScalarArray(
                nominal=array[{ax: item[ax].nominal for ax in item}],
                distribution=array.broadcast_to(shape_distribution)[{ax: item[ax].distribution for ax in item}],
            )
        else:
            if not set(item.shape).issubset(array.shape):
                with pytest.raises(
                    expected_exception=ValueError,
                    match="the axes in item, .*, must be a subset of the axes in array, .*"
                ):
                    array[item]
                return

            result_expected = array[item.nominal & np.all(item.distribution, axis=item.axis_distribution)]

        result = array[item]

        assert np.all(result == result_expected)

    def test__mul__(self, array: na.AbstractScalarArray):
        unit = u.mm
        result = array * unit
        result_ndarray = array.ndarray * unit
        assert np.all(result.ndarray == result_ndarray)

    def test__lshift__(self, array: na.AbstractScalarArray):
        unit = u.mm
        result = array << unit
        result_ndarray = array.ndarray << unit
        assert np.all(result.ndarray == result_ndarray)

    def test__truediv__(self, array: na.AbstractScalarArray):
        unit = u.mm
        result = array / unit
        result_ndarray = array.ndarray / unit
        assert np.all(result.ndarray == result_ndarray)

    class TestUfuncUnary(
        AbstractTestAbstractScalar.TestUfuncUnary,
    ):

        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractScalarArray,
        ):
            super().test_ufunc_unary(ufunc, array)

            kwargs = dict()
            kwargs_ndarray = dict()

            unit = array.unit_normalized
            if ufunc in [np.log, np.log2, np.log10, np.sqrt]:
                kwargs["where"] = array > 0
            elif ufunc in [np.log1p]:
                kwargs["where"] = array >= (-1 * unit)
            elif ufunc in [np.arcsin, np.arccos, np.arctanh]:
                kwargs["where"] = ((-1 * unit) < array) & (array < (1 * unit))
            elif ufunc in [np.arccosh]:
                kwargs["where"] = array >= (1 * unit)
            elif ufunc in [np.reciprocal]:
                kwargs["where"] = array != 0

            if "where" in kwargs:
                kwargs_ndarray["where"] = kwargs["where"].ndarray

            try:
                ufunc(array.ndarray, **kwargs_ndarray)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    ufunc(array, **kwargs)
                return

            result = ufunc(array, **kwargs)
            result_ndarray = ufunc(array.ndarray, **kwargs_ndarray)

            if ufunc.nout == 1:
                out = 0 * result
            else:
                out = tuple(0 * r for r in result)

            result_out = ufunc(array, out=out, **kwargs)

            if ufunc.nout == 1:
                out = (out, )
                result_ndarray = (result_ndarray, )
                result = (result, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].ndarray == result_ndarray[i], **kwargs_ndarray)
                assert np.all(result[i] == result_out[i], **kwargs)
                assert result_out[i] is out[i]

    @pytest.mark.parametrize('array_2', _scalar_arrays_2())
    class TestUfuncBinary(
        AbstractTestAbstractScalar.TestUfuncBinary,
    ):

        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractScalarArray,
                array_2: None | bool | int | float | complex | str | na.AbstractScalarArray,
        ):
            super().test_ufunc_binary(ufunc, array, array_2)

            if array is None or array_2 is None:
                assert ufunc(array, array_2) is None
                return

            array_ndarray = array.ndarray if isinstance(array, na.AbstractArray) else array
            array_2_ndarray = array_2.ndarray if isinstance(array_2, na.AbstractArray) else array_2
            array_2_ndarray = np.transpose(array_2_ndarray)

            kwargs = dict()
            kwargs_ndarray = dict()

            unit_2 = na.unit_normalized(array_2)
            if ufunc in [np.power, np.float_power]:
                kwargs["where"] = (array_2 >= (1 * unit_2)) & (array >= 0)
            elif ufunc in [np.divide, np.floor_divide, np.remainder, np.fmod, np.divmod]:
                kwargs["where"] = array_2 != 0

            shape = na.shape_broadcasted(array, array_2)
            shape = {ax: shape[ax] for ax in sorted(shape)}
            if "where" in kwargs:
                kwargs["where"] = na.broadcast_to(kwargs["where"], shape)
                kwargs_ndarray["where"] = kwargs["where"].ndarray

            try:
                ufunc(array_ndarray, array_2_ndarray, **kwargs_ndarray)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    ufunc(array, array_2, **kwargs)
                return

            result = ufunc(array, array_2, **kwargs)
            result_ndarray = ufunc(array_ndarray, array_2_ndarray, **kwargs_ndarray)

            if ufunc.nout == 1:
                out = 0 * np.nan_to_num(result)
            else:
                out = tuple(0 * np.nan_to_num(r) for r in result)

            result_out = ufunc(array, array_2, out=out, **kwargs)

            if ufunc.nout == 1:
                out = (out,)
                result_ndarray = (result_ndarray,)
                result = (result,)
                result_out = (result_out,)

            for i in range(ufunc.nout):
                assert np.all(result[i].broadcast_to(shape).ndarray == result_ndarray[i], **kwargs_ndarray)
                assert np.all(result[i] == result_out[i], **kwargs)
                assert result_out[i] is out[i]

    @pytest.mark.parametrize('array_2', _scalar_arrays_2())
    class TestMatmul(
        AbstractTestAbstractScalar.TestMatmul,
    ):
        pass

    class TestArrayFunctions(
        AbstractTestAbstractScalar.TestArrayFunctions,
    ):

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
                (na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
            ]
        )
        class TestReductionFunctions(
            AbstractTestAbstractScalar.TestArrayFunctions.TestReductionFunctions
        ):

            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractScalarArray,
                    axis: None | str | Sequence[str],
                    dtype: None | type | np.dtype,
                    keepdims: bool,
                    where: bool | na.AbstractArray,
            ):
                super().test_reduction_functions(
                    func=func,
                    array=array,
                    axis=axis,
                    dtype=dtype,
                    keepdims=keepdims,
                    where=where,
                )

                kwargs = dict(
                    axis=axis,
                    keepdims=keepdims,
                    where=where,
                )

                axis_normalized = na.axis_normalized(array, axis=axis)

                if axis is not None:
                    if not set(axis_normalized).issubset(array.axes):
                        with pytest.raises(
                                ValueError, match=r"the `axis` argument must be `None` or a subset of"):
                            func(array, axis=axis)
                        return

                kwargs_ndarray = dict(
                    axis=tuple(array.axes.index(ax) for ax in axis_normalized),
                    keepdims=keepdims,
                    where=where.ndarray if isinstance(where, na.AbstractArray) else where,
                )

                if dtype is not np._NoValue:
                    kwargs["dtype"] = kwargs_ndarray["dtype"] = dtype

                if func in [np.min, np.nanmin, np.max, np.nanmax]:
                    kwargs["initial"] = kwargs_ndarray["initial"] = 0

                try:
                    result_ndarray = func(array.ndarray, **kwargs_ndarray)
                except (ValueError, TypeError, u.UnitsError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.ndarray == result_ndarray)
                assert np.allclose(result, result_out)
                assert result_out is out

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

            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractScalarArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    keepdims: bool,
            ):
                super().test_percentile_like_functions(
                    func=func,
                    array=array.array,
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                kwargs = dict(
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                q_normalized = q if isinstance(q, na.AbstractArray) else na.ScalarArray(q)
                axis_normalized = na.axis_normalized(array, axis)

                if axis is not None:
                    if not set(axis_normalized).issubset(array.axes):
                        with pytest.raises(
                                ValueError, match=r"the `axis` argument must be `None` or a subset of"):
                            func(array, **kwargs)
                        return

                kwargs_ndarray = dict(
                    q=q_normalized.ndarray,
                    axis=tuple(array.axes.index(ax) for ax in axis_normalized),
                    keepdims=keepdims,
                )

                try:
                    result_ndarray = func(array.ndarray, **kwargs_ndarray)
                except (ValueError, TypeError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.ndarray == result_ndarray)
                assert np.allclose(result, result_out)
                assert result_out is out

        class TestFFTLikeFunctions(
            AbstractTestAbstractScalar.TestArrayFunctions.TestFFTLikeFunctions
        ):

            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractScalarArray,
                    axis: tuple[str, str],
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
            AbstractTestAbstractScalar.TestArrayFunctions.TestFFTNLikeFunctions
        ):

            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractScalarArray,
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

        @pytest.mark.parametrize('axis', [None, 'x', 'y', ('x', 'y')])
        def test_sort(self, array: na.AbstractScalarArray, axis: None | str | Sequence[str]):

            super().test_sort(array=array, axis=axis)

            axis_normalized = na.axis_normalized(array, axis)

            if axis is not None:
                if not set(axis_normalized).issubset(array.axes):
                    with pytest.raises(
                            expected_exception=ValueError,
                            match="`axis`, .* is not a subset of `a.axes`, .*"
                    ):
                        np.sort(array, axis=axis)
                    return
            else:
                if not array.shape:
                    result = np.sort(a=array, axis=axis)
                    assert np.all(result == array)
                return

            axis_flattened = na.flatten_axes(axis_normalized)
            array_normalized = array.combine_axes(
                axes=axis_normalized,
                axis_new=axis_flattened,
            )

            result = np.sort(a=array, axis=axis)
            result_ndarray = np.sort(
                a=array_normalized.ndarray,
                axis=array_normalized.axes.index(axis_flattened) if axis is not None else axis,
            )

            assert np.all(result.ndarray == result_ndarray)

        def test_nonzero(self, array: na.AbstractScalarArray):

            super().test_nonzero(array)

            result = np.nonzero(array)

            if not array.shape:
                assert result == dict()
                return

            expected = np.nonzero(array.ndarray)

            for i, ax in enumerate(array.axes):
                assert np.all(result[ax].ndarray == expected[i])
                assert len(result[ax].axes) == 1
                assert result[ax].axes[0] == array.axes_flattened

        @pytest.mark.parametrize('copy', [False, True])
        def test_nan_to_num(self, array: na.AbstractScalarArray, copy: bool):

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

        @pytest.mark.parametrize('v', _scalar_arrays_2())
        @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
        def test_convolve(self, array: na.AbstractArray, v: na.AbstractArray, mode: str):
            super().test_convolve(array=array, v=v, mode=mode)

            shape_broadcasted = na.shape_broadcasted(array, v)

            if v is None:
                assert np.convolve(array, v, mode=mode) == na.ScalarArray(None)
                return

            if len(shape_broadcasted) > 1:
                with pytest.raises(ValueError, match=r"\'a\' and \'v\' must broadcast to .*"):
                    np.convolve(array, v, mode=mode)
                return

            result = np.convolve(array, v, mode=mode)
            result_expected = np.convolve(
                array.ndarray,
                v.ndarray if isinstance(v, na.AbstractArray) else v,
                mode=mode,
            )

            assert result.axes == tuple(shape_broadcasted.keys())
            assert np.all(result.ndarray == result_expected)


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

    def test_empty(self, shape: dict[str, int], dtype: Type):
        super().test_empty(shape=shape, dtype=dtype)
        result = self.type_array.empty(shape, dtype=dtype)
        assert result.dtype == dtype

    def test_zeros(self, shape: dict[str, int], dtype: Type):
        super().test_zeros(shape=shape, dtype=dtype)
        result = self.type_array.zeros(shape, dtype=dtype)
        assert result.dtype == dtype

    def test_ones(self, shape: dict[str, int], dtype: Type):
        super().test_ones(shape=shape, dtype=dtype)
        result = self.type_array.ones(shape, dtype=dtype)
        assert result.dtype == dtype


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


def _scalar_poisson_random_samples() -> list[na.ScalarPoissonRandomSample]:
    centers = [
        4,
        na.ScalarArray(np.random.randint(1, 5, _num_x), axes=('x', )),
    ]
    units = [None, u.mm]
    shapes_random = [dict(y=_num_y)]
    return [
        na.ScalarPoissonRandomSample(
            center=center << unit if unit is not None else center,
            shape_random=shape_random,
        ) for center in centers for unit in units for shape_random in shapes_random
    ]


@pytest.mark.parametrize('array', _scalar_poisson_random_samples())
class TestScalarPoissonRandomSample(
    AbstractTestAbstractScalarRandomSample,
    tests.test_core.AbstractTestAbstractPoissonRandomSample,
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

def _scalar_stratified_random_spaces():
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
        na.ScalarStratifiedRandomSpace(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            axis='y',
            num=_num_y,
            endpoint=endpoint,
            seed=None
        ) for start in starts for stop in stops for unit in units for endpoint in endpoints
    ]


@pytest.mark.parametrize('array', _scalar_stratified_random_spaces())
class TestStratifiedRandomSpace(
    AbstractTestAbstractScalarSpace,
    tests.test_core.AbstractTestAbstractStratifiedRandomSpace,
):
    pass

def _scalar_logarithmic_spaces():
    start_exponents = [
        0,
        na.ScalarArray(np.random.random(_num_x), axes=('x', )),
    ]
    stop_exponents = [
        2,
        na.ScalarArray(np.random.random(_num_x) + 2, axes=('x', )),
    ]
    bases = [
        2,
        na.ScalarArray(np.random.random(_num_x) + 1, axes=('x',)),
    ]
    endpoints = [
        False,
        True,
    ]
    return [
        na.ScalarLogarithmicSpace(
            start_exponent=start,
            stop_exponent=stop,
            base=base,
            axis='y',
            num=_num_y,
            endpoint=endpoint,
        ) for start in start_exponents for stop in stop_exponents for base in bases for endpoint in endpoints
    ]


@pytest.mark.parametrize('array', _scalar_logarithmic_spaces())
class TestScalarLogarithmicSpace(
    AbstractTestAbstractScalarSpace,
    tests.test_core.AbstractTestAbstractLogarithmicSpace,
):
    pass

def _scalar_geometric_spaces():
    starts = [
        1,
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
        na.ScalarGeometricSpace(
            start=start << unit if unit is not None else start,
            stop=stop << unit if unit is not None else stop,
            axis='y',
            num=_num_y,
            endpoint=endpoint
        ) for start in starts for stop in stops for unit in units for endpoint in endpoints
    ]


@pytest.mark.parametrize('array', _scalar_geometric_spaces())
class TestScalarGeometricSpace(
    AbstractTestAbstractScalarSpace,
    tests.test_core.AbstractTestAbstractGeometricSpace,
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

