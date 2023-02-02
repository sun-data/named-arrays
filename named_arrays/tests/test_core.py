from __future__ import annotations
from typing import Sequence, Type, Callable
import pytest
import abc
import dataclasses
import numpy as np
import astropy.units as u
import astropy.units.quantity_helper.helpers as quantity_helpers
import named_arrays as na

num_x = 11
num_y = 12
num_z = 13


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
    abc.ABC,
):

    @pytest.mark.parametrize('axis', [None, 'x', ('x', 'y')])
    def test_axis_normalized_function(
            self,
            array: na.AbstractArray,
            axis: None | str |Sequence[str],
    ):
        axis_normalized = na.axis_normalized(array, axis=axis)
        assert isinstance(axis_normalized, tuple)
        for ax in axis_normalized:
            assert isinstance(ax, str)

    def test_axes(self, array: na.AbstractArray):
        axes = array.axes
        assert isinstance(axes, tuple)
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

    @abc.abstractmethod
    def test_dtype(self, array: na.AbstractArray):
        assert array.dtype == array.array.dtype

    @abc.abstractmethod
    def test_unit(self, array: na.AbstractArray):
        assert array.unit == array.array.unit

    @abc.abstractmethod
    def test_unit_normalized(self, array: na.AbstractArray):
        assert array.unit_normalized == array.array.unit_normalized

    def test_array(self, array: na.AbstractArray):
        assert isinstance(array.array, na.AbstractExplicitArray)

    def test_type_array(self, array: na.AbstractArray):
        assert issubclass(array.type_array, na.AbstractExplicitArray)

    def test_type_array_abstract(self, array: na.AbstractArray):
        assert issubclass(array.type_array_abstract, na.AbstractArray)
        assert not issubclass(array.type_array_abstract, na.AbstractExplicitArray)
        assert not issubclass(array.type_array_abstract, na.AbstractImplicitArray)

    def test_centers(self, array: na.AbstractArray):
        assert isinstance(array.centers, na.AbstractArray)

    @pytest.mark.parametrize('dtype', [int, float])
    def test_astype(self, array: na.AbstractArray, dtype: type):
        array_new = array.astype(dtype)
        assert array_new.dtype == dtype

    @pytest.mark.parametrize('unit', [u.m, u.s])
    def test_to(self, array: na.AbstractArray, unit: None | u.UnitBase):
        if isinstance(array.unit, u.UnitBase) and array.unit.is_equivalent(unit):
            array_new = array.to(unit)
            assert array_new.unit == unit
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test_length(self, array: na.AbstractArray):
        dtype = array.dtype

        if not isinstance(dtype, dict):
            dtype = dict(x=dtype)
        if not all(np.issubdtype(dtype[c], np.number) for c in dtype):
            with pytest.raises(ValueError):
                array.length
            return

        unit = array.unit
        if isinstance(unit, dict):
            iter_unit = iter(unit)
            unit_0 = na.unit_normalized(unit[next(iter_unit)])
            if not all(unit_0.is_equivalent(na.unit_normalized(unit[c])) for c in iter_unit):
                with pytest.raises(u.UnitConversionError):
                    array.length
                return

        assert isinstance(array.length, (int, float, complex, np.ndarray, na.AbstractScalar))
        assert np.all(array.length >= 0)

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

    def test_copy_shallow(self, array: na.AbstractArray):
        array_copy = array.copy_shallow()
        assert isinstance(array_copy, na.AbstractArray)
        assert dataclasses.is_dataclass(array_copy)
        for field in dataclasses.fields(array_copy):
            assert getattr(array, field.name) is getattr(array_copy, field.name)

    def test_copy(self, array: na.AbstractArray):
        array_copy = array.copy()
        assert isinstance(array_copy, na.AbstractArray)
        assert dataclasses.is_dataclass(array_copy)
        for field in dataclasses.fields(array_copy):
            assert np.all(getattr(array, field.name) == getattr(array_copy, field.name))

    @abc.abstractmethod
    def test__getitem__(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        pass

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
    class TestUfuncUnary(abc.ABC):

        @abc.abstractmethod
        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractArray,
                out: bool,
        ):
            pass

    @pytest.mark.parametrize(
        argnames='ufunc',
        argvalues=[
            np.add,
            np.subtract,
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

        @abc.abstractmethod
        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):
            pass

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

    @pytest.mark.parametrize('out', [False, True])
    class TestMatmul(abc.ABC):
        @abc.abstractmethod
        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):
            pass

        def test_matmul_reversed(
                self,
                array: na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):
            array = np.transpose(array)
            if array_2 is not None:
                array_2 = np.transpose(array_2)
            self.test_matmul(array_2, array, out=out)

    class TestArrayFunctions(abc.ABC):

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
        @pytest.mark.parametrize('axis', [None, 'y', 'x', ('x', 'y')])
        @pytest.mark.parametrize('out', [False, True])
        @pytest.mark.parametrize('keepdims', [False, True])
        @pytest.mark.parametrize('where', [False, True])
        class TestReductionFunctions(abc.ABC):

            @abc.abstractmethod
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
                pass

        @pytest.mark.parametrize(
            argnames='func',
            argvalues=[
                np.percentile,
                np.nanpercentile,
                np.quantile,
                np.nanquantile,
            ]
        )
        @pytest.mark.parametrize('axis', [None, 'y', 'x', ('x', 'y')])
        @pytest.mark.parametrize('out', [False, True])
        @pytest.mark.parametrize('keepdims', [False, True])
        class TestPercentileLikeFunctions(abc.ABC):

            @abc.abstractmethod
            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    out: bool,
                    keepdims: bool,
            ):
                pass

        @pytest.mark.parametrize(
            argnames=('argfunc', 'func'),
            argvalues=[
                (np.argmin, np.min),
                (np.nanargmin, np.nanmin),
                (np.argmax, np.max),
                (np.nanargmax, np.nanmax),
            ]
        )
        @pytest.mark.parametrize('axis', [None, 'y'])
        @pytest.mark.parametrize('out', [False, True])
        @pytest.mark.parametrize('keepdims', [False, True])
        class TestArgReductionFunctions:
            def test_arg_reduction_functions(
                    self,
                    argfunc: Callable,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: None | str,
                    out: bool,
                    keepdims: bool,
            ):
                kwargs = dict()

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

                if out:
                    if axis is not None:
                        kwargs['out'] = array.indices
                        kwargs['out'][axis] = array.type_array.empty(shape_result, dtype=int)
                    else:
                        kwargs['out'] = {ax: array.type_array.empty(shape_result, dtype=int) for ax in array.axes}

                if axis is not None:
                    if axis not in array.axes:
                        with pytest.raises(ValueError, match='Reduction axis .* not in array with axes .*'):
                            argfunc(array, axis=axis, **kwargs)
                        return
                else:
                    if not array.shape:
                        with pytest.raises(
                                expected_exception=ValueError,
                                match=r"Applying .* to zero-dimensional arrays is not supported"
                        ):
                            argfunc(array, axis=axis, **kwargs)
                        return

                if out:
                    with pytest.raises(NotImplementedError, match=r"out keyword argument is not implemented for .*"):
                        argfunc(array, axis=axis, **kwargs)
                    return

                result = argfunc(array, axis=axis, **kwargs)

                array_reduced = array[result]
                array_reduced_expected = func(array, axis=axis, keepdims=keepdims)

                assert np.all(array_reduced == array_reduced_expected)

        @pytest.mark.parametrize(
            argnames='func',
            argvalues=[
                np.fft.fft,
                np.fft.ifft,
                np.fft.rfft,
                np.fft.irfft,
            ]
        )
        @pytest.mark.parametrize('axis', [('x', 'kx'), ('y', 'ky')])
        class TestFFTLikeFunctions(abc.ABC):

            @abc.abstractmethod
            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: tuple[str, str],
            ):
                pass

        @pytest.mark.parametrize(
            argnames='func',
            argvalues=[
                np.fft.fft2,
                np.fft.ifft2,
                np.fft.rfft2,
                np.fft.irfft2,
                np.fft.fftn,
                np.fft.ifftn,
                np.fft.rfftn,
                np.fft.irfftn,
            ]
        )
        @pytest.mark.parametrize('s', [None, dict(y=num_y), dict(x=num_x), dict(x=num_x, y=num_y)])
        class TestFFTNLikeFunctions(abc.ABC):

            @abc.abstractmethod
            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    s: None | dict[str, int],
            ):
                pass

        @pytest.mark.parametrize(
            argnames='shape',
            argvalues=[
                dict(x=num_x, y=num_y),
                dict(x=num_x, y=num_y, z=num_z),
            ]
        )
        def test_broadcast_to(self, array: na.AbstractArray, shape: dict[str, int]):
            result = np.broadcast_to(array, shape=shape)
            assert result.shape == shape

        def test_shape(self, array: na.AbstractArray):
            assert np.shape(array) == array.shape

        @pytest.mark.parametrize(
            argnames='axes',
            argvalues=[
                None,
                ['x', 'y'],
                ['y', 'x'],
            ],
        )
        def test_transpose(self, array: na.AbstractArray, axes: None | Sequence[str]):
            axes_normalized = tuple(reversed(array.axes) if axes is None else axes)
            result = np.transpose(
                a=array,
                axes=axes
            )
            assert result.axes == axes_normalized
            assert {ax: result.shape[ax] for ax in result.shape if ax in array.axes} == array.shape

        @pytest.mark.parametrize(
            argnames='source,destination',
            argvalues=[
                ['y', 'y2'],
                [('x', 'y'), ('x2', 'y2')],
            ]
        )
        def test_moveaxis(
                self,
                array: na.AbstractArray,
                source: str | Sequence[str],
                destination: str | Sequence[str],
        ):
            source_normalized = (source, ) if isinstance(source, str) else source
            destination_normalized = (destination, ) if isinstance(destination, str) else destination

            if any(ax not in array.axes for ax in source_normalized):
                with pytest.raises(ValueError, match=r"source axes .* not in array axes .*"):
                    np.moveaxis(a=array, source=source, destination=destination)
                return

            result = np.moveaxis(a=array, source=source, destination=destination)

            assert np.all(array.sum() == result.sum())
            assert len(array.axes) == len(result.axes)
            assert not any(ax in result.axes for ax in source_normalized)
            assert all(ax in result.axes for ax in destination_normalized)

        @pytest.mark.parametrize('newshape', [dict(r=-1)])
        def test_reshape(self, array: na.AbstractArray, newshape: dict[str, int]):

            result = np.reshape(a=array, newshape=newshape)

            assert result.size == array.size
            assert result.axes == tuple(newshape.keys())

        def test_linalg_inv(self, array: na.AbstractArray):
            with pytest.raises(NotImplementedError):
                np.linalg.inv(array)

        @pytest.mark.parametrize('axis', ['y', 'z'])
        @pytest.mark.parametrize('use_out', [False, True])
        def test_stack(
                self,
                array: na.AbstractArray,
                axis: str,
                use_out: bool,
        ):
            arrays = [array, array]

            if axis in array.axes:
                with pytest.raises(ValueError, match=r"axis .* already in array"):
                    np.stack(arrays, axis=axis)
                return

            if use_out:
                out = array.type_array.empty({axis: len(arrays)} | array.shape, dtype=array.dtype)
                if array.unit is not None:
                    out = out << array.unit
            else:
                out = None

            result = np.stack(arrays=arrays, axis=axis, out=out)

            assert np.all(result[{axis: 0}] == array)
            assert np.all(result[{axis: 1}] == array)

        @pytest.mark.parametrize('axis', ['x', 'y'])
        @pytest.mark.parametrize('use_out', [False, True])
        def test_concatenate(
                self,
                array: na.AbstractArray,
                axis: str,
                use_out: bool,
        ):
            arrays = [array, array]

            shape_out = array.shape
            if axis not in shape_out:
                shape_out[axis] = 1
            shape_out[axis] = 2 * shape_out[axis]

            if use_out:
                out = array.type_array.empty(shape_out, dtype=array.dtype)
                if array.unit is not None:
                    out = out << array.unit
            else:
                out = None

            result = np.concatenate(arrays, axis=axis, out=out)

            assert result.shape == shape_out
            assert np.all(result[{axis: slice(None, shape_out[axis] // 2)}] == array)
            assert np.all(result[{axis: slice(shape_out[axis] // 2, None)}] == array)

        @abc.abstractmethod
        def test_sort(self, array: na.AbstractArray, axis: None | str):
            pass

        @pytest.mark.parametrize('axis', [None, 'x', 'y'])
        def test_argsort(self, array: na.AbstractArray, axis: None | str):

            if axis is not None:
                if axis not in array.axes:
                    with pytest.raises(ValueError, match="axis .* not in input array with axes .*"):
                        np.argsort(a=array, axis=axis)
                    return
            else:
                if not array.shape:
                    with pytest.raises(ValueError, match="sorting zero-dimensional arrays is not supported"):
                        np.argsort(a=array, axis=axis)
                    return

            result = np.argsort(a=array, axis=axis)

            assert isinstance(result, dict)

            array_broadcasted = array.broadcast_to(array.shape)
            if axis is not None:
                sorted = array_broadcasted[result]
            else:
                sorted = array_broadcasted.reshape({array.axes_flattened: -1})[result]

            sorted_expected = np.sort(array_broadcasted, axis=axis)

            assert np.all(sorted == sorted_expected)

        def test_unravel_index(self, array: na.AbstractArray):
            indices_raveled = na.ScalarArrayRange(0, array.size, axis=array.axes_flattened).reshape(array.shape)
            indices_raveled = indices_raveled * array.type_array.ones(shape=dict(), dtype=int)
            result = np.unravel_index(
                indices=indices_raveled,
                shape=array.shape,
            )
            expected = array.indices
            for ax in result:
                assert np.all(result[ax] == expected[ax])

        @pytest.mark.parametrize('array_2', ['copy', 'ones'])
        def test_array_equal(self, array: na.AbstractArray, array_2: str):
            if array_2 == 'copy':
                array_2 = array.copy()
                assert np.array_equal(array, array_2)
                return

            elif array_2 == 'ones':
                array_2 = array.type_array.ones(array.shape)

            assert not np.array_equal(array, array_2)

        @abc.abstractmethod
        def test_nonzero(self, array: na.AbstractArray):
            pass

        @abc.abstractmethod
        def test_nan_to_num(self, array: na.AbstractArray, copy: bool):
            pass

    @pytest.mark.parametrize(
        argnames='shape',
        argvalues=[
            dict(x=num_x, y=num_y),
            dict(x=num_x, y=num_y, z=13),
        ]
    )
    def test_broadcast_to(
            self,
            array: na.AbstractArray,
            shape: dict[str, int],
    ):
        assert np.all(array.broadcast_to(shape) == np.broadcast_to(array, shape))

    @pytest.mark.parametrize('shape', [dict(r=-1)])
    def test_reshape(
            self,
            array: na.AbstractArray,
            shape: dict[str, int],
    ):
        assert np.all(array.reshape(shape) == np.reshape(array, shape))

    def test_min(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.min() == np.min(array))

    def test_max(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.max() == np.max(array))

    def test_sum(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.sum() == np.sum(array))

    def test_ptp(
            self,
            array: na.AbstractArray,
    ):
        if not isinstance(array.dtype, dict):
            if np.issubdtype(array.dtype, bool):
                with pytest.raises(TypeError, match='numpy boolean subtract, .*'):
                    array.ptp()
                return

        assert np.all(array.ptp() == np.ptp(array))

    def test_mean(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.mean() == np.mean(array))

    def test_std(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.std() == np.std(array))

    def test_percentile(
            self,
            array: na.AbstractArray,
    ):
        q = 25 * u.percent
        kwargs = dict(method='closest_observation')
        assert np.all(array.percentile(q, **kwargs) == np.percentile(array, q, **kwargs))

    def test_all(
            self,
            array: na.AbstractArray,
    ):
        if getattr(array, 'unit', None) is None:
            assert np.all(array.all() == np.all(array))
        else:
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.all()

    def test_any(
            self,
            array: na.AbstractArray
    ):
        if getattr(array, 'unit', None) is None:
            assert np.all(array.any() == np.any(array))
        else:
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.any()

    def test_rms(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.rms() == np.sqrt(np.mean(np.square(array))))

    def test_transpose(
            self,
            array: na.AbstractArray,
    ):
        assert np.all(array.transpose() == np.transpose(array))



class AbstractTestAbstractExplicitArray(
    AbstractTestAbstractArray,
):
    pass


@pytest.mark.parametrize('shape', [dict(x=3), dict(x=4, y=5)])
@pytest.mark.parametrize('dtype', [int, float, complex])
class AbstractTestAbstractExplicitArrayCreation(abc.ABC):

    @property
    @abc.abstractmethod
    def type_array(self) -> Type[na.AbstractExplicitArray]:
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


class AbstractTestAbstractImplicitArray(
    AbstractTestAbstractArray,
):
    pass


class AbstractTestAbstractRandomMixin(
    abc.ABC,
):

    def test_seed(self, array: na.AbstractRandomMixin):
        assert isinstance(array.seed, int)


class AbstractTestRandomMixin(
    AbstractTestAbstractRandomMixin,
):
    pass


class AbstractTestAbstractRangeMixin:

    def test_start(self, array: na.AbstractRangeMixin):
        assert isinstance(array.start, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_stop(self, array: na.AbstractRangeMixin):
        assert isinstance(array.stop, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_range(self, array: na.AbstractRangeMixin):
        assert np.all(np.abs(array.range) > 0)


class AbstractTestAbstractSymmetricRangeMixin(
    AbstractTestAbstractRangeMixin,
):
    def test_center(self, array: na.AbstractSymmetricRangeMixin):
        assert isinstance(array.center, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_width(self, array: na.AbstractSymmetricRangeMixin):
        assert isinstance(array.width, (int, float, complex, u.Quantity, na.AbstractArray))
        assert np.all(array.width > 0)


class AbstractTestAbstractRandomSample(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractImplicitArray,
):

    def test_shape_random(self, array: na.AbstractRandomSample):
        shape_random = array.shape_random
        if shape_random is not None:
            assert all(isinstance(k, str) for k in shape_random)
            assert all(isinstance(shape_random[k], int) for k in shape_random)
            assert all(shape_random[k] > 0 for k in shape_random)


class AbstractTestAbstractUniformRandomSample(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractRangeMixin,
    AbstractTestAbstractImplicitArray,
):
    pass


class AbstractTestAbstractNormalRandomSample(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractSymmetricRangeMixin,
    AbstractTestAbstractImplicitArray,
):
    pass


class AbstractTestAbstractParameterizedArray(
    AbstractTestAbstractImplicitArray,
):

    def test_axis(self, array: na.AbstractParameterizedArray):
        assert isinstance(array.axis, (str, na.AbstractArray))

    def test_num(self, array: na.AbstractParameterizedArray):
        assert isinstance(array.num, (int, na.AbstractArray))
        assert array.num == array.shape[array.axis]


class AbstractTestAbstractLinearParametrizedArrayMixin:

    def test_step(self, array: na.AbstractLinearParameterizedArrayMixin):
        assert isinstance(array.step, (int, float, complex, u.Quantity, na.AbstractArray))
        assert np.all(np.abs(array.step) > 0)


class AbstractTestAbstractArrayRange(
    AbstractTestAbstractLinearParametrizedArrayMixin,
    AbstractTestAbstractRangeMixin,
    AbstractTestAbstractParameterizedArray,
):
    pass


class AbstractTestAbstractSpace(
    AbstractTestAbstractParameterizedArray,
):

    def test_endpoint(self, array: na.AbstractSpace):
        assert isinstance(array.endpoint, bool)


class AbstractTestAbstractLinearSpace(
    AbstractTestAbstractLinearParametrizedArrayMixin,
    AbstractTestAbstractRangeMixin,
    AbstractTestAbstractSpace,
):
    pass


class AbstractTestAbstractStratifiedRandomSpace(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractLinearSpace,
):
    pass


class AbstractTestAbstractLogarithmicSpace(
    AbstractTestAbstractRangeMixin,
    AbstractTestAbstractSpace,
):

    def test_start_exponent(self, array: na.AbstractLogarithmicSpace):
        assert isinstance(array.start_exponent, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_stop_exponent(self, array: na.AbstractLogarithmicSpace):
        assert isinstance(array.stop_exponent, (int, float, complex, u.Quantity, na.AbstractArray))

    def test_base(self, array: na.AbstractLogarithmicSpace):
        assert isinstance(array.base, (int, float, complex, u.Quantity, na.AbstractArray))


class AbstractTestAbstractGeometricSpace(
    AbstractTestAbstractRangeMixin,
    AbstractTestAbstractSpace,
):
    pass
