from __future__ import annotations
from typing import Sequence, Type, Callable
import pytest
import abc
import warnings
import dataclasses
import numpy as np
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na

num_x = 3
num_y = 4
num_z = 5
num_distribution = 3


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
    def test_named_array_like(self, array: na.AbstractArray):
        assert na.named_array_like(array)

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
        if not array.shape:
            with pytest.raises(ValueError, match="`axes` must be a non-empty sequence, got .*"):
                array.axes_flattened
            return

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

    def test_value(self, array: na.AbstractArray):
        result = array.value
        assert isinstance(result, array.type_abstract)
        assert isinstance(result + 1, array.type_abstract)


    def test_array(self, array: na.AbstractArray):
        assert isinstance(array.explicit, na.AbstractExplicitArray)

    def test_type_array(self, array: na.AbstractArray):
        assert issubclass(array.type_explicit, na.AbstractExplicitArray)

    def test_type_array_abstract(self, array: na.AbstractArray):
        assert issubclass(array.type_abstract, na.AbstractArray)
        assert not issubclass(array.type_abstract, na.AbstractExplicitArray)
        assert not issubclass(array.type_abstract, na.AbstractImplicitArray)

    def test_broadcasted(self, array: na.AbstractArray):
        result = array.broadcasted
        assert result.shape == array.shape
        assert isinstance(result, array.type_explicit)

    @abc.abstractmethod
    def test_astype(self, array: na.AbstractArray, dtype: type):
        pass

    @abc.abstractmethod
    def test_to(self, array: na.AbstractArray, unit: None | u.UnitBase):
        pass

    @abc.abstractmethod
    def test_length(self, array: na.AbstractArray):
        pass

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

    @pytest.mark.parametrize('axes', [None, ('x', 'y'), ('x', 'y', 'z')])
    def test_combine_axes(
            self,
            array: na.AbstractArray,
            axes: None | Sequence[str]
    ):
        axis_new = 'new_test_axis'
        if axes is None or set(axes).issubset(array.axes):
            array_new = array.combine_axes(axes=axes, axis_new=axis_new)
            assert axis_new in array_new.axes
            axes_normlized = array.axes if axes is None else axes
            num_axis_new =  np.array(
                [array.shape[ax] for ax in axes_normlized]).prod()
            assert array_new.shape[axis_new] == num_axis_new
            for axis in axes_normlized:
                assert axis not in array_new.axes
        else:
            with pytest.raises(ValueError):
                array.combine_axes(axes=axes, axis_new=axis_new)

    @pytest.mark.parametrize(
        argnames="axis",
        argvalues=[
            None,
            "y",
            ("y", ),
            ("x", "y"),
            ("x", "y", "z"),
        ]
    )
    def test_volume_cell(
        self,
        array: na.AbstractArray,
        axis: None | str | Sequence[str],
    ):
        with pytest.raises(NotImplementedError):
            array.volume_cell(axis=axis)

    def test__repr__(self, array: na.AbstractArray):
        assert isinstance(repr(array), str)

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
            attr = getattr(array, field.name)
            attr_copy = getattr(array_copy, field.name)
            if isinstance(attr, dict):
                assert [np.all(attr[key] == attr_copy[key] for key in attr)]
            else:
                assert np.all(attr == attr_copy)

    @abc.abstractmethod
    def test__getitem__(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        pass

    @abc.abstractmethod
    def test__bool__(self, array: na.AbstractArray):
        pass

    @abc.abstractmethod
    def test__mul__(self, array: na.AbstractArray):
        pass

    @abc.abstractmethod
    def test__lshift__(self, array: na.AbstractArray):
        pass

    @abc.abstractmethod
    def test__truediv__(self, array: na.AbstractArray):
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
    class TestUfuncUnary(abc.ABC):

        @abc.abstractmethod
        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractArray,
        ):
            pass

    @pytest.mark.parametrize(
        argnames='ufunc',
        argvalues=[
            np.add,
            np.subtract,
            np.multiply,
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
            np.ldexp,
            np.fmod,
        ]
    )
    class TestUfuncBinary(
        abc.ABC,
    ):

        @abc.abstractmethod
        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            pass

        def test_ufunc_binary_reversed(
                self,
                ufunc: np.ufunc,
                array: na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            array = np.transpose(array)
            if array_2 is not None:
                array_2 = np.transpose(array_2)
            self.test_ufunc_binary(ufunc, array_2, array)

    class TestMatmul(abc.ABC):
        @abc.abstractmethod
        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            pass

        def test_matmul_reversed(
                self,
                array: na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            array = np.transpose(array)
            if array_2 is not None:
                array_2 = np.transpose(array_2)
            self.test_matmul(array_2, array)

    class TestArrayFunctions(abc.ABC):

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                na.asarray,
                na.asanyarray,
            ]
        )
        class TestAsArrayLikeFunctions(abc.ABC):

            @abc.abstractmethod
            def test_asarray_like_functions(
                    self,
                    func: Callable,
                    array: None | float | u.Quantity | na.AbstractArray,
                    array_2: None | float | u.Quantity | na.AbstractArray,
            ):
                pass

            def test_asarray_like_functions_reversed(
                    self,
                    func: Callable,
                    array: None | float | u.Quantity | na.AbstractArray,
                    array_2: None | float | u.Quantity | na.AbstractArray,
            ):
                self.test_asarray_like_functions(
                    func=func,
                    array=array_2,
                    array_2=array,
                )

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                np.real,
                np.imag,
            ]
        )
        class TestSingleArgumentFunctions(abc.ABC):

            @abc.abstractmethod
            def test_single_argument_functions(
                self,
                func: Callable,
                array: na.AbstractArray,
            ):
                pass

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                np.empty_like,
                np.zeros_like,
                np.ones_like,
            ]
        )
        @pytest.mark.parametrize(
            argnames="shape",
            argvalues=[
                None,
                dict(y=num_y),
                dict(x=num_x, y=num_y)
            ]
        )
        @pytest.mark.parametrize("dtype", [None, int, float])
        class TestArrayCreationLikeFunctions(abc.ABC):

            def test_array_creation_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    shape: dict[str, int],
                    dtype: type
            ):
                result = func(array, dtype=dtype, shape=shape)

                if shape is None:
                    shape_normalized = array.shape
                else:
                    shape_normalized = shape

                assert result.shape == shape_normalized
                assert type(result) == array.type_explicit

                if func is np.zeros_like:
                    assert np.all(result.value == 0)
                elif func is np.ones_like:
                    assert np.all(result.value == 1)

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
        @pytest.mark.parametrize('dtype', [np._NoValue, float])
        @pytest.mark.parametrize('axis', [None, 'y', 'x', ('x', 'y')])
        @pytest.mark.parametrize('keepdims', [False, True])
        class TestReductionFunctions(abc.ABC):

            @abc.abstractmethod
            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: None | str | Sequence[str],
                    dtype: None | type | np.dtype,
                    keepdims: bool,
                    where: bool | na.AbstractArray,
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
        @pytest.mark.parametrize('keepdims', [False, True])
        class TestPercentileLikeFunctions(abc.ABC):

            @abc.abstractmethod
            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
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
        @pytest.mark.parametrize('axis', [None, 'y', ('x', 'y')])
        class TestArgReductionFunctions:
            def test_arg_reduction_functions(
                    self,
                    argfunc: Callable,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: None | str,
            ):
                kwargs = dict()

                if axis is not None:
                    if not set(axis).issubset(array.axes):
                        with pytest.raises(ValueError, match='Reduction axes .* are not a subset of the array axes .*'):
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

                result = argfunc(array, axis=axis, **kwargs)

                array_reduced = array[result]
                array_reduced_expected = func(array, axis=axis)

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
        @pytest.mark.parametrize('axes', [dict(y='ky'), dict(x='kx', y='ky')])
        @pytest.mark.parametrize('s', [None, dict(y=5), dict(x=4, y=5)])
        class TestFFTNLikeFunctions(abc.ABC):

            @abc.abstractmethod
            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axes: dict[str, str],
                    s: None | dict[str, int],
            ):
                pass

        @pytest.mark.filterwarnings("ignore: function .* is not known to astropy's Quantity.")
        @pytest.mark.filterwarnings("ignore: divide by zero encountered")
        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                np.emath.sqrt,
                np.emath.log,
                np.emath.log2,
                np.emath.log10,
                np.emath.arccos,
                np.emath.arcsin,
                np.emath.arctanh,
            ]
        )
        class TestEmathFunctions(abc.ABC):

            @abc.abstractmethod
            def test_emath_functions(
                self,
                func: Callable,
                array: na.AbstractArray,
            ):
                pass

        def test_copyto(self, array: na.AbstractArray):
            dst = 0 * array
            np.copyto(dst=dst, src=array)
            assert np.all(array == dst)


        @pytest.mark.parametrize(
            argnames='shape',
            argvalues=[
                dict(x=num_x, y=num_y),
                dict(x=num_x, y=num_y, z=num_z),
            ]
        )
        def test_broadcast_to(self, array: na.AbstractArray, shape: dict[str, int]):
            if not set(array.shape).issubset(shape):
                with pytest.raises(ValueError):
                    np.broadcast_to(array, shape=shape)
                return
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

            if not set(array.axes).issubset(axes_normalized):
                with pytest.raises(ValueError):
                    np.transpose(array, axes=axes)
                return

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

            assert np.all(array == np.moveaxis(a=result, source=destination, destination=source))
            assert len(array.axes) == len(result.axes)
            assert not any(ax in result.axes for ax in source_normalized)
            assert all(ax in result.axes for ax in destination_normalized)

        @pytest.mark.parametrize('newshape', [dict(r=-1)])
        def test_reshape(self, array: na.AbstractArray, newshape: dict[str, int]):

            result = np.reshape(a=array, newshape=newshape)

            assert result.size == array.size
            assert result.axes == tuple(newshape.keys())

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
                out = 0 * np.stack(arrays=arrays, axis=axis)
            else:
                out = None

            result = np.stack(arrays=arrays, axis=axis, out=out)

            assert np.all(result[{axis: 0}] == array)
            assert np.all(result[{axis: 1}] == array)

            if use_out:
                assert result is out

        @pytest.mark.parametrize('axis', ['x', 'y'])
        def test_concatenate(
                self,
                array: na.AbstractArray,
                axis: str,
        ):
            arrays = [array, array]

            if axis not in array.shape:
                with pytest.raises(ValueError, match="axis .* must be present in all the input arrays, got .*"):
                    np.concatenate(arrays, axis=axis)
                return

            result = np.concatenate(arrays, axis=axis)

            out = 0 * result

            result_out = np.concatenate(arrays, axis=axis, out=out)

            assert np.all(result[{axis: slice(None, array.shape[axis])}] == array)
            assert np.all(result[{axis: slice(array.shape[axis], None)}] == array)
            assert np.all(result == result_out)
            assert result_out is out

        @abc.abstractmethod
        def test_sort(self, array: na.AbstractArray, axis: None | str | Sequence[str]):
            pass

        @pytest.mark.parametrize('axis', [None, 'x', 'y', ('x', 'y'), ()])
        def test_argsort(self, array: na.AbstractArray, axis: None | str | Sequence[str]):

            axis_normalized = na.axis_normalized(array, axis)

            if axis is not None:
                if not axis:
                    with pytest.raises(ValueError, match="if `axis` is a sequence, it must not be empty, got .*"):
                        np.argsort(a=array, axis=axis)
                    return

                if not set(axis_normalized).issubset(array.axes):
                    with pytest.raises(ValueError, match="`axis`, .* is not a subset of `a.axes`, .*"):
                        np.argsort(a=array, axis=axis)
                    return

            result = np.argsort(a=array, axis=axis)
            assert isinstance(result, dict)

            sorted = array[result]
            sorted_expected = np.sort(a=array, axis=axis)
            assert np.all(sorted == sorted_expected)

        def test_unravel_index(self, array: na.AbstractArray):
            indices_raveled = na.ScalarArrayRange(0, array.size, axis='raveled').reshape(array.shape)
            indices_raveled = indices_raveled * np.ones_like(array.value, shape=dict(), dtype=int)
            result = np.unravel_index(
                indices=indices_raveled,
                shape=array.shape,
            )
            expected = array.indices
            for ax in result:
                assert np.all(result[ax] == expected[ax])

        @pytest.mark.parametrize('array_2', ['copy', 'zeros'])
        def test_array_equal(self, array: na.AbstractArray, array_2: str):
            if array_2 == "copy":
                array_2 = array.copy()
                assert np.array_equal(array, array_2)

            elif array_2 == "zeros":
                array_2 = 0 * array
                assert not np.array_equal(array, array_2)

        @pytest.mark.parametrize("array_2", ["copy", "broadcast", "zeros"])
        def test_array_equiv(self, array: na.AbstractArray, array_2: str):
            if array_2 == "copy":
                array_2 = array.copy()
                assert np.array_equiv(array, array_2)

            elif array_2 == "broadcast":
                shape_new = array.shape | dict(extra_axis=5)
                array_2 = array.copy().broadcast_to(shape_new)
                assert np.array_equiv(array, array_2)

            elif array_2 == "zeros":
                array_2 = 0 * array
                assert not np.array_equiv(array, array_2)

        @pytest.mark.parametrize("array_2", ["copy", "zeros"])
        def test_allclose(self, array: na.AbstractArray, array_2: str):
            if array_2 == "copy":
                array_2 = array + array.mean() * na.ScalarUniformRandomSample(-1e-10, 1e-10)
                assert np.allclose(array, array_2)

            elif array_2 == "zeros":
                array_2 = 0 * array
                assert not np.allclose(array, array_2)

            else:
                raise NotImplementedError

        def test_nonzero(self, array: na.AbstractArray):
            mask = array > array.mean()
            result = array[np.nonzero(mask)]
            result_expected = array[mask]
            assert np.all(result == result_expected)

        def test_where(self, array: na.AbstractArray):
            condition = array > array.mean()
            result = np.where(
                condition,
                array,
                array.mean(),
            )
            result_expected = np.maximum(array.mean(), array)
            assert np.all(result == result_expected)

        @abc.abstractmethod
        def test_nan_to_num(self, array: na.AbstractArray, copy: bool):
            pass

        @abc.abstractmethod
        def test_convolve(self, array: na.AbstractArray, v: na.AbstractArray, mode: str):
            pass

        @pytest.mark.parametrize(
            argnames="repeats",
            argvalues=[
                2,
                na.random.poisson(2, shape_random=dict(y=num_y))
            ]
        )
        @pytest.mark.parametrize(
            argnames="axis",
            argvalues=[
                "y",
            ]
        )
        def test_repeat(
            self,
            array: na.AbstractArray,
            repeats: int | na.AbstractScalarArray,
            axis: str,
        ):
            if not array.shape:
                with pytest.raises(ValueError):
                    np.repeat(
                        a=array,
                        repeats=repeats,
                        axis=axis,
                    )
                return

            result = np.repeat(
                a=array,
                repeats=repeats,
                axis=axis,
            )

            repeats_ = na.broadcast_to(repeats, shape={axis: array.shape[axis]})
            assert result.type_abstract == array.type_abstract
            for ax in result.shape:
                if ax == axis:
                    assert result.shape[ax] == repeats_.sum()
                else:
                    assert result.shape[ax] == array.shape[ax]

        @pytest.mark.parametrize("axis", ["y"])
        @pytest.mark.parametrize("prepend", [None, 0])
        @pytest.mark.parametrize("append", [None, 0])
        def test_diff_1st_order(
            self,
            array: na.AbstractArray,
            axis: str,
            prepend: None | float | na.AbstractArray,
            append: None | float | na.AbstractArray,
        ):
            unit = na.unit_normalized(array)
            if prepend is not None:
                prepend = prepend * unit
            if append is not None:
                append = append * unit

            kwargs = dict(
                a=array,
                axis=axis,
                prepend=prepend,
                append=append,
            )

            if axis not in array.shape:
                with pytest.raises(ValueError):
                    np.diff(**kwargs)
                return

            result = np.diff(**kwargs)

            array_ = [array]
            if prepend is not None:
                prepend_ = na.as_named_array(prepend).add_axes(axis)
                array_ = [prepend_] + array_
            if append is not None:
                append_ = na.as_named_array(append).add_axes(axis)
                array_ = array_ + [append_]
            array_ = np.concatenate(array_, axis=axis)

            array_left = array_[{axis: slice(1, None)}]
            array_right = array_[{axis: slice(None, ~0)}]
            result_expected = array_left.astype(float) - array_right.astype(float)

            assert np.all(np.abs(result.astype(float)) == np.abs(result_expected))

    @pytest.mark.parametrize(
        argnames='shape',
        argvalues=[
            dict(x=num_x, y=num_y),
            dict(x=num_x, y=num_y, z=num_z),
        ]
    )
    def test_broadcast_to(
            self,
            array: na.AbstractArray,
            shape: dict[str, int],
    ):
        if not set(array.shape).issubset(shape):
            with pytest.raises(ValueError):
                array.broadcast_to(shape)
            return
        assert np.array_equal(array.broadcast_to(shape), np.broadcast_to(array, shape))

    @pytest.mark.parametrize('shape', [dict(r=-1)])
    def test_reshape(
            self,
            array: na.AbstractArray,
            shape: dict[str, int],
    ):
        assert np.array_equal(array.reshape(shape), np.reshape(array, shape))

    def test_min(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.min(), np.min(array))

    def test_max(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.max(), np.max(array))

    def test_sum(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.sum(), np.sum(array))

    def test_ptp(
            self,
            array: na.AbstractArray,
    ):
        try:
            result_expected = np.ptp(array)
        except Exception as e:
            with pytest.raises(type(e)):
                array.ptp()
            return

        result = array.ptp()

        assert np.all(result == result_expected)

    def test_mean(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.mean(), np.mean(array))

    def test_std(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.std(), np.std(array))

    def test_percentile(
            self,
            array: na.AbstractArray,
    ):
        q = 25 * u.percent
        kwargs = dict(method='closest_observation')
        assert np.array_equal(array.percentile(q, **kwargs), np.percentile(array, q, **kwargs))

    def test_all(
            self,
            array: na.AbstractArray,
    ):
        try:
            result_expected = np.all(array)
        except Exception as e:
            with pytest.raises(type(e)):
                array.all()
            return

        result = array.all()

        assert np.all(result == result_expected)

    def test_any(
            self,
            array: na.AbstractArray
    ):
        try:
            result_expected = np.any(array)
        except Exception as e:
            with pytest.raises(type(e)):
                array.any()
            return

        result = array.any()

        assert np.all(result == result_expected)

    def test_rms(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.rms(), np.sqrt(np.mean(np.square(array))))

    def test_transpose(
            self,
            array: na.AbstractArray,
    ):
        assert np.array_equal(array.transpose(), np.transpose(array))

    def test_interp_linear_identity(
            self,
            array: na.AbstractArray,
    ):
        item = array.indices
        result = array.interp_linear(item)
        assert np.allclose(result, array)

    class TestNamedArrayFunctions(abc.ABC):

        def test_unit(self, array: na.AbstractArray):
            result = na.unit(array)
            if result is not None:
                assert isinstance(result, (u.UnitBase, na.AbstractArray))

        def test_unit_normalized(self, array: na.AbstractArray):
            result = na.unit_normalized(array)
            assert isinstance(result, (u.UnitBase, na.AbstractArray))

        def test_strata(self, array: na.AbstractArray):
            result = na.strata(array)
            assert result.type_abstract == array.type_abstract

        @pytest.mark.parametrize(
            argnames="slope",
            argvalues=[
                2,
                na.Cartesian2dVectorArray(x=2, y=3)
            ],
        )
        class TestInterp:
            def test_interp(
                self,
                array: na.AbstractArray,
                slope: float | na.AbstractArray,
            ):

                xp = na.linspace(-100, 100, axis="interp", num=11)

                unit_array = na.unit(array)
                if unit_array is not None:
                    xp = xp * unit_array

                fp = slope * xp

                result = na.interp(
                    x=array,
                    xp=xp,
                    fp=fp,
                    axis="interp",
                )

                assert np.allclose(result, slope * array)

        @pytest.mark.parametrize(
            argnames="func,axis,transformation",
            argvalues=[
                (na.plt.plot, np._NoValue, np._NoValue),
                (na.plt.fill, "y", na.transformations.Translation(0))
            ]
        )
        @pytest.mark.parametrize(
            argnames="ax",
            argvalues=[
                np._NoValue,
                plt.subplots()[1],
                na.plt.subplots(axis_cols="x", ncols=num_x)[1],
            ]
        )
        class TestPltPlotLikeFunctions(abc.ABC):

            def test_plt_plot_like(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    array_2: na.ArrayLike,
                    ax: None | matplotlib.axes.Axes,
                    axis: None | str,
                    where: bool | na.AbstractScalar,
                    transformation: None | na.transformations.AbstractTransformation,
                    alpha: None | str | na.AbstractScalar,
            ):
                args = (array_2, array)
                args = tuple(a for a in args if a is not None)

                kwargs = dict()
                if ax is not np._NoValue:
                    kwargs["ax"] = ax
                if axis is not np._NoValue:
                    kwargs["axis"] = axis
                if where is not np._NoValue:
                    kwargs["where"] = where
                if transformation is not np._NoValue:
                    kwargs["transformation"] = transformation
                if alpha is not np._NoValue:
                    kwargs["alpha"] = alpha

                shape = na.shape_broadcasted(*args)

                axis_normalized = axis
                if axis_normalized is np._NoValue:
                    axis_normalized = None

                if axis_normalized is None:
                    if len(shape) != 1:
                        with pytest.raises(
                            expected_exception=ValueError,
                            match="if `axis` is `None`, the broadcasted shape of .* should have one element"
                        ):
                            func(*args, **kwargs)
                        return
                    axis_normalized = next(iter(shape))

                shape_orthogonal = {a: shape[a] for a in shape if a != axis_normalized}

                if ax is None or ax is np._NoValue:
                    ax_normalized = plt.gca()
                else:
                    ax_normalized = ax
                ax_normalized = na.as_named_array(ax_normalized)

                if not set(ax_normalized.shape).issubset(shape_orthogonal):
                    with pytest.raises(
                            expected_exception=ValueError,
                            match="the shape of .* should be a subset of .*"
                    ):
                        func(*args, **kwargs)
                    return

                for k in kwargs:
                    if not set(na.shape(kwargs[k])).issubset(shape_orthogonal):
                        with pytest.raises(
                            expected_exception=ValueError,
                            match="the shape of .* should be a subset of .*"
                        ):
                            func(*args, **kwargs)
                        return

                with astropy.visualization.quantity_support():
                    result = func(*args, **kwargs)

                assert isinstance(result, na.AbstractArray)
                assert result.dtype == matplotlib.artist.Artist

                if where is None or where is np._NoValue:
                    where_normalized = True
                else:
                    where_normalized = where
                where_normalized = na.broadcast_to(where_normalized, shape_orthogonal)

                for index in ax_normalized.ndindex():
                    if np.any(where_normalized[index]):
                        assert ax_normalized[index].ndarray.has_data()
                    ax_normalized[index].ndarray.clear()

        @pytest.mark.parametrize(
            argnames="ax, transformation",
            argvalues=[
                (None, None),
                (na.plt.subplots(axis_cols="x", ncols=num_x)[1], na.transformations.Translation(0))
            ],
        )
        class TestPltScatter(abc.ABC):
            def test_plt_scatter(
                self,
                array: na.AbstractArray,
                array_2: na.ArrayLike,
                s: None | float | na.AbstractScalar,
                c: None | str | na.AbstractScalar,
                ax: None | matplotlib.axes.Axes,
                where: bool | na.AbstractScalar,
                transformation: None | na.transformations.AbstractTransformation,
            ):
                args = (array_2, array)
                for arg in args:
                    if arg is None:
                        return

                with astropy.visualization.quantity_support():
                    na.plt.scatter(
                        *args,
                        s=s,
                        c=c,
                        ax=ax,
                        where=where,
                        transformation=transformation,
                    )

                if ax is None or ax is np._NoValue:
                    ax_normalized = plt.gca()
                else:
                    ax_normalized = ax
                ax_normalized = na.as_named_array(ax_normalized)

                for index in ax_normalized.ndindex():
                    assert ax_normalized[index].ndarray.has_data()
                    ax_normalized[index].ndarray.clear()

        class TestPltPcolormesh:

            @pytest.mark.parametrize("axis_rgb", [None, "rgb"])
            def test_pcolormesh(
                self,
                array: na.AbstractScalarArray,
                axis_rgb: None | str
            ):
                kwargs = dict(
                    C=array,
                    axis_rgb=axis_rgb,
                )

                if axis_rgb is not None:
                    with pytest.raises(ValueError):
                        na.plt.pcolormesh(**kwargs)
                    return

                if array.ndim != 2:
                    with pytest.raises(ValueError):
                        na.plt.pcolormesh(**kwargs)
                    return

                result = na.plt.pcolormesh(**kwargs)
                assert isinstance(result, na.ScalarArray)

        @pytest.mark.parametrize(
            argnames="dx",
            argvalues=[
                None,
            ]
        )
        class TestJacobian:

            def test_jacobian(
                    self,
                    function: Callable[[na.AbstractArray], na.AbstractArray],
                    array: na.AbstractVectorArray,
                    dx: None | float | na.AbstractVectorArray,
            ):
                x = array

                result = na.jacobian(
                    function=function,
                    x=x,
                    dx=dx,
                )

                assert np.all(result >= 0)
                assert isinstance(result, na.AbstractArray)

        class TestOptimizeRoot:

            def test_optimize_root(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    function: Callable[[na.AbstractArray], na.AbstractArray],
            ):
                def callback(i, x, f, c):
                    global out
                    out = x

                result = func(
                    function=function,
                    guess=array,
                    callback=callback,
                )

                assert np.all(np.abs(function(result)) < 1e-8)
                assert out is result

        class TestOptimizeMinimum:
            def test_optimize_minimum(
                self,
                func: Callable,
                array: na.AbstractArray,
                function: Callable[[na.AbstractArray], na.AbstractArray],
                expected: na.AbstractArray,
            ):
                def callback(i, x, f, c):
                    global out
                    out = x

                result = func(
                    function=function,
                    guess=array,
                    callback=callback,
                )

                assert np.allclose(na.value(result), expected)
                assert out is result

        class TestMeanFilter:
            def test_mean_filter(self, array: na.AbstractArray):

                size = dict(y=1)

                kwargs = dict(
                    array=array,
                    size=size,
                )

                if not set(size).issubset(array.shape):
                    with pytest.raises(ValueError):
                        na.ndfilters.mean_filter(**kwargs)
                    return

                result = na.ndfilters.mean_filter(**kwargs)

                assert np.all(result == array)

        @pytest.mark.parametrize("axis", [None, "y"])
        class TestColorsynth:
            def test_rgb(self, array: na.AbstractArray, axis: None | str):
                with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
                    if axis is None:
                        if array.ndim != 1:
                            with pytest.raises(ValueError):
                                na.colorsynth.rgb(array, axis=axis)
                            return
                        else:
                            result = na.colorsynth.rgb(array, axis=axis)
                            assert result.size == 3
                    else:
                        if array.shape:
                            result = na.colorsynth.rgb(array, axis=axis)
                            assert result.shape[axis] == 3

            def test_colorbar(self, array: na.AbstractArray, axis: None | str):
                if axis is None:
                    if array.ndim != 1:
                        with pytest.raises(ValueError):
                            na.colorsynth.colorbar(array, axis=axis)
                        return

                if array.shape:
                    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
                        result = na.colorsynth.colorbar(array, axis=axis)
                    assert isinstance(result, na.FunctionArray)
                    assert isinstance(result.inputs, na.Cartesian2dVectorArray)
                    assert isinstance(result.outputs, na.AbstractArray)

            def test_rgb_and_colorbar(self, array: na.AbstractArray, axis: None | str):
                with warnings.catch_warnings(action="ignore", category=RuntimeWarning):

                    if not array.shape:
                        return

                    if axis is None:
                        if array.ndim != 1:
                            return

                    try:
                        rgb_expected = na.colorsynth.rgb(array, axis=axis)
                        colorbar_expected = na.colorsynth.colorbar(array, axis=axis)
                    except TypeError:
                        return

                    rgb, colorbar = na.colorsynth.rgb_and_colorbar(array, axis=axis)

                    assert np.allclose(rgb, rgb_expected, equal_nan=True)
                    assert np.allclose(colorbar, colorbar_expected, equal_nan=True)


class AbstractTestAbstractExplicitArray(
    AbstractTestAbstractArray,
):

    @abc.abstractmethod
    def test__setitem__(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
            value: na.AbstractArray
    ):
        result = na.broadcast_to(array, array.shape).astype(float).copy()

        if isinstance(item, na.AbstractArray):
            if not set(item.shape).issubset(array.axes):
                with pytest.raises(
                    expected_exception=ValueError,
                    match="if `item` is an instance of .*, `item.axes`, .*, "
                          "should be a subset of `self.axes`, .*"
                ):
                    result[item] = value
                return

        elif isinstance(item, dict):
            if not set(item).issubset(array.axes):
                with pytest.raises(
                    expected_exception=ValueError,
                    match="if `item` is a .*, the keys in `item`, .*, "
                          "must be a subset of `self.axes`, .*"
                ):
                    result[item] = value
                return

            for axis in item:
                if isinstance(item[axis], int):
                    if axis in na.shape(value):
                        with pytest.raises(
                            expected_exception=ValueError,
                            match="`value` has an axis, .*, that is set to an `int` in `item`"
                        ):
                            result[item] = value
                        return

        try:
            value_0 = na.as_named_array(value).reshape(dict(dummy=-1))[dict(dummy=0)]
            result_0 = result.reshape(dict(dummy=-1))[dict(dummy=0)]
            value_0 + result_0
        except u.UnitConversionError as e:
            with pytest.raises((TypeError, u.UnitConversionError)):
                result[item] = value
            return

        result[item] = value
        assert np.all(result[item] == value)


class AbstractTestAbstractExplicitArrayCreation(
    abc.ABC,
):

    @pytest.mark.parametrize(
        argnames="a",
        argvalues=[
            None,
            2,
            2 * u.mm,
            np.array(2),
            np.array(2) * u.mm,
            na.ScalarArray(2),
            na.ScalarArray(2 * u.mm),
            na.ScalarLinearSpace(0, 1, axis="y", num=num_y),
            na.ScalarLinearSpace(0, 1, axis="y", num=num_y) * u.mm,
        ]
    )
    class TestFromScalarArray:
        @abc.abstractmethod
        def test_from_scalar_array(
                self,
                type_array: type[na.AbstractExplicitArray],
                a: None | float | u.Quantity | na.AbstractScalar,
                like: None | na.AbstractArray
        ):
            result = type_array.from_scalar_array(a=a, like=na.explicit(like))

            assert isinstance(result, type_array)
            if like is not None:
                assert isinstance(result, type(like))


class AbstractTestAbstractImplicitArray(
    abc.ABC,
):
    def test_explicit(self, array: na.AbstractArray):
        result = array.explicit
        assert isinstance(result, na.AbstractExplicitArray)
        assert np.abs(result.sum()) >= 0


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

class AbstractTestAbstractPoissonRandomSample(
    AbstractTestAbstractRandomMixin,
    AbstractTestAbstractImplicitArray,
):

    def test_center(self, array: na.AbstractSymmetricRangeMixin):
        assert isinstance(array.center, (int, float, complex, u.Quantity, na.AbstractArray))


class AbstractTestAbstractParameterizedArray(
    AbstractTestAbstractImplicitArray,
):

    def test_axis(self, array: na.AbstractParameterizedArray):
        assert isinstance(array.axis, (str, na.AbstractArray))

    def test_num(self, array: na.AbstractParameterizedArray):
        assert isinstance(array.num, (int, np.integer, na.AbstractArray))


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
