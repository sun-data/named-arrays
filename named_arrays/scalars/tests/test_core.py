from __future__ import annotations
from typing import Type, Sequence, Callable
import pytest
import numpy as np
import astropy.units as u
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
        super().test_unit(array)
        unit = array.unit
        if unit is not None:
            assert isinstance(unit, u.UnitBase)

    def test_unit_normalized(self, array: na.AbstractScalar):
        super().test_unit_normalized(array)
        if array.unit is None:
            assert array.unit_normalized == u.dimensionless_unscaled
        else:
            assert array.unit_normalized == array.unit

    @pytest.mark.parametrize(
        argnames='shape',
        argvalues=[
            dict(x=_num_x, y=_num_y),
            dict(x=_num_x, y=_num_y, z=13),
        ]
    )
    def test_broadcast_to(
            self,
            array: na.AbstractArray,
            shape: dict[str, int],
    ):
        super().test_broadcast_to(array=array, shape=shape)

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

        @pytest.mark.parametrize('axis', [None, 'y'])
        class TestArgReductionFunctions(
            AbstractTestAbstractScalar.TestArrayFunctions.TestArgReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize('axis', ['x', 'y'])
        class TestFFTLikeFunctions(
            AbstractTestAbstractScalar.TestArrayFunctions.TestFFTLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize('s', [None, dict(y=_num_y), dict(x=_num_x), dict(x=_num_x, y=_num_y)])
        class TestFFTNLikeFunctions(
            AbstractTestAbstractScalar.TestArrayFunctions.TestFFTNLikeFunctions,
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

