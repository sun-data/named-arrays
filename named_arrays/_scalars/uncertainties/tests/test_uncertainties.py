from typing import Type, Sequence, Callable
import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._scalars.tests.test_scalars

__all__ = [
    'AbstractTestAbstractUncertainScalarArray',
    'TestUncertainScalarArray',
    'TestUncertainScalarArrayCreation',
    'AbstractTestAbstractImplicitUncertainScalarArray',
    'TestUniformUncertainScalarArray',
    'TestNormalUncertainScalarArray',
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_distribution = named_arrays.tests.test_core.num_distribution


def _uncertain_scalar_arrays():
    arrays_nominal = [
        4.,
        na.ScalarArray(4.),
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y))
    ]
    arrays_distribution = [
        4 + na.ScalarUniformRandomSample(-0.1, 0.1, shape_random=dict(_distribution=_num_distribution)),
        4 + na.ScalarUniformRandomSample(
            start=-0.1,
            stop=0.1,
            shape_random=dict(x=_num_x, y=_num_y, _distribution=_num_distribution)
        ),
    ]
    units = [1, u.mm]
    arrays = [
        na.UncertainScalarArray(nominal * unit, distribution * unit)
        for nominal in arrays_nominal
        for distribution in arrays_distribution
        for unit in units
    ]
    return arrays


def _uncertain_scalar_arrays_2():
    arrays_exact = [
        5,
        na.ScalarUniformRandomSample(-5, 5, shape_random=dict(y=_num_y)),
    ]
    arrays_nominal = arrays_exact
    arrays_distribution = [
        na.ScalarArray(5.1).add_axes(na.UncertainScalarArray.axis_distribution),
        5 + na.ScalarUniformRandomSample(
            start=-5.1,
            stop=5.1,
            shape_random=dict(x=_num_x, y=_num_y, _distribution=_num_distribution),
        ),
    ]
    units = [1, u.mm]
    arrays_uncertain = [
        na.UncertainScalarArray(nominal * unit, distribution * unit)
        for nominal in arrays_nominal
        for distribution in arrays_distribution
        for unit in units
    ]
    arrays = arrays_exact + arrays_uncertain
    return arrays


class AbstractTestAbstractUncertainScalarArray(
    named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar,
):

    def test_nominal(self, array: na.AbstractUncertainScalarArray):
        assert np.sum(array.nominal) != 0
        assert array.axis_distribution not in na.shape(array.nominal)

    def test_distribution(self, array: na.AbstractUncertainScalarArray):
        assert np.sum(array.distribution) != 0

    def test_num_distribution(self, array: na.AbstractUncertainScalarArray):
        assert isinstance(array.num_distribution, int)
        assert array.num_distribution > 0

    def test_shape_distribution(self, array: na.AbstractUncertainScalarArray):
        assert isinstance(array.shape_distribution, dict)
        for ax in array.shape_distribution:
            assert isinstance(ax, str)
            assert isinstance(array.shape_distribution[ax], int)

        assert array.axis_distribution in array.shape_distribution

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=slice(0, 1)),
            dict(y=na.ScalarArray(np.array([0, 1]), axes=('y', ))),
            dict(
                y=na.UncertainScalarArray(
                    nominal=na.ScalarArray(np.array([0, 1]), axes=('y', )),
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0,], [1,]]),
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
            na.UncertainScalarArray(
                nominal=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                distribution=na.ScalarNormalRandomSample(
                    center=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                    width=0.1,
                    shape_random={na.UncertainScalarArray.axis_distribution: _num_distribution},
                )
            ) > 0.5,
        ]
    )
    def test__getitem__(
            self,
            array: na.AbstractUncertainScalarArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

        if isinstance(item, na.AbstractArray):

            if not set(item.shape).issubset(array.shape_distribution):
                with pytest.raises(ValueError):
                    array[item]
                return

            if isinstance(item, na.AbstractUncertainScalarArray):
                item_nominal = item.nominal & np.all(item.distribution, na.UncertainScalarArray.axis_distribution)
                item_distribution = item_nominal
            else:
                item_nominal = item_distribution = item

        elif isinstance(item, dict):

            if not set(item).issubset(array.shape_distribution):
                with pytest.raises(ValueError):
                    array[item]
                return

            item_nominal = dict()
            item_distribution = dict()

            for ax in item:
                if isinstance(item[ax], na.AbstractArray):
                    if isinstance(item[ax], na.AbstractUncertainScalarArray):
                        item_nominal[ax] = item[ax].nominal
                        item_distribution[ax] = item[ax].distribution
                    else:
                        item_nominal[ax] = item_distribution[ax] = item[ax]
                else:
                    item_nominal[ax] = item_distribution[ax] = item[ax]

        result = array[item]
        result_expected = na.UncertainScalarArray(
            array.broadcasted.nominal[item_nominal],
            array.broadcasted.distribution[item_distribution],
        )

        assert np.all(result == result_expected)

    def test__mul__(self, array: na.AbstractUncertainScalarArray):
        unit = u.mm
        result = array * unit
        result_nominal = array.nominal * unit
        result_distribution = array.distribution * unit
        assert np.all(result.nominal == result_nominal)
        assert np.all(result.distribution == result_distribution)

    def test__lshift__(self, array: na.AbstractUncertainScalarArray):
        unit = u.mm
        result = array << unit
        result_nominal = array.nominal << unit
        result_distribution = array.distribution << unit
        assert np.all(result.nominal == result_nominal)
        assert np.all(result.distribution == result_distribution)

    def test__truediv__(self, array: na.AbstractUncertainScalarArray):
        unit = u.mm
        result = array / unit
        result_nominal = array.nominal / unit
        result_distribution = array.distribution / unit
        assert np.all(result.nominal == result_nominal)
        assert np.all(result.distribution == result_distribution)

    class TestUfuncUnary(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestUfuncUnary
    ):

        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractUncertainScalarArray,
        ):
            super().test_ufunc_unary(ufunc, array)

            kwargs = dict()
            kwargs_nominal = dict()
            kwargs_distribution = dict()

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
                kwargs_nominal["where"] = kwargs["where"].nominal
                kwargs_distribution["where"] = kwargs["where"].distribution

            try:
                ufunc(array.nominal, **kwargs_nominal)
                ufunc(array.distribution, **kwargs_distribution)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    ufunc(array, **kwargs)
                return

            result = ufunc(array, **kwargs)
            result_nominal = ufunc(array.nominal, **kwargs_nominal)
            result_distribution = ufunc(array.distribution, **kwargs_distribution)

            if ufunc.nout == 1:
                out = 0 * result
            else:
                out = tuple(0 * r for r in result)

            result_out = ufunc(array, out=out, **kwargs)

            if ufunc.nout == 1:
                out = (out, )
                result = (result, )
                result_nominal = (result_nominal, )
                result_distribution = (result_distribution, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].nominal == result_nominal[i], **kwargs_nominal)
                assert np.all(result[i].distribution == result_distribution[i], **kwargs_distribution)
                assert np.all(result[i] == result_out[i], **kwargs)
                assert result_out[i] is out[i]

    @pytest.mark.parametrize('array_2', _uncertain_scalar_arrays_2())
    class TestUfuncBinary(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestUfuncBinary
    ):

        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | na.AbstractUncertainScalarArray,
                array_2: None | bool | int | float | complex | na.AbstractUncertainScalarArray,
        ):
            super().test_ufunc_binary(ufunc=ufunc, array=array, array_2=array_2)

            if not isinstance(array, na.AbstractUncertainScalarArray):
                array_normalized = na.UncertainScalarArray(
                    nominal=array,
                    distribution=na.add_axes(array, na.UncertainScalarArray.axis_distribution),
                )
            else:
                array_normalized = array

            if not isinstance(array_2, na.AbstractUncertainScalarArray):
                array_2_normalized = na.UncertainScalarArray(
                    nominal=array_2,
                    distribution=na.add_axes(array_2, na.UncertainScalarArray.axis_distribution),
                )
            else:
                array_2_normalized = array_2

            kwargs = dict()
            kwargs_nominal = dict()
            kwargs_distribution = dict()

            unit_2 = na.unit_normalized(array_2)
            if ufunc in [np.power, np.float_power]:
                kwargs["where"] = (array_2_normalized >= (1 * unit_2)) & (array_normalized >= 0)
            elif ufunc in [np.divide, np.floor_divide, np.remainder, np.fmod, np.divmod]:
                kwargs["where"] = array_2_normalized != 0

            if "where" in kwargs:
                kwargs_nominal["where"] = kwargs["where"].nominal
                kwargs_distribution["where"] = kwargs["where"].distribution

            try:
                ufunc(array_normalized.nominal, array_2_normalized.nominal, **kwargs_nominal)
                ufunc(array_normalized.distribution, array_2_normalized.distribution, **kwargs_distribution)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    ufunc(array, array_2, **kwargs)
                return

            result = ufunc(array, array_2, **kwargs)
            result_nominal = ufunc(array_normalized.nominal, array_2_normalized.nominal, **kwargs_nominal)
            result_distribution = ufunc(
                array_normalized.distribution, array_2_normalized.distribution, **kwargs_distribution)

            if ufunc.nout == 1:
                out = 0 * np.nan_to_num(result)
            else:
                out = tuple(0 * np.nan_to_num(r) for r in result)

            result_out = ufunc(array, array_2, out=out, **kwargs)

            if ufunc.nout == 1:
                out = (out, )
                result = (result, )
                result_nominal = (result_nominal, )
                result_distribution = (result_distribution, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].nominal == result_nominal[i], **kwargs_nominal)
                assert np.all(result[i].distribution == result_distribution[i], **kwargs_distribution)
                assert np.all(result[i] == result_out[i], **kwargs)
                assert result_out[i] is out[i]

    @pytest.mark.parametrize('array_2', _uncertain_scalar_arrays_2())
    class TestMatmul(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions,
    ):

        @pytest.mark.parametrize("array_2", _uncertain_scalar_arrays_2())
        class TestAsArrayLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions
            .TestAsArrayLikeFunctions
        ):

            def test_asarray_like_functions(
                    self,
                    func: Callable,
                    array: None | float | u.Quantity | na.AbstractArray,
                    array_2: None | float | u.Quantity | na.AbstractArray,
            ):
                a = array
                like = array_2

                if a is None:
                    assert func(a, like=like) is None
                    return

                result = func(a, like=like)

                assert isinstance(result, na.UncertainScalarArray)
                assert isinstance(result.nominal, na.ScalarArray)
                assert isinstance(result.distribution, na.ScalarArray)
                assert isinstance(result.nominal.ndarray, np.ndarray)
                assert isinstance(result.distribution.ndarray, np.ndarray)

                assert np.all(result.value == na.value(a))

                super().test_asarray_like_functions(
                    func=func,
                    array=array,
                    array_2=array_2,
                )

        class TestSingleArgumentFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestSingleArgumentFunctions,
        ):
            def test_single_argument_functions(
                self,
                func: Callable,
                array: na.AbstractUncertainScalarArray,
            ):
                result = func(array)
                assert np.all(result.nominal == func(array.nominal))
                assert np.all(result.distribution == func(array.distribution))

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
                (na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                na.UncertainScalarArray(
                    nominal=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0)
                            | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                    distribution=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0)
                                 | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0)
                                 | (na.ScalarLinearSpace(-1, 1, '_distribution', _num_distribution) >= 0)
                ),
            ]
        )
        class TestReductionFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.
            TestReductionFunctions,
        ):

            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
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
                    dtype=dtype,
                    keepdims=keepdims,
                    where=where,
                )

                kwargs_nominal = kwargs.copy()
                kwargs_distribution = kwargs.copy()

                if axis is None:
                    axis_normalized = na.axis_normalized(array, axis)
                    kwargs_nominal["axis"] = axis_normalized
                    kwargs_distribution["axis"] = axis_normalized

                if isinstance(where, na.AbstractUncertainScalarArray):
                    kwargs_nominal["where"] = where.nominal
                    kwargs_distribution["where"] = where.distribution
                else:
                    kwargs_nominal["where"] = kwargs_distribution["where"] = where

                try:
                    result_nominal = func(array.broadcasted.nominal, **kwargs_nominal)
                    result_distribution = func(array.broadcasted.distribution, **kwargs_distribution)
                except (ValueError, TypeError, u.UnitsError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)
                assert np.allclose(result, result_out)
                assert result_out is out

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                .25,
                25 * u.percent,
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
                na.UncertainScalarArray(
                    nominal=25 * u.percent,
                    distribution=na.ScalarNormalRandomSample(
                        center=25 * u.percent,
                        width=1 * u.percent,
                        shape_random=dict(_distribution=_num_distribution)
                    )
                )
            ]
        )
        class TestPercentileLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.
            TestPercentileLikeFunctions
        ):

            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    keepdims: bool,
            ):
                super().test_percentile_like_functions(
                    func=func,
                    array=array,
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                array_normalized = array
                if isinstance(array, na.AbstractArray):
                    if isinstance(array, na.AbstractScalar):
                        if isinstance(array, na.AbstractScalarArray):
                            array_normalized = na.UncertainScalarArray(array, array)

                kwargs = dict(
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                kwargs_nominal = kwargs.copy()
                kwargs_distribution = kwargs.copy()

                if isinstance(q, na.AbstractUncertainScalarArray):
                    kwargs_nominal["q"] = q.nominal
                    kwargs_distribution["q"] = q.distribution
                else:
                    kwargs_nominal["q"] = kwargs_distribution["q"] = q

                if axis is None:
                    axis_normalized = na.axis_normalized(array, axis)
                    kwargs_nominal["axis"] = axis_normalized
                    kwargs_distribution["axis"] = axis_normalized

                try:
                    result_nominal = func(array_normalized.broadcasted.nominal, **kwargs_nominal)
                    result_distribution = func(array_normalized.broadcasted.distribution, **kwargs_distribution)
                except (ValueError, TypeError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)
                assert np.all(result == result_out)
                assert result_out is out

        class TestFFTLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestFFTLikeFunctions,
        ):

            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    axis: tuple[str, str],
            ):
                if axis[0] not in array.shape:
                    with pytest.raises(ValueError, match="`axis` .* not in array with shape .*"):
                        func(array, axis=axis)
                    return

                result = func(array, axis=axis)
                result_nominal = func(array.broadcasted.nominal, axis=axis)
                result_distribution = func(array.broadcasted.distribution, axis=axis)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)

        class TestFFTNLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestFFTNLikeFunctions
        ):
            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    axes: dict[str, str],
                    s: None | dict[str, int],
            ):
                if not set(axes).issubset(array.shape):
                    with pytest.raises(ValueError, match="`axes`, .*, not a subset of array axes, .*"):
                        func(array, axes=axes, s=s)
                    return

                if s is not None and axes.keys() != s.keys():
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                result = func(array, axes=axes, s=s)
                result_nominal = func(array.broadcasted.nominal, axes=axes, s=s)
                result_distribution = func(array.broadcasted.distribution, axes=axes, s=s)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)

        class TestEmathFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestEmathFunctions,
        ):
            def test_emath_functions(
                self,
                func: Callable,
                array: na.AbstractUncertainScalarArray,
            ):
                result = func(array)
                assert np.all(result.nominal == func(array.nominal))
                assert np.all(result.distribution == func(array.distribution))

        @pytest.mark.parametrize('axis', [None, 'x', 'y', ('x', 'y'), ()])
        def test_sort(self, array: na.AbstractUncertainScalarArray, axis: None | str | Sequence[str]):

            axis_normalized = na.axis_normalized(array, axis)

            if axis is not None:
                if not axis:
                    with pytest.raises(ValueError, match=f"if `axis` is a sequence, it must not be empty, got .*"):
                        np.sort(array, axis=axis)
                    return

                if not set(axis_normalized).issubset(array.shape):
                    with pytest.raises(ValueError, match="`axis`, .* is not a subset of `a.axes`, .*"):
                        np.sort(array, axis=axis)
                    return

            result = np.sort(array, axis=axis)

            array_broadcasted = na.broadcast_to(array, array.shape)
            if axis_normalized:
                result_distribution = np.sort(
                    a=array_broadcasted.distribution.mean(array.axis_distribution),
                    axis=axis_normalized,
                )
            else:
                result_distribution = array.distribution.mean(array.axis_distribution)

            assert np.all(result.distribution.mean(array.axis_distribution) == result_distribution)

        @pytest.mark.parametrize('copy', [False, True])
        def test_nan_to_num(
                self,
                array: na.AbstractUncertainScalarArray,
                copy: bool,
        ):

            super().test_nan_to_num(array=array, copy=copy)

            if not copy and not isinstance(array, na.AbstractExplicitArray):
                with pytest.raises(TypeError, match="can't write to an array .*"):
                    np.nan_to_num(array, copy=copy)
                return

            result = np.nan_to_num(array, copy=copy)
            result_nominal = np.nan_to_num(array.nominal, copy=copy)
            result_distribution = np.nan_to_num(array.distribution, copy=copy)

            assert np.all(result.nominal == result_nominal)
            assert np.all(result.distribution == result_distribution)

        @pytest.mark.parametrize('v', _uncertain_scalar_arrays_2())
        @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
        def test_convolve(self, array: na.AbstractArray, v: na.AbstractArray, mode: str):
            super().test_convolve(array=array, v=v, mode=mode)
            with pytest.raises(ValueError, match="`numpy.convolve` is not supported .*"):
                np.convolve(array, v=v, mode=mode)

    class TestNamedArrayFunctions(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions
    ):
        @pytest.mark.parametrize(
            argnames="array_2",
            argvalues=_uncertain_scalar_arrays_2(),
        )
        @pytest.mark.parametrize(
            argnames="where",
            argvalues=[
                np._NoValue,
                True,
                na.linspace(0, 1, axis="x", num=_num_x) > 0.5,
            ]
        )
        @pytest.mark.parametrize(
            argnames="alpha",
            argvalues=[
                np._NoValue,
                na.linspace(0, 1, axis="x", num=_num_x),
            ]
        )
        class TestPltPlotLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="array_2",
            argvalues=_uncertain_scalar_arrays_2(),
        )
        @pytest.mark.parametrize(
            argnames="s",
            argvalues=[
                None,
            ]
        )
        @pytest.mark.parametrize(
            argnames="c",
            argvalues=[
                None,
            ]
        )
        @pytest.mark.parametrize(
            argnames="where",
            argvalues=[
                True,
                na.linspace(0, 1, axis="x", num=_num_x) > 0.5,
            ]
        )
        class TestPltScatter(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestPltScatter,
        ):
            pass

        @pytest.mark.skip
        class TestPltPcolormesh(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestPltPcolormesh,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="function",
            argvalues=[
                lambda x: a * x ** 3
                for a in [2, na.UniformUncertainScalarArray(2, width=0.5, num_distribution=_num_distribution)]
            ]
        )
        class TestJacobian(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestJacobian,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                na.optimize.root_secant,
                na.optimize.root_newton,
            ],
        )
        @pytest.mark.parametrize(
            argnames="function",
            argvalues=[
                lambda x: np.square(na.value(x) - shift_horizontal) + shift_vertical
                for shift_horizontal in [
                    20,
                    na.UniformUncertainScalarArray(20, width=1, num_distribution=_num_distribution),
                ]
                for shift_vertical in [-1]
            ]
        )
        class TestOptimizeRoot(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestOptimizeRoot,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeMinimum(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestOptimizeMinimum,
        ):
            pass


@pytest.mark.parametrize('array', _uncertain_scalar_arrays())
class TestUncertainScalarArray(
    AbstractTestAbstractUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray
):

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(x=0, y=0),
            dict(y=slice(None)),
            dict(y=na.ScalarArrayRange(0, _num_y, axis='y')),
            dict(x=na.ScalarArrayRange(0, _num_x, axis='x'), y=na.ScalarArrayRange(0, _num_y, axis='y')),
            na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.ScalarUniformRandomSample(-5, 5, dict(y=_num_y)),
            na.UncertainScalarUniformRandomSample(
                start=na.UniformUncertainScalarArray(-5, 0, num_distribution=_num_distribution),
                stop=5,
                shape_random=dict(y=_num_y),
            )
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


@pytest.mark.parametrize("type_array", [na.UncertainScalarArray])
class TestUncertainScalarArrayCreation(
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation,
):

    @pytest.mark.parametrize("like", [None] + _uncertain_scalar_arrays())
    class TestFromScalarArray(
        named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation.TestFromScalarArray,
    ):
        pass


class AbstractTestAbstractImplicitUncertainScalarArray(
    AbstractTestAbstractUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractImplicitArray,
):
    pass


def _uniform_uncertain_scalar_arrays():
    arrays_exact = [
        4,
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y)),
    ]
    widths = [
        1,
        na.ScalarLinearSpace(1, 2, axis='y', num=_num_y)
    ]
    units = [
        1,
    ]
    arrays = [
        na.UniformUncertainScalarArray(
            nominal=array_exact * unit,
            width=width * unit,
            num_distribution=_num_distribution
        )
        for array_exact in arrays_exact
        for width in widths
        for unit in units
    ]
    return arrays


@pytest.mark.parametrize('array', _uniform_uncertain_scalar_arrays())
class TestUniformUncertainScalarArray(
    AbstractTestAbstractImplicitUncertainScalarArray,
):
    pass


def _normal_uncertain_scalar_arrays():
    arrays_exact = [
        na.ScalarArray(4),
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y)),
    ]
    widths = [
        1,
        na.ScalarLinearSpace(1, 2, axis='y', num=_num_y)
    ]
    units = [
        u.mm,
    ]
    arrays = [
        na.NormalUncertainScalarArray(
            nominal=array_exact * unit,
            width=width * unit,
            num_distribution=_num_distribution
        )
        for array_exact in arrays_exact
        for width in widths
        for unit in units
    ]
    return arrays


@pytest.mark.parametrize('array', _normal_uncertain_scalar_arrays())
class TestNormalUncertainScalarArray(
    AbstractTestAbstractImplicitUncertainScalarArray,
):
    pass


class AbstractTestAbstractUncertainScalarRandomSample(
    AbstractTestAbstractImplicitUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractRandomSample,
):
    pass


def _uncertain_scalar_uniform_random_samples() -> tuple[na.UncertainScalarUniformRandomSample, ...]:

    starts = (
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            width=0.2,
            num_distribution=_num_distribution,
        ),
    )
    stops = (
        2,
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = [
        u.mm,
    ]
    arrays = tuple(
        na.UncertainScalarUniformRandomSample(
            start=start * unit,
            stop=stop * unit,
            shape_random=dict(y=_num_y),
        )
        for start in starts
        for stop in stops
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_uniform_random_samples())
class TestUncertainScalarUniformRandomSample(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractUniformRandomSample,
):
    pass


def _uncertain_scalar_normal_random_samples() -> tuple[na.UncertainScalarNormalRandomSample, ...]:

    centers = (
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            width=0.2,
            num_distribution=_num_distribution,
        ),
    )
    widths = (
        na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = [
        1,
    ]
    arrays = tuple(
        na.UncertainScalarNormalRandomSample(
            center=center * unit,
            width=width * unit,
            shape_random=dict(y=_num_y),
        )
        for center in centers
        for width in widths
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_normal_random_samples())
class TestUncertainScalarNormalRandomSample(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractNormalRandomSample,
):
    pass


def _uncertain_scalar_poisson_random_samples() -> tuple[na.UncertainScalarPoissionRandomSample, ...]:

    centers = (
        na.UniformUncertainScalarArray(2, width=0.1, num_distribution=_num_distribution),
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = (1, u.mm)
    arrays = tuple(
        na.UncertainScalarPoissionRandomSample(
            center=center * unit,
            shape_random=dict(y=_num_y),
        )
        for center in centers
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_poisson_random_samples())
class TestUncertainScalarPoissonRandomSample(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractPoissonRandomSample,
):
    pass



class AbstractTestAbstractParameterizedUncertainScalarArray(
    AbstractTestAbstractImplicitUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractParameterizedArray,
):
    pass


class TestUncertainScalarLinearSpace(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractPoissonRandomSample,
):
    pass


class AbstractTestAbstractScalarSpace(
    AbstractTestAbstractParameterizedUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractSpace,
):
    pass


def _uncertain_scalar_linear_spaces() -> tuple[na.UncertainScalarLinearSpace, ...]:

    starts = (
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            width=0.2,
            num_distribution=_num_distribution,
        ),
    )
    stops = (
        2,
        na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = (1, u.mm)
    arrays = tuple(
        na.UncertainScalarLinearSpace(
            start=start * unit,
            stop=stop * unit,
            axis='y',
            num=_num_y,
        )
        for start in starts
        for stop in stops
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_linear_spaces())
class TestUncertainScalarLinearSpace(
    AbstractTestAbstractScalarSpace,
    named_arrays.tests.test_core.AbstractTestAbstractLinearSpace,
):
    @pytest.mark.parametrize(
        argnames="axis",
        argvalues=[
            None,
            "x",
            "y",
            "z",
        ]
    )
    def test_volume_cell(
            self,
            array: na.ScalarLinearSpace,
            axis: None | str | Sequence[str],
    ):
        super().test_volume_cell(array=array, axis=axis)

        axis_ = na.axis_normalized(array, axis)
        if len(axis_) != 1:
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        if not set(axis_).issubset(array.shape):
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        assert np.allclose(array.volume_cell(axis), array.explicit.volume_cell(axis))
