from typing import Sequence, Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._vectors.cartesian.tests.test_vectors_cartesian_2d

__all__ = [
    "AbstractTestAbstractFunctionArray",
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_distribution = named_arrays.tests.test_core.num_distribution


def _function_arrays():
    functions_1d = [
        na.FunctionArray(
            inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
            outputs=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(y=_num_y))
        )
    ]

    inputs_2d = [
        na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
        na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=1,
            axis=na.Cartesian2dVectorArray('x', 'y'),
            num=na.Cartesian2dVectorArray(_num_x, _num_y)
        )
    ]

    outputs_2d = [
        na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
        na.UniformUncertainScalarArray(
            nominal=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
            width=1,
            num_distribution=_num_distribution,
        ),
        na.Cartesian2dVectorUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
    ]

    functions_2d = [
        na.FunctionArray(
            inputs=inputs,
            outputs=outputs,
        )
        for inputs in inputs_2d
        for outputs in outputs_2d
    ]

    functions = functions_1d + functions_2d

    return functions


def _function_arrays_2():
    return (
        6,
        na.ScalarArray(6),
        na.UniformUncertainScalarArray(6, width=1, num_distribution=_num_distribution),
        na.Cartesian2dVectorArray(6, 7),
        na.FunctionArray(
            inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
            outputs=na.ScalarUniformRandomSample(-6, 6, shape_random=dict(y=_num_y))
        ),
        na.FunctionArray(
            inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
            outputs=na.Cartesian2dVectorUniformRandomSample(-6, 6, shape_random=dict(y=_num_y))
        )
    )


class AbstractTestAbstractFunctionArray(
    named_arrays.tests.test_core.AbstractTestAbstractArray,
):

    def test_inputs(self, array: na.AbstractFunctionArray):
        assert isinstance(array.inputs, na.AbstractArray)

    def test_outputs(self, array: na.AbstractFunctionArray):
        assert isinstance(array.outputs, na.AbstractArray)

    @pytest.mark.parametrize("dtype", (int, float))
    def test_astype(self, array: na.AbstractFunctionArray, dtype: type):
        super().test_astype(array=array, dtype=dtype)
        result = array.astype(dtype)
        if isinstance(result.outputs, na.AbstractScalar):
            assert result.outputs.dtype == dtype
        elif isinstance(result.outputs, na.AbstractVectorArray):
            for e in result.outputs.entries:
                assert result.outputs.entries[e].dtype == dtype

    @pytest.mark.parametrize('unit', [u.m, u.s])
    def test_to(self, array: na.AbstractFunctionArray, unit: None | u.UnitBase):
        super().test_to(array=array, unit=unit)
        if isinstance(array.outputs, na.ScalarArray):
            if na.unit_normalized(array.outputs).is_equivalent(unit):
                result = array.to(unit)
                assert result.outputs.unit == unit
                return

        if isinstance(array.outputs, na.AbstractVectorArray):
            entries = array.outputs.entries
            if all(unit.is_equivalent(na.unit_normalized(entries[e])) for e in entries):
                result = array.to(unit)
                assert result.type_abstract == array.type_abstract
                assert all(result.outputs.entries[e].unit == unit for e in result.outputs.entries)
                return

        with pytest.raises(u.UnitConversionError):
            array.to(unit)

    def test_length(self, array: na.AbstractFunctionArray):
        result = array.length
        assert np.all(result.outputs.length == array.outputs.length)
        assert np.all(result.inputs == array.inputs)

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=slice(0, 1)),
            dict(y=na.ScalarArrayRange(0, 2, axis='y')),
            dict(
                y=na.FunctionArray(
                    inputs=na.ScalarArrayRange(0, 2, axis='y'),
                    outputs=na.ScalarArrayRange(0, 2, axis='y'),
                )
            ),
            na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                outputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
            ),
            na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                outputs=na.UniformUncertainScalarArray(
                    nominal=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                    width=0.1,
                    num_distribution=_num_distribution,
                ) > 0.5,
            )
        ],
    )
    def test__getitem__(
            self,
            array: na.AbstractFunctionArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

        if not array.shape:
            with pytest.raises(ValueError):
                array[item]
            return

        if isinstance(item, na.AbstractFunctionArray):
            if np.any(item.inputs != array.inputs):
                with pytest.raises(ValueError):
                    array[item]
                return

            item_outputs = item_inputs = item.outputs

        elif isinstance(item, dict):
            item_inputs = {
                ax: item[ax].inputs if isinstance(item[ax], na.AbstractFunctionArray) else item[ax]
                for ax in item
            }
            item_outputs = {
                ax: item[ax].outputs if isinstance(item[ax], na.AbstractFunctionArray) else item[ax]
                for ax in item
            }

        result = array[item]

        assert np.all(result.inputs == array.inputs[item_inputs])
        assert np.all(result.outputs == array.outputs[item_outputs])

    def test__bool__(self, array: na.AbstractFunctionArray):
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

    def test__mul__(self, array: na.AbstractFunctionArray):
        unit = u.mm
        result = array * unit
        assert np.all(result.outputs == array.outputs * unit)
        assert np.all(result.inputs == array.inputs)

    def test__lshift__(self, array: na.AbstractFunctionArray):
        unit = u.mm
        result = array << unit
        assert np.all(result.outputs == array.outputs << unit)
        assert np.all(result.inputs == array.inputs)

    def test__truediv__(self, array: na.AbstractFunctionArray):
        unit = u.mm
        result = array / unit
        assert np.all(result.outputs == array.outputs / unit)
        assert np.all(result.inputs == array.inputs)

    class TestUfuncUnary(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestUfuncUnary
    ):

        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractFunctionArray,
        ):
            try:
                outputs_expected = ufunc(array.outputs)
                inputs_expected = array.inputs
                if ufunc.nout != 1:
                    inputs_expected = (inputs_expected, ) * ufunc.nout
            except Exception as e:
                with pytest.raises(type(e)):
                    ufunc(array)
                return

            result = ufunc(array)

            if ufunc.nout == 1:
                out = 0 * result
            else:
                out = tuple(0 * r for r in result)

            result_out = ufunc(array, out=out)

            if ufunc.nout == 1:
                out = (out, )
                outputs_expected = (outputs_expected, )
                inputs_expected = (inputs_expected, )
                result = (result, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].outputs == outputs_expected[i])
                assert np.all(result[i].inputs == inputs_expected[i])
                assert np.all(result[i] == result_out[i])
                assert result_out[i] is out[i]

    @pytest.mark.parametrize("array_2", _function_arrays_2())
    class TestUfuncBinary(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestUfuncBinary
    ):

        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            if isinstance(array, na.AbstractFunctionArray):
                array_normalized = array
                if isinstance(array_2, na.AbstractFunctionArray):
                    array_2_normalized = array_2
                else:
                    array_2_normalized = na.FunctionArray(inputs=array.inputs, outputs=array_2)
            else:
                if isinstance(array_2, na.AbstractFunctionArray):
                    array_normalized = na.FunctionArray(inputs=array_2.inputs, outputs=array)
                    array_2_normalized = array_2
                else:
                    raise TypeError("neither `array` nor `array_2` are instances of`named_arrays.FunctionArray`")

            try:
                outputs_expected = ufunc(array_normalized.outputs, array_2_normalized.outputs)
                inputs_expected = array_normalized.inputs
                if ufunc.nout != 1:
                    inputs_expected = (inputs_expected, ) * ufunc.nout
            except Exception as e:
                with pytest.raises(type(e)):
                    ufunc(array, array_2)
                return

            result = ufunc(array, array_2)

            if ufunc.nout == 1:
                out = 0 * result
            else:
                out = tuple(0 * r for r in result)

            result_out = ufunc(array, array_2, out=out)

            if ufunc.nout == 1:
                out = (out, )
                outputs_expected = (outputs_expected, )
                inputs_expected = (inputs_expected, )
                result = (result, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].outputs == outputs_expected[i])
                assert np.all(result[i].inputs == inputs_expected[i])
                assert np.all(result[i] == result_out[i])
                assert result_out[i] is out[i]

    @pytest.mark.parametrize("array_2", _function_arrays_2())
    class TestMatmul(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestMatmul
    ):

        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):
            if isinstance(array, na.AbstractFunctionArray):
                array_normalized = array
                if isinstance(array_2, na.AbstractFunctionArray):
                    array_2_normalized = array_2
                else:
                    array_2_normalized = na.FunctionArray(inputs=array.inputs, outputs=array_2)
            else:
                if isinstance(array_2, na.AbstractFunctionArray):
                    array_normalized = na.FunctionArray(inputs=array_2.inputs, outputs=array)
                    array_2_normalized = array_2
                else:
                    raise TypeError("neither `array` nor `array_2` are instances of`named_arrays.FunctionArray`")

            try:
                outputs_expected = np.matmul(array_normalized.outputs, array_2_normalized.outputs)
                inputs_expected = array_normalized.inputs
            except Exception as e:
                with pytest.raises(type(e)):
                    np.matmul(array, array_2)
                return

            result = np.matmul(array, array_2)

            assert np.all(result.outputs == outputs_expected)
            assert np.all(result.inputs == inputs_expected)

    class TestArrayFunctions(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions
    ):
        @pytest.mark.parametrize("array_2", _function_arrays_2())
        class TestAsArrayLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestAsArrayLikeFunctions
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

                assert isinstance(result, na.FunctionArray)
                assert isinstance(result.inputs, na.AbstractArray)
                assert isinstance(result.outputs, na.AbstractArray)

                assert np.all(result.value == na.value(a))

                super().test_asarray_like_functions(
                    func=func,
                    array=array,
                    array_2=array_2,
                )

        @pytest.mark.skip
        class TestSingleArgumentFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestSingleArgumentFunctions,
        ):
            def test_single_argument_functions(
                self,
                func: Callable,
                array: na.AbstractFunctionArray,
            ):
                assert False

        @pytest.mark.xfail
        class TestArrayCreationLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestArrayCreationLikeFunctions
        ):
            pass

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
                na.FunctionArray(
                    inputs=na.Cartesian2dVectorLinearSpace(
                        start=0,
                        stop=1,
                        axis=na.Cartesian2dVectorArray('x', 'y'),
                        num=na.Cartesian2dVectorArray(_num_x, _num_y)
                    ),
                    outputs=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0)
                            | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0)
                )
            ]
        )
        class TestReductionFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestReductionFunctions
        ):

            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractFunctionArray,
                    axis: None | str | Sequence[str],
                    dtype: None | type | np.dtype,
                    keepdims: bool,
                    where: bool | na.AbstractFunctionArray,
            ):

                array_broadcasted = na.broadcast_to(array, na.shape_broadcasted(array, where))

                where_normalized = where.outputs if isinstance(where, na.FunctionArray) else where

                kwargs = dict()
                kwargs_output = dict()
                if dtype is not np._NoValue:
                    kwargs["dtype"] = kwargs_output["dtype"] = dtype
                if func in [np.min, np.nanmin, np.max, np.nanmax]:
                    kwargs["initial"] = kwargs_output["initial"] = 0
                if where is not np._NoValue:
                    kwargs["where"] = where
                    kwargs_output["where"] = where_normalized

                try:
                    if isinstance(where, na.FunctionArray):
                        if np.any(where.inputs != array.inputs):
                            raise na.InputValueError
                    outputs_expected = func(
                        a=array_broadcasted.outputs,
                        axis=axis,
                        keepdims=keepdims,
                        **kwargs_output,
                    )
                    if keepdims:
                        inputs_expected = array_broadcasted.inputs
                    else:
                        inputs_expected = np.mean(
                            a=array_broadcasted.inputs,
                            axis=axis,
                            keepdims=keepdims,
                            where=where_normalized,
                        )
                except Exception as e:
                    with pytest.raises(type(e)):
                        func(array, axis=axis, keepdims=keepdims, **kwargs)
                    return

                result = func(array, axis=axis, keepdims=keepdims, **kwargs)

                out = 0 * result
                out.inputs = 0 * out.inputs
                result_out = func(array, axis=axis, out=out, keepdims=keepdims, **kwargs)

                assert np.all(result.inputs == inputs_expected)
                assert np.all(result.outputs == outputs_expected)
                assert np.all(result == result_out)
                assert result_out is out

        class TestPercentileLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestPercentileLikeFunctions,
        ):

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
            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractFunctionArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    keepdims: bool,
            ):

                array_broadcasted = na.broadcast_to(array, array.shape)

                kwargs = dict()
                kwargs_output = dict()

                try:
                    outputs_expected = func(
                        a=array_broadcasted.outputs,
                        q=q,
                        axis=axis,
                        keepdims=keepdims,
                        **kwargs_output,
                    )
                    if keepdims:
                        inputs_expected = array_broadcasted.inputs
                    else:
                        inputs_expected = np.mean(
                            a=array_broadcasted.inputs,
                            axis=axis,
                            keepdims=keepdims,
                        )
                except Exception as e:
                    with pytest.raises(type(e)):
                        func(array, q=q, axis=axis, keepdims=keepdims, **kwargs)
                    return

                result = func(array, q=q, axis=axis, keepdims=keepdims, **kwargs)

                out = 0 * result
                out.inputs = 0 * out.inputs
                result_out = func(array, q=q, axis=axis, out=out, keepdims=keepdims, **kwargs)

                assert np.all(result.inputs == inputs_expected)
                assert np.all(result.outputs == outputs_expected)
                assert np.all(result == result_out)
                assert result_out is out

        @pytest.mark.skip
        class TestArgReductionFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestArgReductionFunctions
        ):

            def test_arg_reduction_functions(
                    self,
                    argfunc: Callable,
                    func: Callable,
                    array: na.AbstractFunctionArray,
                    axis: None | str,
            ):
                try:
                    outputs_expected = argfunc(array.outputs, axis=axis)
                except Exception as e:
                    with pytest.raises(type(e)):
                        argfunc(array, axis=axis)
                    return

                result = argfunc(array, axis=axis)

                assert np.all(result.outputs == outputs_expected)


        class TestFFTLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTLikeFunctions,
        ):

            @pytest.mark.skip
            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractFunctionArray,
                    axis: tuple[str, str],
            ):
                pass

        class TestFFTNLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTNLikeFunctions,
        ):

            @pytest.mark.skip
            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractFunctionArray,
                    axes: dict[str, str],
                    s: None | dict[str, int],
            ):
                pass

        class TestEmathFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestEmathFunctions,
        ):

            @pytest.mark.skip
            def test_emath_functions(
                self,
                func: Callable,
                array: na.AbstractArray,
            ):
                pass

        @pytest.mark.xfail
        def test_sort(self, array: na.AbstractFunctionArray, axis: None | str | Sequence[str]):
            pass

        @pytest.mark.xfail
        def test_argsort(self, array: na.AbstractArray, axis: None | str | Sequence[str]):
            super().test_argsort(array=array, axis=axis)

        @pytest.mark.xfail
        def test_unravel_index(self, array: na.AbstractArray):
            super().test_unravel_index(array=array)

        @pytest.mark.skip
        def test_where(self, array: na.AbstractArray):
            super().test_where(array=array)

        @pytest.mark.skip
        def test_nan_to_num(self, array: na.AbstractFunctionArray, copy: bool):
            pass

        @pytest.mark.skip
        def test_convolve(self, array: na.AbstractFunctionArray, v: na.AbstractArray, mode: str):
            pass

        @pytest.mark.skip
        def test_diff_1st_order(
            self,
            array: na.AbstractArray,
            axis: str,
        ):
            pass    # pragma: nocover

    class TestNamedArrayFunctions(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions
    ):
        @pytest.mark.skip
        class TestInterp(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestInterp,
        ):
            pass

        @pytest.mark.xfail
        class TestPltPlotLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions
        ):
            pass

        @pytest.mark.skip
        class TestPltScatter(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestPltScatter
        ):
            pass

        class TestPltPcolormesh(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestPltPcolormesh,
        ):
            @pytest.mark.parametrize("axis_rgb", [None, "rgb"])
            def test_pcolormesh(
                self,
                array: na.AbstractScalarArray,
                axis_rgb: None | str
            ):
                if not isinstance(array.inputs, na.AbstractVectorArray):
                    return

                components = list(array.inputs.components.keys())[:2]

                kwargs = dict(
                    C=array,
                    axis_rgb=axis_rgb,
                    components=components,
                )

                if isinstance(array.outputs, na.AbstractVectorArray):
                    with pytest.raises(TypeError):
                        na.plt.pcolormesh(**kwargs)
                    return
                elif isinstance(array.outputs, na.AbstractUncertainScalarArray):
                    with pytest.raises(TypeError):
                        na.plt.pcolormesh(**kwargs)
                    return

                if axis_rgb is not None:
                    with pytest.raises(ValueError):
                        na.plt.pcolormesh(**kwargs)
                    return

                result = na.plt.pcolormesh(**kwargs)
                assert isinstance(result, na.ScalarArray)

        @pytest.mark.xfail
        class TestJacobian(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestJacobian,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeRoot(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeRoot,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeMinimum(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeMinimum,
        ):
            pass

        class TestColorsynth(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestColorsynth,
        ):

            def test_colorbar(self, array: na.AbstractFunctionArray, axis: None | str):
                if isinstance(array.outputs, na.AbstractVectorArray):
                    return
                super().test_colorbar(array=array, axis=axis)


@pytest.mark.parametrize("array", _function_arrays())
class TestFunctionArray(
    AbstractTestAbstractFunctionArray,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray,
):

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(x=0, y=0),
            dict(y=slice(None)),
            dict(y=na.ScalarArrayRange(0, _num_y, axis='y')),
            dict(x=na.ScalarArrayRange(0, _num_x, axis='x'), y=na.ScalarArrayRange(0, _num_y, axis='y')),
            na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                outputs=na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
            )
        ]
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                outputs=na.ScalarUniformRandomSample(-5, 5, dict(y=_num_y)),
            )
        ]
    )
    def test__setitem__(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.ScalarArray] | na.AbstractFunctionArray,
            value: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ):
        super().test__setitem__(array=array.explicit, item=item, value=value)


@pytest.mark.parametrize("type_array", [na.FunctionArray])
class TestFunctionArrayCreation(
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation,
):
    @pytest.mark.parametrize("like", [None] + _function_arrays())
    class TestFromScalarArray(
        named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation.TestFromScalarArray,
    ):
        pass
