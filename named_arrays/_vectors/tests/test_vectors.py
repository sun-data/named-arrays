from typing import Type, Callable, Sequence
import pytest
import abc
import numpy as np
import matplotlib.axes
import astropy.units as u
import named_arrays as na

import named_arrays.tests

__all__ = [
    'AbstractTestAbstractVectorArray',
    'AbstractTestAbstractExplicitVectorArray',
    'AbstractTestAbstractExplicitVectorArrayCreation',
    'AbstractTestAbstractImplicitVectorArray',
    'AbstractTestAbstractVectorRandomSample',
    'AbstractTestAbstractVectorUniformRandomSample',
    'AbstractTestAbstractVectorNormalRandomSample',
    'AbstractTestAbstractParameterizedVectorArray',
    'AbstractTestAbstractVectorArrayRange',
    'AbstractTestAbstractVectorSpace',
    'AbstractTestAbstractVectorLinearSpace',
    'AbstractTestAbstractVectorStratifiedRandomSpace',
    'AbstractTestAbstractVectorLogarithmicSpace',
    'AbstractTestAbstractVectorGeometricSpace',
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_z = named_arrays.tests.test_core.num_z
_num_distribution = named_arrays.tests.test_core.num_distribution


class AbstractTestAbstractVectorArray(
    named_arrays.tests.test_core.AbstractTestAbstractArray,
):
    def test_cartesian_nd(self, array: na.AbstractVectorArray):
        cartesian_nd = array.cartesian_nd
        assert isinstance(cartesian_nd, na.AbstractCartesianNdVectorArray)
        for c in cartesian_nd.components:
            assert isinstance(na.as_named_array(cartesian_nd.components[c]), na.AbstractScalar)

    def test_from_cartesian_nd(self, array: na.AbstractVectorArray):
        assert np.all(array.type_explicit.from_cartesian_nd(array.cartesian_nd, like=array) == array)

    def test_matrix(self, array: na.AbstractVectorArray):
        def _recursive_test(array: na.AbstractMatrixArray):
            assert isinstance(array, na.AbstractMatrixArray)
            components = array.components
            for c in components:
                component = components[c]
                if isinstance(component, na.AbstractVectorArray):
                    _recursive_test(component)
                else:
                    assert isinstance(na.as_named_array(component), na.AbstractScalar)
        _recursive_test(array.matrix)

    def test_components(self, array: na.AbstractVectorArray):
        components = array.components
        assert isinstance(components, dict)
        for component in components:
            assert isinstance(component, str)
            assert isinstance(components[component], (int, float, complex, np.generic, np.ndarray, na.AbstractArray))

    def test_axes(self, array: na.AbstractVectorArray):
        super().test_axes(array)
        components = array.broadcasted.components
        for c in components:
            assert array.axes == components[c].axes

    @pytest.mark.parametrize('dtype', [int, float])
    def test_astype(self, array: na.AbstractVectorArray, dtype: Type):
        super().test_astype(array=array, dtype=dtype)
        array_new = array.astype(dtype)
        for e in array_new.entries:
            entry = array_new.entries[e]
            assert entry.dtype == dtype

    @pytest.mark.parametrize('unit', [u.mm, u.s])
    def test_to(self, array: na.AbstractVectorArray, unit: None | u.UnitBase):
        super().test_to(array=array, unit=unit)
        entries = array.cartesian_nd.entries
        if all(unit.is_equivalent(na.unit_normalized(entries[e])) for e in entries):
            array_new = array.to(unit)
            assert array_new.type_abstract == array.type_abstract
            assert all(array_new.cartesian_nd.entries[e].unit == unit for e in array_new.cartesian_nd.entries)
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test_length(self, array: na.AbstractVectorArray):
        super().test_length(array=array)
        entries = array.cartesian_nd.entries
        try:
            sum(entries.values())
        except u.UnitConversionError:
            with pytest.raises(u.UnitConversionError):
                array.length
            return

        length = array.length
        assert isinstance(length, (int, float, np.ndarray, na.AbstractScalar))
        assert np.all(length >= 0)

    def test__getitem__(
            self,
            array: na.AbstractVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

        if not array.shape:
            with pytest.raises(ValueError):
                array[item]
            return

        components = array.broadcasted.components
        components_expected = dict()

        if isinstance(item, dict):
            components_item = {c: dict() for c in components}
            for ax in item:
                if isinstance(item[ax], na.AbstractArray) and item[ax].type_abstract == array.type_abstract:
                    components_item_ax = item[ax].components
                else:
                    components_item_ax = array.type_explicit.from_scalar(item[ax], like=array).components
                for c in components:
                    components_item[c][ax] = components_item_ax[c]

        else:
            if not item.type_abstract == array.type_abstract:
                components_item = array.type_explicit.from_scalar(item, like=array).components
            else:
                components_item = item.components
                item_accumulated = True
                for c in components_item:
                    item_accumulated = item_accumulated & components_item[c]
                components_item = item.type_explicit.from_scalar(item_accumulated, like=array).components

        for c in components:
            components_expected[c] = na.as_named_array(components[c])[components_item[c]]

        result_expected = array.type_explicit.from_components(components_expected)

        result = array[item]

        assert isinstance(result.shape, dict)
        assert np.all(result == result_expected)

    def test__bool__(self, array: na.AbstractVectorArray):
        if array.shape or any(na.unit(array.cartesian_nd.entries[e]) is not None for e in array.cartesian_nd.entries):
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
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestMatmul
    ):

        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
        ):

            try:
                if isinstance(array, na.AbstractVectorArray) and isinstance(array_2, na.AbstractVectorArray):
                    components_1 = array.cartesian_nd.components
                    components_2 = array_2.cartesian_nd.components
                    if np.all(components_2.keys() == components_1.keys()):
                        result_expected = 0
                        for c in components_1:
                            result_expected = result_expected + components_1[c] * components_2[c]
                    else:
                        raise TypeError
                else:
                    result_expected = np.multiply(array, array_2)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    np.matmul(array, array_2)
                return

            result = np.matmul(array, array_2)

            out = 0 * result
            result_out = np.matmul(array, array_2, out=out)

            assert np.all(result == result_expected)
            assert np.all(result == result_out)
            assert result_out is out

    class TestArrayFunctions(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions,
    ):
        class TestAsArrayLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestAsArrayLikeFunctions,
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

                if isinstance(a, na.AbstractVectorArray):
                    if isinstance(like, na.AbstractVectorArray):
                        if a.type_explicit != like.type_explicit:
                            with pytest.raises(
                                    expected_exception=TypeError,
                                    match="all types, .*, returned .* for function .*",
                            ):
                                func(a, like=like)
                            return

                result = func(a, like=like)

                assert isinstance(result, na.AbstractExplicitVectorArray)
                for c in result.components:
                    assert isinstance(result.components[c], (na.AbstractScalar, na.AbstractVectorArray))

                super().test_asarray_like_functions(
                    func=func,
                    array=array,
                    array_2=array_2,
                )

        class TestSingleArgumentFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestSingleArgumentFunctions,
        ):
            def test_single_argument_functions(
                self,
                func: Callable,
                array: na.AbstractVectorArray,
            ):
                result = func(array)
                for c in array.components:
                    assert np.all(result.components[c] == func(array.components[c]))

        class TestReductionFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestReductionFunctions,
        ):

            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractVectorArray,
                    axis: None | str | Sequence[str],
                    dtype: Type,
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

                shape = na.shape_broadcasted(array, where)
                components = array.components

                kwargs = dict(
                    axis=axis,
                    keepdims=keepdims,
                    where=where,
                )

                if dtype is not np._NoValue:
                    kwargs["dtype"] = dtype

                if func in [np.min, np.nanmin, np.max, np.nanmax]:
                    kwargs["initial"] = 0

                kwargs_components = {c: dict() for c in components}
                for c in components:
                    for k in kwargs:
                        if isinstance(kwargs[k], na.AbstractVectorArray):
                            kwargs_components[c][k] = kwargs[k].components[c]
                        else:
                            kwargs_components[c][k] = kwargs[k]

                        if isinstance(kwargs_components[c][k], na.AbstractArray):
                            kwargs_components[c][k] = kwargs_components[c][k].broadcast_to(shape)

                try:
                    result_expected = array.prototype_vector
                    for c in components:
                        component = na.as_named_array(array.components[c]).broadcast_to(shape)
                        result_expected.components[c] = func(component, **kwargs_components[c])
                except (ValueError, TypeError, u.UnitsError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result

                result_out = func(array, out=out, **kwargs)

                assert np.allclose(result, result_expected)
                assert np.allclose(result, result_out)
                assert result_out is out

        class TestPercentileLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestPercentileLikeFunctions,
        ):

            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractVectorArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    keepdims: bool,
            ):
                super().test_percentile_like_functions(
                    func=func,
                    array=array.explicit,
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                shape = array.shape
                components = array.components
                components_q = q.components if isinstance(q, na.AbstractVectorArray) else {c: q for c in components}

                kwargs = dict(
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                kwargs_components = dict()
                for c in components:
                    kwargs_components[c] = dict(
                        q=components_q[c],
                        axis=axis,
                        keepdims=keepdims,
                    )

                try:
                    result_expected = array.prototype_vector
                    for c in components:
                        component = na.as_named_array(array.components[c]).broadcast_to(shape)
                        result_expected.components[c] = func(component, **kwargs_components[c])
                except (ValueError, TypeError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result

                result_out = func(array, out=out, **kwargs)

                assert np.all(result == result_expected)
                assert np.all(result == result_out)
                assert result_out is out

        class TestFFTLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTLikeFunctions,
        ):

            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractVectorArray,
                    axis: tuple[str, str],
            ):
                super().test_fft_like_functions(
                    func=func,
                    array=array,
                    axis=axis,
                )

                if axis[0] not in array.shape:
                    with pytest.raises(ValueError, match="`axis` .* not in array with shape .*"):
                        func(array, axis=axis)
                    return

                result = func(array, axis=axis)

                result_expected = array.prototype_vector
                for c in array.components:
                    result_expected.components[c] = func(array.broadcasted.components[c], axis=axis)

                assert np.all(result == result_expected)

        class TestFFTNLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTNLikeFunctions,
        ):

            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractVectorArray,
                    axes: dict[str, str],
                    s: None | dict[str, int],
            ):
                super().test_fftn_like_functions(
                    func=func,
                    array=array,
                    axes=axes,
                    s=s,
                )

                if not set(axes).issubset(array.shape):
                    with pytest.raises(ValueError, match="`axes`, .*, not a subset of array axes, .*"):
                        func(array, axes=axes, s=s)
                    return

                if s is not None and axes.keys() != s.keys():
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                result = func(array, axes=axes, s=s)

                result_expected = array.prototype_vector
                for c in array.components:
                    result_expected.components[c] = func(
                        array.broadcasted.components[c],
                        axes=axes,
                        s=s,
                    )

                assert np.all(result == result_expected)

        class TestEmathFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestEmathFunctions,
        ):
            def test_emath_functions(
                self,
                func: Callable,
                array: na.AbstractVectorArray,
            ):
                result = func(array)
                for c in array.components:
                    assert np.all(result.components[c] == func(array.components[c]))

        @pytest.mark.parametrize('axis', [None, 'x', 'y', ('x', 'y'), ()])
        def test_sort(self, array: na.AbstractVectorArray, axis: None | str | Sequence[str]):
            super().test_sort(array=array, axis=axis)

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
            components_broadcasted = array_broadcasted.components

            if axis_normalized:
                result_expected = array.prototype_vector
                for c in components_broadcasted:
                    result_expected.components[c] = np.sort(components_broadcasted[c], axis=axis_normalized)
            else:
                result_expected = array

            assert np.all(result == result_expected)

        @pytest.mark.parametrize('copy', [False, True])
        def test_nan_to_num(self, array: na.AbstractVectorArray, copy: bool):
            components = array.components

            components_expected = {c: np.nan_to_num(components[c], copy=copy) for c in components}
            result_expected = array.type_explicit.from_components(components_expected)

            if not copy and isinstance(array, na.AbstractImplicitArray):
                with pytest.raises(ValueError, match=r"can\'t write to an array that is not an instance of .*"):
                    np.nan_to_num(array, copy=copy)
                return

            result = np.nan_to_num(array, copy=copy)

            assert np.all(result == result_expected)

        @pytest.mark.xfail
        def test_convolve(self, array: na.AbstractVectorArray, v: na.AbstractScalarOrVectorArray, mode: str):
            np.convolve(array, v=v, mode=mode)

    def test_broadcasted(self, array: na.AbstractVectorArray):
        super().test_broadcasted(array=array)
        array_broadcasted = array.broadcasted
        shape = array.shape
        components = array_broadcasted.components
        for component in components:
            assert components[component].shape == shape

    class TestNamedArrayFunctions(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions
    ):
        @pytest.mark.skip
        class TestInterp(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestInterp
        ):
            pass

        @pytest.mark.parametrize("array_2", [None])
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
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions
        ):
            pass

        @pytest.mark.skip
        class TestPltScatter(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestPltScatter,
        ):
            pass

        @pytest.mark.skip
        class TestPltPcolormesh(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestPltPcolormesh,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="function",
            argvalues=[
                lambda x: 2 * x ** 3,
                lambda x: 2 * list(x.components.values())[0] ** 3,
            ]
        )
        class TestJacobian(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestJacobian,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                na.optimize.root_newton,
            ],
        )
        @pytest.mark.parametrize(
            argnames="function",
            argvalues=[
                lambda x: np.square(na.value(x) - shift_horizontal) + shift_vertical
                for shift_horizontal in [20,]
                for shift_vertical in [-1]
            ],
        )
        class TestOptimizeRoot(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeRoot,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                na.optimize.minimum_gradient_descent,
            ],
        )
        @pytest.mark.parametrize(
            argnames="function,expected",
            argvalues=[
                (lambda x: (np.square(na.value(x) - shift_horizontal) + shift_vertical).length, shift_horizontal)
                for shift_horizontal in [20,]
                for shift_vertical in [1,]
            ]
        )
        class TestOptimizeMinimum(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeMinimum,
        ):
            pass

        class TestColorsynth(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestColorsynth,
        ):

            @pytest.mark.skip
            def test_colorbar(self, array: na.AbstractArray, axis: None | str):
                pass    # pragma: nocover


class AbstractTestAbstractExplicitVectorArray(
    AbstractTestAbstractVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray,
):
    pass


class AbstractTestAbstractExplicitVectorArrayCreation(
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation
):
    pass


class AbstractTestAbstractImplicitVectorArray(
    named_arrays.tests.test_core.AbstractTestAbstractImplicitArray,
):
    pass


class AbstractTestAbstractVectorRandomSample(
    AbstractTestAbstractImplicitVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractRandomSample,
):
    pass


class AbstractTestAbstractVectorUniformRandomSample(
    AbstractTestAbstractVectorRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractUniformRandomSample,
):
    pass


class AbstractTestAbstractVectorNormalRandomSample(
    AbstractTestAbstractVectorRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedVectorArray(
    AbstractTestAbstractImplicitVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractParameterizedArray,
):
    pass


class AbstractTestAbstractVectorArrayRange(
    AbstractTestAbstractParameterizedVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractArrayRange,
):
    pass


class AbstractTestAbstractVectorSpace(
    AbstractTestAbstractParameterizedVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractSpace,
):
    pass


class AbstractTestAbstractVectorLinearSpace(
    AbstractTestAbstractVectorSpace,
    named_arrays.tests.test_core.AbstractTestAbstractLinearSpace,
):
    pass


class AbstractTestAbstractVectorStratifiedRandomSpace(
    AbstractTestAbstractVectorLinearSpace,
    named_arrays.tests.test_core.AbstractTestAbstractStratifiedRandomSpace,
):
    pass


class AbstractTestAbstractVectorLogarithmicSpace(
    AbstractTestAbstractVectorSpace,
    named_arrays.tests.test_core.AbstractTestAbstractLogarithmicSpace,
):
    pass


class AbstractTestAbstractVectorGeometricSpace(
    AbstractTestAbstractVectorSpace,
    named_arrays.tests.test_core.AbstractTestAbstractGeometricSpace,
):
    pass


class AbstractTestAbstractWcsVector(
    AbstractTestAbstractImplicitVectorArray,
):
    def test_crval(self, array: na.AbstractWcsVector):
        result = array.crval
        assert isinstance(result, na.AbstractVectorArray)

    def test_crpix(self, array: na.AbstractWcsVector):
        result = array.crpix
        assert isinstance(result, na.AbstractCartesianNdVectorArray)

    def test_cdelt(self, array: na.AbstractWcsVector):
        result = array.cdelt
        assert isinstance(result, na.AbstractVectorArray)

    def test_pc(self, array: na.AbstractWcsVector):
        result = array.pc
        assert isinstance(result, na.AbstractMatrixArray)
        components = result.components
        for c in components:
            assert isinstance(components[c], na.AbstractVectorArray)

    def test_shape_wcs(self, array: na.AbstractWcsVector):
        result = array.shape_wcs
        assert isinstance(result, dict)
        for k in result:
            assert isinstance(k, str)
            assert isinstance(result[k], int)
