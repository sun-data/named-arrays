from typing import Mapping, Type, Callable, Sequence, Literal
import pytest
import numpy as np
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
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

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
            if not array.shape:
                with pytest.raises(ValueError):
                    array[item]
                return

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

            out = na.asanyarray(0 * result)
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

        class TestCumulativeReductionFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestCumulativeReductionFunctions,
        ):

            def test_cumulative_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractVectorArray,
                    axis: None | str | Sequence[str],
                    dtype: Type,
            ):
                super().test_cumulative_reduction_functions(
                    func=func,
                    array=array,
                    axis=axis,
                    dtype=dtype,
                )

                shape = array.shape
                components = array.components

                if not shape:
                    return

                kwargs = dict(
                    axis=axis,
                )

                if dtype is not np._NoValue:
                    kwargs["dtype"] = dtype

                kwargs_components = {c: kwargs for c in components}

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
                    with pytest.raises(ValueError, match="if `axis` is a sequence, it must not be empty, got .*"):
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

            if not copy and isinstance(array, na.AbstractImplicitArray):
                with pytest.raises(ValueError, match=r"can\'t write to an array that is not an instance of .*"):
                    np.nan_to_num(array, copy=copy)
                return

            try:
                components_expected = {c: np.nan_to_num(components[c], copy=copy) for c in components}
                result_expected = array.type_explicit.from_components(components_expected)
            except ValueError as e:
                match = "Unable to avoid copy"
                if e.args[0].startswith(match):
                    with pytest.raises(ValueError, match=match):
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

        def test_nominal(self, array: na.AbstractVectorArray):
            result = na.nominal(array)

            components = array.components

            for c in components:
                assert np.all(result.components[c] == na.nominal(components[c]))

        @pytest.mark.skip
        class TestInterp(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestInterp
        ):
            pass

        @pytest.mark.parametrize(
            argnames="bins",
            argvalues=[
                "dict",
                "array",
            ],
        )
        class TestHistogram(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestHistogram,
        ):
            def test_histogram(
                    self,
                    array: na.AbstractVectorArray,
                    bins: Literal["dict", "array"],
                    axis: None | str | Sequence[str],
                    min: None | na.AbstractScalarArray | na.AbstractVectorArray,
                    max: None | na.AbstractScalarArray | na.AbstractVectorArray,
                    weights: None | na.AbstractScalarArray,
            ):
                if bins == "dict":
                    bins = {f"axis_{c}": 2 for c in array.cartesian_nd.components}
                elif bins == "array":
                    bins = {
                        c: na.linspace(0, 1, f"axis_{c}", 2)
                        for c in array.cartesian_nd.components
                    }
                    bins = na.CartesianNdVectorArray(bins)
                    bins = array.type_explicit.from_cartesian_nd(bins, like=array)
                super().test_histogram(
                    array=array,
                    bins=bins,
                    axis=axis,
                    min=min,
                    max=max,
                    weights=weights,
                )

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
                (
                    lambda x: (np.square((na.value(x) - shift_horizontal).length) + shift_vertical) * u.ph,
                    shift_horizontal,
                )
                for shift_horizontal in [2,]
                for shift_vertical in [1,]
            ]
        )
        class TestOptimizeMinimum(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeMinimum,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeMinimumBrent(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeMinimumBrent,
        ):
            pass

        class TestColorsynth(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestColorsynth,
        ):

            @pytest.mark.skip
            def test_colorbar(
                self,
                array: na.AbstractArray,
                wavelength: None | na.AbstractScalar,
                axis: None | str,
            ):
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

    def test_shape_explicit(self, array: na.AbstractWcsVector) -> None:
        explicit = array.explicit
        assert array.shape == explicit.shape
        assert array.axes == explicit.axes
        assert array.ndim == explicit.ndim
        assert array.size == explicit.size

    @pytest.mark.parametrize(
        argnames="item,lazy",
        argvalues=[
            (dict(x=slice(1, None)), True),
            (dict(x=slice(None, -1), y=slice(1, 3)), True),
            (dict(y=slice(-2, None, 1)), True),
            (dict(z=0), True),
            # fewer than two pixels are computed explicitly
            (dict(x=slice(2, 1)), False),
            (dict(x=slice(1, 2)), False),
            (dict(x=0), False),
            (dict(y=slice(None, None, 2)), False),
            (dict(x=slice(None, None, -1)), False),
            # an index array along another axis, which has a WCS axis
            (dict(z=na.ScalarArray(np.array([0, 1]), axes="y")), False),
            (dict(z=None), False),
        ],
    )
    def test__getitem__wcs(
        self,
        array: na.AbstractWcsVector,
        item: dict[str, int | slice],
        lazy: bool,
    ) -> None:
        result = array[item]
        expected = array.explicit[item]
        if lazy:
            assert type(result) is type(array)
            assert result.shape == result.explicit.shape
        else:
            assert isinstance(result, na.AbstractExplicitVectorArray)
        assert result.shape == expected.shape
        assert np.all(result.explicit == expected)


def _vectors_single_element() -> list[na.AbstractVectorArray]:
    """
    Vectors and a matrix with a component which has a single element along
    an axis where the other components have many.
    """
    x = na.arange(0, 10, axis="x")
    single = na.ScalarArray(np.array([2]), axes="x")
    return [
        na.Cartesian2dVectorArray(x=x, y=single),
        na.Cartesian2dVectorArray(x=single, y=x + na.arange(0, 3, axis="y")),
        na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(x=x, y=single),
            y=na.Cartesian2dVectorArray(x=single, y=1),
        ),
    ]


@pytest.mark.parametrize("array", _vectors_single_element())
@pytest.mark.parametrize(
    argnames="item",
    argvalues=[
        dict(x=slice(2, 5)),
        dict(x=5),
        dict(x=-1),
        dict(x=slice(None, None, -3)),
        dict(x=slice(5, 2)),
        dict(x=na.ScalarArray(np.array([7, 2]), axes="x")),
        dict(x=na.ScalarArray(np.array([7, 2]), axes="z")),
        dict(x=slice(2, 5), y=1),
        na.arange(0, 10, axis="x") > 4,
    ],
)
def test__getitem__single_element(
    array: na.AbstractVectorArray,
    item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
) -> None:
    """
    A component with a single element along an indexed axis, where the other
    components have more, is indexed as if it were broadcast against them.
    """
    result = array[item]
    expected = array.broadcasted[item]
    assert result.shape == expected.shape
    assert np.all(result == expected)


@pytest.mark.parametrize(
    argnames="item,writeable",
    argvalues=[
        (dict(x=slice(None)), True),
        (dict(x=slice(None, None, -1)), True),
        (dict(x=slice(2, 5)), False),
        (dict(x=slice(0, 1)), False),
        (dict(x=3), False),
    ],
)
def test__getitem__single_element_view(
    item: dict[str, int | slice],
    writeable: bool,
) -> None:
    """
    A component with a single element along an indexed axis keeps it as a
    view, which can only be written to if the selection includes every
    element along the axis, since writing to it changes all of them.
    """
    y = na.ScalarArray(np.array([[2.0, 3.0]]), axes=("x", "z"))
    array = na.Cartesian2dVectorArray(x=na.arange(0, 10, axis="x"), y=y)
    result = array[item]
    assert np.shares_memory(result.y.ndarray, y.ndarray)
    assert result.y.ndarray.flags.writeable == writeable


def _wcs_channels(
    crpix: float,
    single: bool = False,
) -> na.ExplicitTemporalWcsPositionalVectorArray:
    """
    A WCS vector whose parameters vary along a `channel` axis.

    If `single`, the time, a component of `crval`, and `cdelt` have a single
    element along `channel` instead, which is broadcast against the others.
    """
    channel = na.ScalarArray(np.arange(3), axes="channel")
    time = channel
    crval_y = -channel
    cdelt_x = 1 + channel
    if single:
        time = crval_y = cdelt_x = na.ScalarArray(np.array([2]), axes="channel")
    return na.ExplicitTemporalWcsPositionalVectorArray(
        time=time * u.s,
        crval=na.PositionalVectorArray(
            position=na.Cartesian2dVectorArray(channel, crval_y) * u.arcsec,
        ),
        crpix=na.CartesianNdVectorArray(dict(x=crpix + 0 * channel, y=crpix + channel)),
        cdelt=na.PositionalVectorArray(
            position=na.Cartesian2dVectorArray(cdelt_x, 1) * u.arcsec,
        ),
        pc=na.PositionalMatrixArray(
            position=na.Cartesian2dMatrixArray(
                x=na.CartesianNdVectorArray(dict(x=1, y=0.125 * channel)),
                y=na.CartesianNdVectorArray(dict(x=-0.125 * channel, y=1)),
            ),
        ),
        shape_wcs=dict(x=9, y=10),
    )


@pytest.mark.parametrize(
    argnames="item,lazy",
    argvalues=[
        (dict(x=slice(4, 8)), True),
        (dict(channel=1, x=slice(4, 8)), True),
        (dict(channel=slice(1, None), y=slice(-5, None)), True),
        (dict(channel=na.ScalarArray(np.array([0, 2]), axes="channel"), x=slice(2, 6)), True),
        (dict(channel=-1, y=slice(2, 6)), True),
        (dict(x=slice(4, 5)), False),
        (dict(x=slice(4, 4)), False),
    ],
)
@pytest.mark.parametrize("crpix", [4, 3.5])
@pytest.mark.parametrize("single", [False, True])
def test__getitem__wcs_channels(
    item: dict[str, int | slice | na.AbstractArray],
    lazy: bool,
    crpix: float,
    single: bool,
) -> None:
    """
    Indexing a WCS vector along the axis of its parameters and along its WCS
    axes gives the same vector as indexing the explicit vector, including
    when a slice starts half a pixel past `crpix`, where the coordinates of
    the first pixel along that axis are zero, and when some parameters have
    a single element along `channel`.
    """
    array = _wcs_channels(crpix, single)
    result = array[item]
    expected = array.explicit[item]
    if lazy:
        assert type(result) is type(array)
        assert result.shape == result.explicit.shape
    else:
        assert isinstance(result, na.AbstractExplicitVectorArray)
    assert result.shape == expected.shape
    assert np.all(result.explicit == expected)
    assert np.all(result.explicit == array.broadcasted[item])


@pytest.mark.parametrize("num_crval", [1, 5])
def test__getitem__wcs_parameter_along_wcs_axis(num_crval: int) -> None:
    """
    A WCS parameter which varies along a sliced WCS axis, even one with a
    single element broadcast against the pixels, is indexed through the
    explicit vector.
    """
    x = na.ScalarArray(np.arange(num_crval) * u.arcsec, axes="x")
    array = na.ExplicitTemporalWcsPositionalVectorArray(
        time=10 * u.s,
        crval=na.PositionalVectorArray(
            position=na.Cartesian2dVectorArray(x, 1 * u.arcsec),
        ),
        crpix=na.CartesianNdVectorArray(dict(x=2, y=3)),
        cdelt=na.PositionalVectorArray(
            position=na.Cartesian2dVectorArray(1, 1) * u.arcsec,
        ),
        pc=na.PositionalMatrixArray(
            position=na.Cartesian2dMatrixArray(
                x=na.CartesianNdVectorArray(dict(x=1, y=0)),
                y=na.CartesianNdVectorArray(dict(x=0, y=1)),
            ),
        ),
        shape_wcs=dict(x=5, y=4),
    )
    assert array.shape == array.explicit.shape
    item = dict(x=slice(1, 3))
    result = array[item]
    expected = array.explicit[item]
    assert isinstance(result, na.AbstractExplicitVectorArray)
    assert result.shape == expected.shape == dict(x=2, y=4)
    assert np.all(result == expected)


def test_searchsorted_sorter_per_component():
    """
    Each component is searched through its own permutation, so a sorter is
    taken apart component by component like every other argument.
    """
    a = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([3.0, 1.0, 2.0, 0.0]) * u.mm, axes=("w",)),
        y=na.ScalarArray(np.array([30.0, 10.0, 20.0, 0.0]) * u.mm, axes=("w",)),
    )
    sorter = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.argsort(a.x.ndarray), axes=("w",)),
        y=na.ScalarArray(np.argsort(a.y.ndarray), axes=("w",)),
    )
    v = na.ScalarArray(
        ndarray=np.array([0.5, 1.5, 2.5]) * u.mm,
        axes=("line",),
    )

    result = na.searchsorted(a, v, axis="w", sorter=sorter)

    assert isinstance(result, na.AbstractCartesian2dVectorArray)

    for c in ("x", "y"):
        expected = np.searchsorted(
            a.components[c].ndarray,
            v.ndarray,
            sorter=sorter.components[c].ndarray,
        )
        assert np.all(result.components[c].ndarray == expected)

    # the two components are spread differently, so this really is per
    # component and not one answer copied across both
    assert np.any(result.x.ndarray != result.y.ndarray)


def _grid_shared() -> na.ScalarArray:
    """One grid, of no particular component, for the mixed-type tests."""
    return na.ScalarArray(
        ndarray=np.array([0.0, 1.0, 2.0, 3.0]) * u.mm,
        axes=("w",),
    )


def _values_2d() -> na.Cartesian2dVectorArray:
    return na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([1.5]) * u.mm, axes=("line",)),
        y=na.ScalarArray(np.array([2.5]) * u.mm, axes=("line",)),
    )


def test_searchsorted_scalar_grid_and_vector_values():
    """
    A grid shared by every component, which is the scalar implementation
    declining the vector values and this one taking them.
    """
    result = na.searchsorted(_grid_shared(), _values_2d(), axis="w")

    assert np.all(result.x.ndarray == 2)
    assert np.all(result.y.ndarray == 3)


def test_digitize_scalar_bins_and_vector_values():
    """The same sharing, for bins rather than a grid."""
    result = na.digitize(_values_2d(), _grid_shared(), axis="w")

    assert np.all(result.x.ndarray == 2)
    assert np.all(result.y.ndarray == 3)


def test_searchsorted_mixed_vector_types():
    """
    Two different kinds of vector have no components in common, so neither
    one's implementation can take the pair and the dispatch runs out.
    """
    a = na.Cartesian2dVectorArray(
        x=_grid_shared(),
        y=_grid_shared(),
    )
    v = na.Cartesian3dVectorArray(x=1 * u.mm, y=1 * u.mm, z=1 * u.mm)

    with pytest.raises(TypeError, match="returned `NotImplemented`"):
        na.searchsorted(a, v, axis="w")


def test_digitize_mixed_vector_types():
    """The same, for bins rather than a grid."""
    bins = na.Cartesian2dVectorArray(
        x=_grid_shared(),
        y=_grid_shared(),
    )
    x = na.Cartesian3dVectorArray(x=1 * u.mm, y=1 * u.mm, z=1 * u.mm)

    with pytest.raises(TypeError, match="returned `NotImplemented`"):
        na.digitize(x, bins, axis="w")
