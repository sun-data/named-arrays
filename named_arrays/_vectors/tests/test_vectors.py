from typing import Type, Callable, Sequence
import pytest
import abc
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

class AbstractTestAbstractVectorArray(
    named_arrays.tests.test_core.AbstractTestAbstractArray,
):

    def test_components(self, array: na.AbstractVectorArray):
        components = array.components
        assert isinstance(components, dict)
        for component in components:
            assert isinstance(component, str)
            assert isinstance(components[component], (int, float, complex, np.ndarray, na.AbstractArray))

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
            assert array_new.entries[e].dtype == dtype

    @pytest.mark.parametrize('unit', [u.mm, u.s])
    def test_to(self, array: na.AbstractVectorArray, unit: None | u.UnitBase):
        super().test_to(array=array, unit=unit)
        entries = array.entries
        if all(unit.is_equivalent(na.unit_normalized(entries[e])) for e in entries):
            array_new = array.to(unit)
            assert array_new.type_array_abstract == array.type_array_abstract
            assert all(array_new.entries[e].unit == unit for e in array_new.entries)
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test_length(self, array: na.AbstractVectorArray):
        super().test_length(array=array)
        entries = array.entries
        entries_iter = iter(entries)
        entry_0 = entries[next(entries_iter)]
        if all(na.unit_normalized(entry_0).is_equivalent(na.unit_normalized(entries[e])) for e in entries_iter):
            length = array.length
            assert isinstance(length, (int, float, np.ndarray, na.AbstractScalar))
            assert np.all(length >= 0)
        else:
            with pytest.raises(u.UnitConversionError):
                array.length

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
                if isinstance(item[ax], na.AbstractArray) and item[ax].type_array_abstract == array.type_array_abstract:
                    components_item_ax = item[ax].components
                else:
                    components_item_ax = array.type_array.from_scalar(item[ax]).components
                for c in components:
                    components_item[c][ax] = components_item_ax[c]

        else:
            if not item.type_array_abstract == array.type_array_abstract:
                components_item = array.type_array.from_scalar(item).components
            else:
                components_item = item.components
                item_accumulated = True
                for c in components_item:
                    item_accumulated = item_accumulated & components_item[c]
                components_item = item.type_array.from_scalar(item_accumulated).components

        for c in components:
            components_expected[c] = na.as_named_array(components[c])[components_item[c]]

        result_expected = array.type_array.from_components(components_expected)

        result = array[item]

        assert isinstance(result.shape, dict)
        assert np.all(result == result_expected)

    def test__bool__(self, array: na.AbstractVectorArray):
        if array.shape or any(na.unit(array.entries[e]) is not None for e in array.entries):
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
                    components_1 = array.components
                    components_2 = array_2.components
                    result_expected = 0
                    for c in components_1:
                        result_expected = result_expected + components_1[c] * components_2[c]
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
                    result_expected = array.type_array()
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

                assert np.all(result == result_expected)
                assert np.all(result == result_out)
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
                    result_expected = array.type_array()
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

                result_expected = array.type_array()
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

                result_expected = array.type_array()
                for c in array.components:
                    result_expected.components[c] = func(
                        array.broadcasted.components[c],
                        axes=axes,
                        s=s,
                    )

                assert np.all(result == result_expected)

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
                result_expected = array.type_array()
                for c in components_broadcasted:
                    result_expected.components[c] = np.sort(components_broadcasted[c], axis=axis_normalized)
            else:
                result_expected = array

            assert np.all(result == result_expected)

        @pytest.mark.parametrize('copy', [False, True])
        def test_nan_to_num(self, array: na.AbstractVectorArray, copy: bool):
            components = array.components

            components_expected = {c: np.nan_to_num(components[c], copy=copy) for c in components}
            result_expected = array.type_array.from_components(components_expected)

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
        array_broadcasted = array.broadcasted
        shape = array.shape
        components = array_broadcasted.components
        for component in components:
            assert components[component].shape == shape

    def test_ptp(
            self,
            array: na.AbstractVectorArray,
    ):
        super().test_ptp(array=array)
        if any(np.issubdtype(na.get_dtype(array.entries[e]), bool) for e in array.entries):
            with pytest.raises(TypeError, match='numpy boolean subtract, .*'):
                array.ptp()
            return

        assert np.all(array.ptp() == np.ptp(array))

    def test_all(
            self,
            array: na.AbstractVectorArray,
    ):
        super().test_all(array=array)
        entries = array.entries
        if any(na.unit(entries[e]) is not None for e in entries):
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.all()
            return

        assert np.all(array.all() == np.all(array))

    def test_any(
            self,
            array: na.AbstractVectorArray,
    ):
        super().test_any(array=array)
        entries = array.entries
        if any(na.unit(entries[e]) is not None for e in entries):
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.any()
            return

        assert np.all(array.any() == np.any(array))


class AbstractTestAbstractExplicitVectorArray(
    AbstractTestAbstractVectorArray,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray,
):
    pass


class AbstractTestAbstractExplicitVectorArrayCreation(
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation,
):
    @property
    @abc.abstractmethod
    def type_array(self) -> Type[na.AbstractExplicitVectorArray]:
        pass

    def test_empty(self, shape: dict[str, int], dtype: Type):
        super().test_empty(shape=shape, dtype=dtype)
        result = self.type_array.empty(shape, dtype=dtype)
        for c in result.components:
            assert result.components[c].dtype == dtype

    def test_zeros(self, shape: dict[str, int], dtype: Type):
        super().test_zeros(shape=shape, dtype=dtype)
        result = self.type_array.zeros(shape, dtype=dtype)
        for c in result.components:
            assert result.components[c].dtype == dtype

    def test_ones(self, shape: dict[str, int], dtype: Type):
        super().test_ones(shape=shape, dtype=dtype)
        result = self.type_array.ones(shape, dtype=dtype)
        for c in result.components:
            assert result.components[c].dtype == dtype


class AbstractTestAbstractImplicitVectorArray(
    AbstractTestAbstractVectorArray,
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
