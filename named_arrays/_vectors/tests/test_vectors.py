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
            assert len(array.axes) == np.ndim(components[c].ndarray)

    @pytest.mark.parametrize('dtype', [int, float])
    def test_astype(self, array: na.AbstractVectorArray, dtype: Type):
        super().test_astype(array=array, dtype=dtype)
        array_new = array.astype(dtype)
        for c in array_new.components:
            assert array_new.components[c].dtype == dtype

    @pytest.mark.parametrize('unit', [u.mm, u.s])
    def test_to(self, array: na.AbstractVectorArray, unit: None | u.UnitBase):
        super().test_to(array=array, unit=unit)
        components = array.components
        if all(unit.is_equivalent(na.unit_normalized(components[c])) for c in components):
            array_new = array.to(unit)
            assert array_new.type_array_abstract == array.type_array_abstract
            assert all(array_new.components[c].unit == unit for c in array_new.components)
        else:
            with pytest.raises(u.UnitConversionError):
                array.to(unit)

    def test_length(self, array: na.AbstractVectorArray):
        super().test_length(array=array)
        components = array.components
        components_iter = iter(components)
        comp_0 = components[next(components_iter)]
        if all(na.unit_normalized(comp_0).is_equivalent(na.unit_normalized(components[c])) for c in components_iter):
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

        for c in components:
            components_expected[c] = na.as_named_array(components[c])[components_item[c]]

        result_expected = array.type_array.from_components(components_expected)

        result = array[item]

        assert np.all(result == result_expected)

    class TestMatmul(
        named_arrays.tests.test_core.AbstractTestAbstractArray.TestMatmul
    ):

        def test_matmul(
                self,
                array: None | bool | int | float | complex | str | na.AbstractArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray,
                out: bool,
        ):

            if isinstance(array, na.AbstractVectorArray) and isinstance(array_2, na.AbstractVectorArray):
                components_1 = array.components
                components_2 = array_2.components
                units = [na.unit_normalized(components_1[c]) * na.unit_normalized(components_2[c])
                         for c in components_1]
                units_are_compatible = all(units[0].is_equivalent(unit) for unit in units[1:])
                if not units_are_compatible:
                    with pytest.raises(u.UnitConversionError):
                        np.matmul(array, array_2)
                    return

            result_prototype = np.matmul(array, array_2)

            if out:
                out = 0 * result_prototype
                # if isinstance(out, na.AbstractVectorArray):
                #     for c in out.components:
                #         if not isinstance(out.components[c], na.AbstractArray):
                #             out.components[c] = np.asanyarray(out.components[c])
                #         elif isinstance(out.components[c], na.AbstractScalarArray):
                #             out.components[c].ndarray = np.asanyarray(out.components[c].ndarray)
                #         elif isinstance(out, na.ScalarArray):
                #             out.ndarray = np.asanyarray(out.ndarray)
            else:
                out = None

            print('out', out)

            result = np.matmul(array, array_2, out=out)

            if isinstance(array, na.AbstractVectorArray) and isinstance(array_2, na.AbstractVectorArray):
                assert isinstance(result, na.AbstractScalar)
            else:
                assert isinstance(result, na.AbstractVectorArray)

            if out is not None:
                assert result is out

            assert result.sum() != 0

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

                components = array.components

                kwargs = dict()
                if dtype is not None:
                    kwargs['dtype'] = dtype
                if where and (func in [np.min, np.amin, np.nanmin, np.max, np.amax, np.nanmax]):
                    kwargs['initial'] = 0
                if where:
                    kwargs['where'] = array > 0

                components_expected = dict()
                try:
                    for c in components:
                        kwargs_c = {
                            k: kwargs[k].components[c] if isinstance(kwargs[k], na.AbstractVectorArray) else kwargs[k]
                            for k in kwargs
                        }
                        components_expected[c] = func(
                            na.as_named_array(components[c]),
                            axis=axis,
                            keepdims=keepdims,
                            **kwargs_c,
                        )

                except Exception as e:
                    with pytest.raises(type(e)):
                        func(
                            array,
                            axis=axis,
                            keepdims=keepdims,
                            **kwargs,
                        )
                    return

                result_expected = array.type_array.from_components(components_expected)

                if out:
                    out = 0 * result_expected
                else:
                    out = None

                result = func(
                    array,
                    axis=axis,
                    out=out,
                    keepdims=keepdims,
                    **kwargs
                )

                assert np.all(result == result_expected)

        class TestPercentileLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestPercentileLikeFunctions,
        ):

            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractVectorArray,
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

                components = array.components

                if not isinstance(q, na.AbstractVectorArray):
                    components_q = array.type_array.from_scalar(q).components
                else:
                    components_q = q.components

                components_expected = dict()
                try:
                    for c in components:
                        components_expected[c] = func(
                            components[c],
                            q=components_q[c],
                            axis=axis,
                            keepdims=keepdims,
                        )

                except Exception as e:
                    with pytest.raises(type(e)):
                        func(array, q=q, axis=axis, keepdims=keepdims)
                    return

                result_expected = array.type_array.from_components(components_expected)

                if out:
                    out = 0 * result_expected
                    # if isinstance(out, na.AbstractVectorArray):
                    #     for c in out.components:
                    #         if not isinstance(out.components[c], na.AbstractArray):
                    #             out.components[c] = np.asanyarray(out.components[c])
                else:
                    out = None

                result = func(
                    array,
                    q=q,
                    axis=axis,
                    out=out,
                    keepdims=keepdims,
                )

                assert np.all(result == result_expected)

        class TestFFTLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestFFTLikeFunctions,
        ):

            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractArray,
                    axis: tuple[str, str],
            ):
                super().test_fft_like_functions(
                    func=func,
                    array=array,
                    axis=axis,
                )

                if axis not in array.shape:
                    with pytest.raises(ValueError):
                        func(a=array, axis=axis)
                    return

                result = func(a=array, axis=axis)

                assert result.type_array_abstract == array.type_array_abstract
                assert axis[1] in result.axes
                assert not axis[0] in result.axes
                assert result.size == array.size

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

                if not all(ax in array.shape for ax in axes):
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                if s is not None and axes.keys() != s.keys():
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                result = func(a=array, axes=axes, s=s)

                assert result.type_array_abstract == array.type_array_abstract
                assert all(axes[ax] in result.axes for ax in axes)
                assert all(ax not in result.axes for ax in axes)

        @pytest.mark.parametrize('axis', [None, 'x', 'y'])
        def test_sort(self, array: na.AbstractVectorArray, axis: None | str):
            super().test_sort(array=array, axis=axis)

            components = array.components
            components_expected = dict()
            try:
                for c in components:
                    if not na.shape(components[c]):
                        components_expected[c] = components[c]
                    else:
                        components_expected[c] = np.sort(components[c], axis=axis)

            except Exception as e:
                with pytest.raises(type(e)):
                    np.sort(array, axis=axis)
                return

            result_expected = array.type_array.from_components(components_expected)

            result = np.sort(array, axis=axis)

            assert np.all(result == result_expected)

        def test_nonzero(self, array: na.AbstractVectorArray):
            if not array.shape:
                with pytest.raises(DeprecationWarning, match="Calling nonzero on 0d arrays is deprecated, .*"):
                    np.nonzero(array)
                return

            result_expected = np.moveaxis(
                a=array[array != 0],
                source=f"{array.axes_flattened}_boolean",
                destination=f"{array.axes_flattened}_nonzero"
            )
            assert np.all(array[np.nonzero(array)] == result_expected)

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
        if any(np.issubdtype(na.get_dtype(array.components[c]), bool) for c in array.components):
            with pytest.raises(TypeError, match='numpy boolean subtract, .*'):
                array.ptp()
            return

        assert np.all(array.ptp() == np.ptp(array))

    def test_all(
            self,
            array: na.AbstractVectorArray,
    ):
        super().test_all(array=array)
        components = array.components
        if any(na.unit(components[c]) is not None for c in components):
            with pytest.raises(TypeError, match="no implementation found for .*"):
                array.all()
            return

        assert np.all(array.all() == np.all(array))

    def test_any(
            self,
            array: na.AbstractVectorArray,
    ):
        super().test_any(array=array)
        components = array.components
        if any(na.unit(components[c]) is not None for c in components):
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
