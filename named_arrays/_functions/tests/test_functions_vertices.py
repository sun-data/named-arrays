from typing import Sequence, Literal, Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._vectors.cartesian.tests.test_vectors_cartesian_2d
from . import test_functions

__all__ = [
    "AbstractTestAbstractFunctionArrayVertices",
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_z = named_arrays.tests.test_core.num_z
_num_distribution = named_arrays.tests.test_core.num_distribution


def _function_arrays():
    functions_1d = [
        na.FunctionArray(
            inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y+1),
            outputs=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(y=_num_y))
        )
    ]

    inputs_2d = [
        na.ScalarLinearSpace(0, 1, axis='y', num=_num_y+1),
        na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=1,
            axis=na.Cartesian2dVectorArray('x', 'y'),
            num=na.Cartesian2dVectorArray(_num_x+1, _num_y+1)
        )
    ]

    outputs_2d = [
        na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y, z=_num_z)),
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
            inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y+1),
            outputs=na.ScalarUniformRandomSample(-6, 6, shape_random=dict(y=_num_y))
        ),
        na.FunctionArray(
            inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y+1),
            outputs=na.Cartesian2dVectorUniformRandomSample(-6, 6, shape_random=dict(y=_num_y))
        )
    )


class AbstractTestAbstractFunctionArrayVertices(
    test_functions.AbstractTestAbstractFunctionArray,
):
    @pytest.mark.parametrize(
        argnames="axis,method",
        argvalues=[
            (("x", "y"), "conservative"),
        ],
    )
    def test__call__(
        self,
        array: na.FunctionArray,
        axis: None | str | tuple[str],
        method: Literal['multilinear', 'conservative'],
    ):
        if len(array.axes_vertex) == len(axis):
            super().test__call__(
                array=array,
                axis=axis,
                method=method,
            )

    @pytest.mark.parametrize(
        argnames="axis,method",
        argvalues=[
            (("x", "y"), "conservative"),
        ],
    )
    def test__call__with_weights(
        self,
        array: na.FunctionArray,
        axis: None | str | tuple[str],
        method: Literal['multilinear', 'conservative'],
    ):
        if len(array.axes_vertex) == len(axis):
            super().test__call__with_weights(
                array=array,
                axis=axis,
                method=method,
            )

    @pytest.mark.parametrize('newshape', [dict(r=-1)])
    def test_reshape(self, array: na.AbstractArray, newshape: dict[str, int]):

        for ax in newshape:
            if ax in array.axes_vertex or (ax not in array.axes and len(array.axes_vertex) != 0):
                with pytest.raises(ValueError):
                    np.reshape(array, newshape=newshape)
                return

        super().test_reshape(array, newshape)
    @pytest.mark.parametrize('axes', [None, ('x', 'y'), ('x', 'y', 'z')])
    def test_combine_axes(
            self,
            array: na.AbstractArray,
            axes: None | Sequence[str]
    ):
        axis_new = 'new_test_axis'
        if axes is None and len(array.axes_vertex) != 0:
            with pytest.raises(ValueError):
                array.combine_axes(axes=axes, axis_new=axis_new)
            return

        if np.any([ax in array.axes_vertex for ax in axes]):
            with pytest.raises(ValueError):
                array.combine_axes(axes=axes, axis_new=axis_new)
            return

        super().test_combine_axes(array=array, axes=axes)


    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=slice(0, 1)),
        ]

    )
    def test__getitem__(
            self,
            array: na.AbstractFunctionArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):

        super().test__getitem__(array=array, item=item)


    @pytest.mark.parametrize("array_2", _function_arrays_2())
    class TestUfuncBinary(
        test_functions.AbstractTestAbstractFunctionArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize("array_2", _function_arrays_2())
    class TestMatmul(
        test_functions.AbstractTestAbstractFunctionArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_functions.AbstractTestAbstractFunctionArray.TestArrayFunctions
    ):
        @pytest.mark.parametrize('newshape', [dict(r=-1)])
        def test_reshape(self, array: na.AbstractArray, newshape: dict[str, int]):

            for ax in newshape:
                if ax in array.axes_vertex or (ax not in array.axes and len(array.axes_vertex) != 0):
                    with pytest.raises(ValueError):
                        np.reshape(array, newshape=newshape)
                    return

            super().test_reshape(array, newshape)

        @pytest.mark.parametrize('axis', ['x', 'y'])
        def test_concatenate(
                self,
                array: na.AbstractArray,
                axis: str,
        ):
            arrays = [array,array]
            if axis in array.axes_vertex:
                with pytest.raises(ValueError):
                    np.concatenate(arrays, axis=axis)
                return
            super().test_concatenate(array, axis)

        def test_nonzero(self, array: na.AbstractArray):
            if len(array.axes_vertex) != 0:
                with pytest.raises(ValueError, match=f"item not supported by array with type {type(array)}"):
                    #matches test_core, but is a bad test. nonzero only applied to boolean arrays
                    array[np.nonzero(array>array.mean())]
                return
            super().test_nonzero(array=array)

        class TestReductionFunctions(
            test_functions.AbstractTestAbstractFunctionArray.TestArrayFunctions.TestReductionFunctions
        ):
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
                            num=na.Cartesian2dVectorArray(_num_x + 1, _num_y + 1)
                        ),
                        outputs=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0)
                                | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0)
                    )
                ]
            )
            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractFunctionArray,
                    axis: None | str | Sequence[str],
                    dtype: None | type | np.dtype,
                    keepdims: bool,
                    where: bool | na.AbstractFunctionArray,
            ):
                super().test_reduction_functions(
                    func=func,
                    array=array,
                    axis=axis,
                    dtype=dtype,
                    keepdims=keepdims,
                    where=where,
                )

    class TestNamedArrayFunctions(
        test_functions.AbstractTestAbstractFunctionArray.TestNamedArrayFunctions
    ):
        class TestHistogram(
            test_functions.AbstractTestAbstractFunctionArray.TestNamedArrayFunctions.TestHistogram,
        ):
            def test_histogram(
                self,
                array: na.AbstractFunctionArray,
                bins: Literal["dict"],
                axis: None | str | Sequence[str],
                min: None | na.AbstractScalarArray | na.AbstractVectorArray,
                max: None | na.AbstractScalarArray | na.AbstractVectorArray,
                weights: None | na.AbstractScalarArray,
            ):

                axis_normalized = tuple(array.shape) if axis is None else (axis,) if isinstance(axis, str) else axis
                for ax in axis_normalized:
                    if ax in array.axes_vertex:
                        with pytest.raises(ValueError, match="Taking a histogram of a histogram doesn't work right now."):
                            na.histogram(
                                a=array,
                                bins=bins,
                                axis=axis,
                                min=min,
                                max=max,
                                weights=weights,
                            )
                        return

                super().test_histogram(
                    array=array,
                    bins=bins,
                    axis=axis,
                    min=min,
                    max=max,
                    weights=weights,
                )


@pytest.mark.parametrize("array", _function_arrays())
class TestFunctionArrayVertices(
    AbstractTestAbstractFunctionArrayVertices,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray,
):

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(x=0, y=0),
            dict(y=slice(None)),
            # dict(y=na.ScalarArrayRange(0, _num_y, axis='y')),
            # dict(x=na.ScalarArrayRange(0, _num_x, axis='x'), y=na.ScalarArrayRange(0, _num_y, axis='y')),
            na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y+1),
                outputs=na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
            )
        ]
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y+1),
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