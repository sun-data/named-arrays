from __future__ import annotations
import functools
from typing import TypeVar, Generic, Type, ClassVar, Sequence, Callable, Collection, Any, Literal
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import astropy.visualization
import named_arrays as na
import itertools

__all__ = [
    "InputValueError",
    "AbstractFunctionArray",
    "FunctionArray",
    "AbstractPolynomialFunctionArray",
    "PolynomialFitFunctionArray",
]

InputsT = TypeVar("InputsT", bound=na.AbstractArray)
OutputsT = TypeVar("OutputsT", bound=na.AbstractArray)


class InputValueError(ValueError):
    """
    Exception raised when the inputs of two functions do not match
    """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFunctionArray(
    na.AbstractArray,
):

    @property
    @abc.abstractmethod
    def inputs(self) -> na.AbstractArray:
        """
        The arrays representing the inputs to this function.
        """

    @property
    @abc.abstractmethod
    def outputs(self) -> na.AbstractArray:
        """
        The arrays representing the outputs of this function.
        """

    __named_array_priority__: ClassVar[float] = (100 * na.AbstractVectorArray.__named_array_priority__)

    @property
    def axes_center(self) -> tuple(str):
        """
        Return keys corresponding to all input axes representing bin centers
        """
        axes_center = tuple()
        input_shape = self.inputs.shape
        output_shape = self.outputs.shape

        for axis in self.axes:
            if axis in input_shape:
                if axis in output_shape:
                    if input_shape[axis] == output_shape[axis]:
                        axes_center += (axis,)
                    else:
                        if input_shape[axis] == 1 or output_shape[axis] == 1:
                            axes_center += (axis,)
                        elif input_shape[axis] != output_shape[axis] + 1: # pragma: no cover
                            raise ValueError(
                                f"Output {axis=} dimension, {output_shape[axis]=}, must either match input axis dimension  {input_shape[axis]=}, (representing"
                                " bin centers) or exceed by one (representing bin vertices)."
                            )
                else:
                    axes_center += (axis,)
            else:
                axes_center += (axis,)

        return axes_center

    @property
    def axes_vertex(self) -> tuple(str):
        """
        Return keys corresponding to all input axes representing bin vertices
        """
        axes_vertex = tuple()

        for axis in self.axes:
            if axis not in self.axes_center:
                axes_vertex += (axis,)

        return axes_vertex

    @property
    def type_abstract(self) -> Type[AbstractFunctionArray]:
        return AbstractFunctionArray

    @property
    def type_explicit(self) -> Type[FunctionArray]:
        return type(self)

    @property
    def value(self) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=self.outputs.value,
        )

    @property
    def broadcasted(self) -> FunctionArray:

        exp = self.explicit

        broadcasted_shape_outputs = exp.shape
        broadcasted_shape_inputs = broadcasted_shape_outputs.copy()
        
        axes_vertex = exp.axes_vertex
        inputs = exp.inputs
        for key in broadcasted_shape_outputs:
            axes_vertex = axes_vertex
            if key in axes_vertex:  
                broadcasted_shape_inputs[key] = inputs.shape[key]

        return exp.replace(
            inputs=exp.inputs.broadcast_to(broadcasted_shape_inputs),
            outputs=exp.outputs.broadcast_to(broadcasted_shape_outputs),
        )

    def astype(
            self,
            dtype: str | np.dtype | type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> FunctionArray:

        exp = self.explicit

        return exp.replace(
            inputs=exp.inputs,
            outputs=exp.outputs.astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            )
        )

    def to(
        self,
        unit: u.UnitBase,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = [],
        copy: bool = True,
    ) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.to(
                unit=unit,
                equivalencies=equivalencies,
                copy=copy,
            ),
        )

    @property
    def length(self) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.length,
        )

    def add_axes(self, axes: str | Sequence[str]) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.add_axes(axes),
        )

    def combine_axes(
            self,
            axes: Sequence[str] = None,
            axis_new: str = None,
    ) -> FunctionArray:

        self = self.explicit

        axes = self.axes if axes is None else axes

        if not set(axes).issubset(set(self.axes)):
            raise ValueError(f"Axes {set(axes) - set(self.axes)} not in {self.axes}")

        for axis in axes:
            if axis in self.axes_vertex:
                raise ValueError(f"Axis {axis} describes input vertices and cannot be used in combine_axes.")

        inputs = self.inputs
        outputs = self.outputs
        inputs_shape_broadcasted = inputs.shape
        outputs_shape_broadcasted = outputs.shape

        shape = self.shape
        for axis in axes:
            if axis not in inputs_shape_broadcasted:
                inputs_shape_broadcasted[axis] = shape[axis]
            if axis not in outputs_shape_broadcasted:
                outputs_shape_broadcasted[axis] = shape[axis]

        inputs = na.broadcast_to(inputs, inputs_shape_broadcasted)
        outputs = na.broadcast_to(outputs, outputs_shape_broadcasted)

        return self.replace(
            inputs=inputs.combine_axes(axes=axes, axis_new=axis_new),
            outputs=outputs.combine_axes(axes=axes, axis_new=axis_new),
        )

    def __call__(
        self,
        inputs: na.AbstractArray,
        axis: None | str | tuple[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
        weights: None | tuple[na.AbstractScalar, dict[str, int], dict[str, int]] = None,
    ) -> Self:
        return self.regrid(
            inputs=inputs,
            axis=axis,
            method=method,
            weights=weights,
        )

    def regrid(
        self,
        inputs: na.AbstractArray,
        axis: None | str | tuple[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
        weights: None | tuple[na.AbstractScalar, dict[str, int], dict[str, int]] = None,
    ) -> Self:
        """
        Resample this function array onto a new set of input coordinates
        using :func:`named_arrays.regridding.regrid`.

        Parameters
        ----------
        inputs
            The new input coordinates on which to resample the outputs.
        axis
            The logical axes of the input over which to resample.
        method
            The resampling method to use.
        weights
            Optional weights which can be computed in advance using :meth:`weights`
            to greatly speed repeated resampling of the same `inputs`.

        See Also
        --------
        :meth:`weights`: If you need to resample the same coordinates more than once.
        """

        _self = self.explicit

        inputs_new, _, _ = _self._normalize__regrid__args(
            inputs=inputs,
            axis=axis,
        )

        if weights is not None:
            _weights, shape_input, shape_output = weights

        else:
            _weights, shape_input, shape_output = _self.weights(
                inputs=inputs,
                axis=axis,
                method=method,
            )

        outputs_new = na.regridding.regrid_from_weights(
            weights=_weights,
            shape_input=shape_input,
            shape_output=shape_output,
            values_input=_self.outputs,
        )

        final_coordinates_dict = {}

        if isinstance(inputs, na.AbstractVectorArray) and isinstance(_self.inputs, na.AbstractVectorArray):

            for c in _self.inputs.cartesian_nd.components:
                if inputs.cartesian_nd.components[c] is None:  # pragma: no cover
                    final_coordinates_dict[c] = _self.inputs.cartesian_nd.components[c]
                else:
                    final_coordinates_dict[c] = inputs.cartesian_nd.components[c]

            return _self.replace(
                inputs=_self.inputs.type_explicit.from_cartesian_nd(
                    array=na.CartesianNdVectorArray(final_coordinates_dict),
                    like=_self.inputs
                ),
                outputs=outputs_new,
            )
        else:
            return _self.replace(
                inputs=inputs_new,
                outputs=outputs_new,
            )

    def weights(
        self,
        inputs: na.AbstractArray,
        axis: None | str | tuple[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        Compute the resampling weights of this array using
        :func:`named_arrays.regridding.weights`.
        The output of this method is designed to be used by :meth:`regrid`.

        Parameters
        ----------
        inputs
            The new input coordinates on which to resample the outputs.
        axis
            The logical axes of the input over which to resample.
        method
            The resampling method to use.

        See Also
        --------
        :meth:`regrid`: A method designed to use these weights.
        """

        coordinates_new, coordinates_old, axis_input = self._normalize__regrid__args(
            inputs=inputs,
            axis=axis,
        )

        return na.regridding.weights(
            coordinates_input=coordinates_old,
            coordinates_output=coordinates_new,
            axis_input=axis_input,
            method=method
        )

    def _normalize__regrid__args(
        self,
        inputs: na.AbstractArray,
        axis: None | str | tuple[str],
    ):

        inputs_old = self.inputs
        inputs_new = inputs

        if isinstance(inputs_new, na.AbstractVectorArray):
            new_input_components = inputs_new.cartesian_nd.components
        else:
            new_input_components = dict(_dummy=inputs_new.explicit)

        if isinstance(inputs_old, na.AbstractVectorArray):
            old_input_components = inputs_old.cartesian_nd.components
        else:
            old_input_components = dict(_dummy=inputs_old.explicit)

        # broadcast new inputs against value to be interpolated
        if axis is None:
            axis = na.shape_broadcasted(
                new_input_components[c] for c in new_input_components if new_input_components[c] is not None
            )
            axis = tuple(axis)

        # check physical(vector) dimensions of each input match
        if new_input_components.keys() == old_input_components.keys():

            coordinates_new = {}
            coordinates_old = {}
            for c in new_input_components:
                component = new_input_components[c]
                if component is not None:
                    # if input components logical axes do not include interp axes, skip
                    if not set(axis).isdisjoint(component.axes):
                        coordinates_new[c] = component
                        coordinates_old[c] = old_input_components[c]

                else:
                    # check if uninterpolated physical axes vary along interpolation axes
                    if not set(axis).isdisjoint(old_input_components[c].axes):  # pragma: no cover
                        raise ValueError(
                            f"If a component is marked separable using `None`, its shape, {old_input_components[c].axes},"
                            f"should be disjoint from the interpolated axes, {axis}.",
                        )

        else:
            raise ValueError('Physical axes of new and old inputs must match.')  # pragma: no cover

        if isinstance(inputs_new, na.AbstractVectorArray):
            coordinates_new = na.CartesianNdVectorArray.from_components(coordinates_new)
        else:
            coordinates_new = coordinates_new['_dummy']

        if isinstance(inputs_old, na.AbstractVectorArray):
            coordinates_old = na.CartesianNdVectorArray.from_components(coordinates_old)
        else:
            coordinates_old = coordinates_old['_dummy']

        return coordinates_new, coordinates_old, axis

    def cell_centers(
        self,
        axis: None | str | Sequence[str] = None,
        random: bool = False,
    ) -> na.AbstractExplicitArray:
        exp = self.explicit
        return exp.replace(
            inputs=exp.inputs.cell_centers(axis, random=random),
            outputs=exp.outputs.cell_centers(axis, random=random),
        )

    def to_string_array(
        self,
        format_value: str = "%.2f",
        format_unit: str = "latex_inline",
        pad_unit: str = r"$\,$",
    ):
        exp = self.explicit
        kwargs = dict(
            format_value=format_value,
            format_unit=format_unit,
            pad_unit=pad_unit,
        )
        return exp.replace(
            inputs=na.as_named_array(exp.inputs).to_string_array(**kwargs),
            outputs=na.as_named_array(exp.outputs).to_string_array(**kwargs),
        )

    def _getitem(
            self,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray | na.AbstractFunctionArray,
    ) -> FunctionArray:

        array = self.explicit
        inputs = array.inputs
        outputs = array.outputs

        if isinstance(item, na.AbstractArray):
            if isinstance(item, na.AbstractFunctionArray):
                if not np.all(item.inputs == array.inputs):
                    raise ValueError("item inputs do not match function inputs.")

                axes = set(inputs.axes) - set(array.axes_center)
                inputs = inputs.cell_centers(axis=axes)
                item_inputs = item.outputs
                item_outputs = item.outputs
            else:
                item_inputs = item_outputs = item

            shape_item_inputs = item_inputs.shape
            shape_item_outputs = item_outputs.shape

            inputs = na.broadcast_to(inputs, na.broadcast_shapes(inputs.shape, shape_item_inputs))
            outputs = na.broadcast_to(outputs, na.broadcast_shapes(outputs.shape, shape_item_outputs))

        elif isinstance(item, dict):

            if not set(item).issubset(array.axes): # pragma: no cover
                raise ValueError(f"item contains axes {set(item) - set(array.axes)} that does not exist in {set(array.axes)}")

            item_inputs = dict()
            item_outputs = dict()
            for ax in item:
                item_ax = item[ax]
                if isinstance(item_ax, na.AbstractFunctionArray):
                    item_inputs[ax] = item_ax.inputs
                    item_outputs[ax] = item_ax.outputs
                else:
                    axes_center = array.axes_center
                    if ax in axes_center:
                        #can't assume center ax is in both outputs and inputs
                        if ax in inputs.shape:
                            item_inputs[ax] = item_ax
                        if ax in outputs.shape:
                            item_outputs[ax] = item_ax
                    axes_vertex = array.axes_vertex
                    if ax in axes_vertex:
                        if isinstance(item_ax, int):
                            item_outputs[ax] = slice(item_ax, item_ax + 1)
                            item_inputs[ax] = slice(item_ax, item_ax + 2)
                        elif isinstance(item_ax, slice):
                            item_outputs[ax] = item_ax
                            if item_ax.start is None and item_ax.stop is None:
                                item_inputs[ax] = item_ax
                            else:
                                if item_ax.stop is not None:
                                    item_inputs[ax] = slice(item_ax.start, item_ax.stop + 1)
                                else:
                                    item_inputs[ax] = slice(item_ax.start, None)
                        else:
                            return NotImplemented

        else:
            return NotImplemented

        return array.replace(
            inputs=inputs[item_inputs],
            outputs=outputs[item_outputs],
        )

    def _getitem_reversed(
            self,
            array: na.AbstractArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractFunctionArray
    ):
        if isinstance(array, (na.AbstractScalar, na.AbstractVectorArray)):
            array = na.FunctionArray(array, array)
        else:
            return NotImplemented
        return array._getitem(item)

    def __bool__(self) -> bool:
        result = super().__bool__()
        return result and bool(self.outputs)

    def __mul__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            exp = self.explicit
            return exp.replace(
                outputs=exp.outputs * other,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            exp = self.explicit
            return exp.replace(
                outputs=self.outputs << other,
            )
        else:
            return super().__lshift__(other)

    def __truediv__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            exp = self.explicit
            return exp.replace(
                outputs=self.outputs / other,
            )
        else:
            return super().__truediv__(other)

    def __array_matmul__(
            self,
            x1: float | u.Quantity | FunctionArray,
            x2: float | u.Quantity | FunctionArray,
            out: None | na.FunctionArray = None,
            **kwargs,
    ) -> FunctionArray:

        if isinstance(x1, na.AbstractArray):
            if isinstance(x1, AbstractFunctionArray):
                inputs_1 = x1.inputs
                outputs_1 = x1.outputs
            else:
                return NotImplemented
        else:
            inputs_1 = None
            outputs_1 = x1

        if isinstance(x2, na.AbstractArray):
            if isinstance(x2, AbstractFunctionArray):
                inputs_2 = x2.inputs
                outputs_2 = x2.outputs
            else:
                return NotImplemented
        else:
            inputs_2 = None
            outputs_2 = x2

        if inputs_1 is not None:
            inputs = inputs_1
            if inputs_2 is not None:
                if np.any(inputs_1 != inputs_2):
                    raise InputValueError("`x1.inputs` and `x2.inputs` must be equal")
            else:
                pass
        else:
            if inputs_2 is not None:
                inputs = inputs_2
            else:
                return NotImplemented

        if out is not None:
            if not np.all(inputs == out.inputs):
                raise InputValueError("`out.inputs` must be equal to `x1.inputs` and `x2.inputs`")
        result = self.explicit.replace(
            inputs=inputs,
            outputs=np.matmul(
                outputs_1,
                outputs_2,
                out=out.outputs if out is not None else out,
                **kwargs,
            ),
        )

        if out is not None:
            result = out

        return result

    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | na.AbstractArray | tuple[na.AbstractArray, ...]:

        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

        nout = function.nout

        inputs_inputs = []
        inputs_outputs = []
        for inp in inputs:
            if isinstance(inp, na.AbstractArray):
                if isinstance(inp, AbstractFunctionArray):
                    inputs_inputs.append(inp.inputs)
                    inputs_outputs.append(inp.outputs)
                else:
                    inputs_outputs.append(inp)
            else:
                inputs_outputs.append(inp)

        for inputs_input in inputs_inputs[1:]:
            if np.any(inputs_input != inputs_inputs[0]):
                raise InputValueError(f"all inputs to {function} must have the same coordinates")

        if "out" in kwargs:
            out = kwargs.pop("out")
            outputs_out = list()
            for o in out:
                if o is not None:
                    if isinstance(o, self.type_explicit):
                        outputs_o = o.outputs
                    else:
                        raise ValueError(
                            f"each element of `out` must be an instance of {self.type_explicit}, got {type(o)}"
                        )
                else:
                    outputs_o = None
                outputs_out.append(outputs_o)
            if nout == 1:
                outputs_out = outputs_out[0]
            else:
                outputs_out = tuple(outputs_out)

            kwargs["out"] = outputs_out
        else:
            out = (None,) * nout

        if "where" in kwargs:
            where = kwargs.pop("where")
            if isinstance(where, na.AbstractArray):
                if isinstance(where, AbstractFunctionArray):
                    if np.any(where.inputs != inputs_inputs[0]):
                        raise InputValueError(f"`where.inputs` must match the rest of the inputs")
                    kwargs["where"] = where.outputs
                else:
                    return NotImplemented
            else:
                kwargs["where"] = where

        inputs_result = inputs_inputs[0]
        outputs_result = getattr(function, method)(*inputs_outputs, **kwargs)

        if nout == 1:
            outputs_result = (outputs_result,)

        result = list(
            self.explicit.replace(
                inputs=inputs_result,
                outputs=outputs_result[i],
            )
            for i in range(nout)
        )

        for i in range(nout):
            if out[i] is not None:
                out[i].inputs = result[i].inputs
                out[i].outputs = result[i].outputs
                result[i] = out[i]

        if nout == 1:
            result = result[0]
        else:
            result = tuple(result)

        return result

    def __array_function__(
            self,
            func: Callable,
            types: Collection,
            args: tuple,
            kwargs: dict[str, Any],
    ):
        result = super().__array_function__(func=func, types=types, args=args, kwargs=kwargs)
        if result is not NotImplemented:
            return result

        from . import function_array_functions

        if func in function_array_functions.DEFAULT_FUNCTIONS:
            return function_array_functions.array_function_default(func, *args, **kwargs)

        if func in function_array_functions.PERCENTILE_LIKE_FUNCTIONS:
            return function_array_functions.array_function_percentile_like(func, *args, **kwargs)

        if func in function_array_functions.ARG_REDUCE_FUNCTIONS:
            return function_array_functions.array_function_arg_reduce(func, *args, **kwargs)

        if func in function_array_functions.STACK_LIKE_FUNCTIONS:
            return function_array_functions.array_function_stack_like(func, *args, **kwargs)

        if func in function_array_functions.HANDLED_FUNCTIONS:
            return function_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def __named_array_function__(self, func, *args, **kwargs):
        result = super().__named_array_function__(func, *args, **kwargs)
        if result is not NotImplemented:
            return result

        from . import function_named_array_functions

        if func in function_named_array_functions.ASARRAY_LIKE_FUNCTIONS:
            return function_named_array_functions.asarray_like(func=func, *args, **kwargs)

        if func in function_named_array_functions.NDFILTER_FUNCTIONS:
            return function_named_array_functions.ndfilter(func, *args, **kwargs)

        if func in function_named_array_functions.HANDLED_FUNCTIONS:
            return function_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def interp_linear(
            self,
            item: dict[str, na.AbstractArray],
    ) -> FunctionArray:
        raise NotImplementedError


    def pcolormesh(
            self,
            axs: np.ndarray,
            input_component_x: str,
            input_component_y: str,
            input_component_row: None | str = None,
            input_component_column: None | str = None,
            index: None | dict[str, int] = None,
            output_component_color: None | str = None,
            **kwargs: Any,
    ):
        """
        Plot a :class:`FunctionArray` via :func:`matplotlib.pyplot.pcolormesh`.

        :func:`FunctionArray.pcolormesh` takes in an axes object, or array of axes objects, along with components to
        be plotted along the x and y plot axes (:attr:`input_component_x` and :attr:`input_component_y`). Additional
        components can be tiled along subplot row/column and are specified in :attr:`input_component_row` and
        :attr:`input_component_column`.

        .. jupyter-execute::

            import named_arrays as na
            import numpy as np
            import astropy.units as u
            import matplotlib.pyplot as plt

            position = na.Cartesian2dVectorLinearSpace(
                start=-10,
                stop=10,
                axis=na.Cartesian2dVectorArray(
                    x='position_x',
                    y='position_y',
                ),
                num=21,
            ) * u.m

            x_width = 5 * u.m
            y_width = 2 * u.m
            velocity = 1 * u.m/u.s
            time = na.ScalarLinearSpace(
                start=0 * u.s,
                stop=3 * u.s,
                num=4,
                axis='time'
            )

            intensity = np.exp(-(((position.x + velocity*time)/x_width) ** 2 + ((position.y + 2*velocity*time)/y_width)** 2))
            scene = na.FunctionArray(
                inputs=position,
                outputs=intensity,
            )

            fig, axs = plt.subplots(
                nrows=scene.outputs.shape['time'],
                squeeze=False,
                sharex=True,
                subplot_kw=dict(aspect='equal'),
            )
            scene.pcolormesh(
                axs=axs,
                input_component_x='x',
                input_component_y='y',
                input_component_row='time',
            )
        """

        if axs.ndim == 1:
            if input_component_row is not None:
                axs = axs[..., np.newaxis]
            if input_component_column is not None:
                axs = axs[np.newaxis, ...]

        axs = na.ScalarArray(
            ndarray=axs,
            axes=('row', 'column')
        )

        if index is None:
            index = dict()

        with astropy.visualization.quantity_support():
            for index_subplot in axs.ndindex():

                index_final = index.copy()
                if input_component_row is not None:
                    index_final[input_component_row] = index_subplot['row']
                if input_component_column is not None:
                    index_final[input_component_column] = index_subplot['column']

                inp = self[index_final].inputs.cartesian_nd

                inp_x = inp.components[input_component_x].ndarray
                inp_y = inp.components[input_component_y].ndarray

                out = self[index_final].outputs
                if output_component_color is not None:
                    out = out.components[output_component_color]

                ax = axs[index_subplot].ndarray
                ax.pcolormesh(
                    inp_x,
                    inp_y,
                    out.ndarray,
                    shading='auto',
                    **kwargs,
                )

                if index_subplot['row'] == axs.shape['row'] - 1:
                    if isinstance(inp_x, u.Quantity):
                        ax.set_xlabel(f'{input_component_x} ({inp_x.unit})')
                    else:
                        ax.set_xlabel(f'{input_component_x}')
                else:
                    ax.set_xlabel(None)

                if index_subplot['column'] == 0:
                    if isinstance(inp_y, u.Quantity):
                        ax.set_ylabel(f'{input_component_y} ({inp_y.unit})')
                    else:
                        ax.set_ylabel(f'{input_component_y}')
                else:
                    ax.set_ylabel(None)

                if input_component_column is not None:
                    if input_component_column in inp.components:
                        if index_subplot['row'] == 0:
                            inp_column = inp.components[input_component_column]
                            ax.text(
                                x=0.5,
                                y=1.01,
                                s=f'{inp_column.mean().ndarray.value:0.03f} {inp_column.unit:latex_inline}',
                                transform=ax.transAxes,
                                ha='center',
                                va='bottom'
                            )

                if input_component_row is not None:
                    if input_component_row in inp.components:
                        if index_subplot['column'] == axs.shape['column'] - 1:
                            inp_row = inp.components[input_component_row]
                            ax.text(
                                x=1.01,
                                y=0.5,
                                s=f'{inp_row.mean().ndarray.value:0.03f} {inp_row.unit:latex_inline}',
                                transform=ax.transAxes,
                                va='center',
                                ha='left',
                                rotation=-90,
                            )


@dataclasses.dataclass(eq=False, repr=False)
class FunctionArray(
    AbstractFunctionArray,
    na.AbstractExplicitArray,
    Generic[InputsT, OutputsT],
):
    """
    A representation of a discrete function.

    A composition of two arrays: :attr:`inputs` and :attr:`outputs`.
    :attr:`inputs` represents the inputs (or independent variables) of the
    function, and :attr:`outputs` represents the outputs (or dependent variables) of
    the function.
    """
    inputs: InputsT = 0
    """The inputs of the function."""

    outputs: OutputsT = 0
    """The outputs of the function."""

    @classmethod
    def from_scalar_array(
            cls: type[Self],
            a: float | u.Quantity | na.AbstractScalarArray,
            like: None | Self = None,
    ) -> Self:

        self = super().from_scalar_array(a=a, like=like)

        if like is None:
            self.inputs = a
            self.outputs = a
        else:
            if isinstance(like.inputs, na.AbstractArray):
                self.inputs = like.inputs.from_scalar_array(a=a, like=like.inputs)
            else:
                self.inputs = a

            if isinstance(like.outputs, na.AbstractArray):
                self.outputs = like.outputs.from_scalar_array(a=a, like=like.outputs)
            else:
                self.outputs = a

        return self

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self.inputs.shape | self.outputs.shape)

    @property
    def shape(self) -> dict[str, int]:

        axes_center = self.axes_center
        axes_vertex = self.axes_vertex

        outputs_shape = self.outputs.shape
        inputs_shape = self.inputs.shape

        outputs_shape_center = {ax: outputs_shape[ax] for ax in outputs_shape if ax in axes_center}
        inputs_shape_center = {ax: inputs_shape[ax] for ax in inputs_shape if ax in axes_center}
        outputs_shape_vertex = {ax: outputs_shape[ax] for ax in outputs_shape if ax in axes_vertex}

        shape = na.broadcast_shapes(
            outputs_shape_center,
            outputs_shape_vertex,
            inputs_shape_center,
        )

        return shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.array(tuple(self.shape.values())).prod())

    @property
    def explicit(self) -> FunctionArray:
        return self.replace(
            inputs=na.explicit(self.inputs),
            outputs=na.explicit(self.outputs),
        )

    def __setitem__(
            self,
            item: dict[str, int | slice | na.AbstractScalar | na.AbstractFunctionArray] | na.AbstractFunctionArray,
            value: float | u.Quantity | na.FunctionArray,
    ):

        if isinstance(item, na.AbstractFunctionArray):
            if not np.all(item.inputs == self.inputs):
                raise ValueError("boolean advanced index does not have the same inputs as the array")

            item_inputs = item.outputs
            item_outputs = item.outputs

        elif isinstance(item, dict):

            item_inputs = dict()
            item_outputs = dict()
            for ax in item:
                item_ax = item[ax]
                if isinstance(item_ax, na.AbstractFunctionArray):
                    item_inputs[ax] = item_ax.inputs
                    item_outputs[ax] = item_ax.outputs
                else:
                    item_inputs[ax] = item_outputs[ax] = item_ax
        else:
            raise TypeError(
                f"`item` must be an instance of `{dict.__name__}`, or `{na.AbstractFunctionArray.__name__}`, "
                f"got `{type(item)}`"
            )

        if isinstance(value, na.AbstractArray):
            if isinstance(value, na.AbstractFunctionArray):
                value_inputs = value.inputs
                value_outputs = value.outputs

                # maybe this should only set to None vertex axes only
                axes_vertex = self.axes_vertex
                for ax in value_inputs.shape:
                    axes_vertex = axes_vertex
                    if ax in axes_vertex:
                        value_inputs = None

            else:
                if value.shape:
                    raise ValueError(
                        f"if `value` is an instance of `{na.AbstractArray.__name__}`, "
                        f"but not an instance of `{na.AbstractFunctionArray.__name__}`, "
                        f"`value.shape` should be empty, got {value.shape}"
                    )
                else:
                    value_inputs = None
                    value_outputs = value
        else:
            value_inputs = None
            value_outputs = value

        if value_inputs is not None:
            self.inputs[item_inputs] = value_inputs

        self.outputs[item_outputs] = value_outputs


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolynomialFunctionArray(
    AbstractFunctionArray,
):
    @property
    @abc.abstractmethod
    def coefficients(self) -> na.AbstractVectorArray | na.AbstractMatrixArray:
        """
        A vector or matrix representing the coefficients of the polynomial.

        If this function is scalar-valued, :attr:`coefficients` should be a vector,
        and if this function is vector-valued, :attr`coefficients` should be a matrix.
        """

    @property
    @abc.abstractmethod
    def degree(self) -> int:
        """degree of the polynomial"""

    @property
    @abc.abstractmethod
    def components_polynomial(self) -> None | str | Sequence[str]:
        """The components of the input that this polynomial depends on."""

    @property
    @abc.abstractmethod
    def axis_polynomial(self) -> None | str | Sequence[str]:
        """the logical axes along which this polynomial is distributed"""

    @abc.abstractmethod
    def design_matrix(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:
        """
        The `design matrix <https://en.wikipedia.org/wiki/Design_matrix>`_
        corresponding to the given inputs.

        Note that while this is `called` a matrix, this function returns a vector
        since the rows are independent observations, and this is concept is
        already captured by the logical axes in the array.

        Parameters
        ----------
        inputs
            the set of independent variables to convert into the design matrix
        """

    def __call__(
        self,
        inputs: float | u.Quantity | na.ScalarArray | na.AbstractVectorArray,
    ) -> na.AbstractFunctionArray:
        return na.FunctionArray(
            inputs, self.design_matrix(inputs) @ self.coefficients
        )

    @property
    def predictions(self) -> OutputsT:
        """
        The outputs of the polynomial model given :attr:`inputs`.
        Equivalent to ``self(self.inputs).outputs``.
        """
        return self(self.inputs).outputs


@dataclasses.dataclass(eq=False, repr=False)
class PolynomialFitFunctionArray(
    FunctionArray,
    AbstractPolynomialFunctionArray,
):
    """
    A :class:`named_arrays.PolynomialFitFunctionArray` carries the independent variables, inputs, and dependent variables, outputs,
    of a discrete function, and a linear least squares polynomial fit of specified degree to that function.

    Parameters
    ----------
    inputs
        the set of independent variables
    outputs
        the set of dependent variables
    degree
        the degree of the polynomial
    components_polynomial
        the components used in the polynomial fit
    axis_polynomial
        the logical axis of the polynomial fit

    Examples
    --------

    .. nblinkgallery::
        :caption: Relevant Tutorials
        :name: rst-link-gallery

        ../tutorials/PolynomialFunctionArray
    """

    degree: int = None
    components_polynomial: None | str | Sequence[str] = None
    axis_polynomial: None | str | Sequence[str] = None

    @functools.cached_property
    def coefficients(self) -> na.AbstractVectorArray | na.AbstractMatrixArray:
        d = self.design_matrix(self.inputs)
        dTd = self._outer(d, d, self.axis_polynomial)
        dTo = self._outer(d, self.outputs, self.axis_polynomial)

        return dTd.inverse @ dTo

    def design_matrix(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:

        design_matrix = {}

        if isinstance(inputs, na.AbstractScalar):
            inputs = na.CartesianNdVectorArray({"dummy": inputs})
        inputs = inputs.cartesian_nd.broadcasted.components

        components = self.components_polynomial

        if components is None:
            components = tuple(inputs)
        elif isinstance(components, str):
            components = (components,)

        inputs = {c: inputs[c] for c in components}

        for i in range(self.degree + 1):
            combinations = itertools.combinations_with_replacement(
                inputs, i
            )
            for combination in combinations:
                key = "*".join(combination)
                design_matrix[key] = 1
                for k in combination:
                    design_matrix[key] = design_matrix[key] * inputs[k]

        design_matrix = na.CartesianNdVectorArray(design_matrix)

        return design_matrix

    @classmethod
    def _outer(cls, v1, v2, axis):
        v1_T_v2_components = {}

        if isinstance(v1, na.AbstractVectorArray):
            v1_broadcasted = v1.broadcasted.components
            if isinstance(v2, na.AbstractVectorArray):
                for c1 in v1_broadcasted:
                    v2_broadcasted = v2.broadcasted.components
                    row_components = {}
                    for c2 in v2_broadcasted:
                        row_components[c2] = (
                                v1_broadcasted[c1] * v2_broadcasted[c2]
                        ).sum(axis=axis)
                    v1_T_v2_components[c1] = v2.type_explicit.from_components(row_components)
                v1_T_v2 = v1.type_matrix.from_components(v1_T_v2_components)

            else:
                for c1 in v1_broadcasted:
                        row_components = (v1_broadcasted[c1] * v2).sum(axis=axis)
                        v1_T_v2_components[c1] = row_components
                v1_T_v2 = v1.type_explicit.from_components(v1_T_v2_components)

        return v1_T_v2



