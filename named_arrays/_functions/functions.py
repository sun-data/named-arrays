from __future__ import annotations
from typing import TypeVar, Generic, Type, ClassVar, Sequence, Callable, Collection, Any
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import astropy.visualization
import named_arrays as na

__all__ = [
    "InputValueError",
    "AbstractFunctionArray",
    "FunctionArray",
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

    __named_array_priority__: ClassVar[float] = 100 * na.AbstractVectorArray.__named_array_priority__

    @property
    def type_abstract(self) -> Type[AbstractFunctionArray]:
        return AbstractFunctionArray

    @property
    def type_explicit(self) -> Type[FunctionArray]:
        return FunctionArray

    @property
    def value(self) -> FunctionArray:
        return self.type_explicit(
            inputs=self.inputs,
            outputs=self.outputs.value,
        )

    def astype(
            self,
            dtype: str | np.dtype | type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> FunctionArray:

        return self.type_explicit(
            inputs=self.inputs,
            outputs=self.outputs.astype(
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
        return self.type_explicit(
            inputs=self.inputs,
            outputs=self.outputs.to(
                unit=unit,
                equivalencies=equivalencies,
                copy=copy,
            ),
        )

    @property
    def length(self) -> FunctionArray:
        return self.type_explicit(
            inputs=self.inputs,
            outputs=self.outputs.length,
        )

    def add_axes(self, axes: str | Sequence[str]) -> FunctionArray:
        return self.type_explicit(
            inputs=self.inputs,
            outputs=self.outputs.add_axes(axes),
        )

    def combine_axes(
            self,
            axes: Sequence[str] = None,
            axis_new: str = None,
    ) -> FunctionArray:

        a = self.explicit
        shape = a.shape

        axes = tuple(shape) if axes is None else axes

        shape_base = {ax: shape[ax] for ax in shape if ax in axes}

        inputs = a.inputs.broadcast_to(na.broadcast_shapes(a.inputs.shape, shape_base))
        outputs = a.outputs.broadcast_to(na.broadcast_shapes(a.outputs.shape, shape_base))
        return self.type_explicit(
            inputs=inputs.combine_axes(axes=axes, axis_new=axis_new),
            outputs=outputs.combine_axes(axes=axes, axis_new=axis_new),
        )

    def _getitem(
            self,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractFunctionArray,
    ) -> FunctionArray:

        array = self.explicit
        inputs = array.inputs
        outputs = array.outputs

        shape = array.shape
        shape_inputs = inputs.shape
        shape_outputs = outputs.shape

        if isinstance(item, na.AbstractFunctionArray):
            if np.any(item.inputs != self.inputs):
                raise ValueError("boolean advanced index does not have the same inputs as the array")

            item_inputs = item.outputs
            item_outputs = item.outputs

            shape_item_inputs = item_inputs.shape
            shape_item_outputs = item_outputs.shape

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

            shape_item_inputs = {ax: shape[ax] for ax in item_inputs if ax in shape}
            shape_item_outputs = {ax: shape[ax] for ax in item_outputs if ax in shape}

        else:
            return NotImplemented

        inputs = na.broadcast_to(inputs, na.broadcast_shapes(shape_inputs, shape_item_inputs))
        outputs = na.broadcast_to(outputs, na.broadcast_shapes(shape_outputs, shape_item_outputs))

        return self.type_explicit(
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
            return self.type_explicit(
                inputs=self.inputs,
                outputs=self.outputs * other,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            return self.type_explicit(
                inputs=self.inputs,
                outputs=self.outputs << other,
            )
        else:
            return super().__lshift__(other)

    def __truediv__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            return self.type_explicit(
                inputs=self.inputs,
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

        result = self.type_explicit(
            inputs=inputs,
            outputs=np.matmul(
                outputs_1,
                outputs_2,
                out=out.outputs if out is not None else out,
                **kwargs,
            )
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
                    if not inp.shape:
                        inputs_outputs.append(inp)
                    else:
                        return NotImplemented
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
            self.type_explicit(inputs=inputs_result, outputs=outputs_result[i])
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

        if func in function_named_array_functions.HANDLED_FUNCTIONS:
            return function_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def interp_linear(
            self,
            item: dict[str, na.AbstractArray],
    ) -> FunctionArray:
        a = self.broadcasted
        return a.type_explicit(
            inputs=a.inputs[item],
            outputs=a.outputs[item],
        )

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

                inp = self[index_final].inputs

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
                    shading='nearest',
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
                                s=f'{inp_column.mean().array.value:0.03f} {inp_column.unit:latex_inline}',
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
                                s=f'{inp_row.mean().array.value:0.03f} {inp_row.unit:latex_inline}',
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
    inputs: InputsT = 0
    outputs: OutputsT = 0

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
        return tuple(self.shape.keys())

    @property
    def shape(self) -> dict[str, int]:
        return na.shape_broadcasted(self.inputs, self.outputs)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.array(tuple(self.shape.values())).prod())

    @property
    def explicit(self) -> FunctionArray:
        return self.type_explicit(
            inputs=na.explicit(self.inputs),
            outputs=na.explicit(self.outputs),
        )

    def __setitem__(
            self,
            item: dict[str, int | slice | na.AbstractScalar | na.AbstractFunctionArray] | na.AbstractFunctionArray,
            value: float | u.Quantity | na.FunctionArray,
    ):

        if isinstance(item, na.AbstractFunctionArray):
            if np.any(item.inputs != self.inputs):
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
