from __future__ import annotations
from typing import TypeVar, Generic, Type, ClassVar, Sequence, Callable, Collection, Any
import abc
import dataclasses
import numpy as np
import astropy.units as u
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

    @property
    def centers(self) -> FunctionArray:
        return self.type_explicit(
            inputs=self.inputs.centers,
            outputs=self.outputs.centers,
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

    def to(self, unit: u.UnitBase) -> FunctionArray:
        return self.type_explicit(
            inputs=self.inputs,
            outputs=self.outputs.to(unit),
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
            out = (None, ) * nout

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
            outputs_result = (outputs_result, )

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


@dataclasses.dataclass(eq=False, repr=False)
class FunctionArray(
    AbstractFunctionArray,
    na.AbstractExplicitArray,
    Generic[InputsT, OutputsT],
):
    inputs: InputsT = dataclasses.MISSING
    outputs: OutputsT = dataclasses.MISSING

    @classmethod
    def empty(cls, shape: dict[str, int], dtype: Type = float) -> FunctionArray:
        raise NotImplementedError

    @classmethod
    def zeros(cls, shape: dict[str, int], dtype: Type = float) -> FunctionArray:
        raise NotImplementedError

    @classmethod
    def ones(cls, shape: dict[str, int], dtype: Type = float) -> FunctionArray:
        raise NotImplementedError

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
        return self
