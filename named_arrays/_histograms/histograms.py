from __future__ import annotations
from typing import Generic, TypeVar
import dataclasses
import named_arrays as na
import regridding

from named_arrays import AbstractExplicitArray

InputsT = TypeVar("InputsT", bound=na.AbstractArray)
OutputsT = TypeVar("OutputsT", bound=na.AbstractArray)

__all__ = [
    'AbstractHistogramArray',
    'HistogramArray',
]

@dataclasses.dataclass(eq=False, repr=False)
class AbstractHistogramArray(
    na.AbstractFunctionArray,
):
    def __call__(
        self,
        new_inputs: na.AbstractArray,
    ) -> HistogramArray:

        interpolation_axes = [component for component in new_inputs.components if component is not None]

        old_inputs = (self.inputs.x.ndarray, self.inputs.y.ndarray)

        new_outputs = regridding.regrid(
            vertices_input=old_inputs,
            vertices_output=new_inputs,
            values_input=self.outputs.ndarray,
        )

        return HistogramArray(inputs=new_inputs, outputs=new_outputs)



@dataclasses.dataclass(eq=False, repr=False)
class HistogramArray(
    AbstractHistogramArray,
    na.FunctionArray[InputsT, OutputsT],
):
    inputs: InputsT = dataclasses.MISSING
    outputs: OutputsT = dataclasses.MISSING
