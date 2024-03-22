from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np

__all__ = [
    "AbstractInputOutputMatrixArray",
    "InputOutputMatrixArray",
]

WavelengthT = TypeVar("WavelengthT", bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInputOutputMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractInputOutputVectorArray,
):
    @property
    @abc.abstractmethod
    def input(self) -> na.AbstractVectorArray:
        """
        The `input` component of the matrix.
        """

    @property
    @abc.abstractmethod
    def output(self) -> na.AbstractVectorArray:
        """
        The `output` component of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractInputOutputMatrixArray]:
        return AbstractInputOutputMatrixArray

    @property
    def type_explicit(self) -> Type[InputOutputMatrixArray]:
        return InputOutputMatrixArray

    @property
    def type_vector(self) -> Type[na.InputOutputVectorArray]:
        return na.InputOutputVectorArray

    @property
    def determinant(self) -> na.ScalarLike:     # pragma: nocover
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class InputOutputMatrixArray(
    na.InputOutputVectorArray,
    AbstractInputOutputMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[WavelengthT],
):
    pass
