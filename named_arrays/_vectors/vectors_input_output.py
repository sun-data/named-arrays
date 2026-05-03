from __future__ import annotations
from typing import Type, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractInputOutputVectorArray",
    "InputOutputVectorArray",
    "AbstractImplicitInputOutputVectorArray",
    "AbstractParameterizedInputOutputVectorArray",
    "AbstractInputOutputVectorSpace",
    "InputOutputVectorLinearSpace",
]

InputT = TypeVar("InputT", bound=na.ArrayLike)
OutputT = TypeVar("OutputT", bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInputOutputVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    @abc.abstractmethod
    def input(self) -> na.ArrayLike:
        """
        The `input` component of the vector.
        """

    @property
    @abc.abstractmethod
    def output(self) -> na.ArrayLike:
        """
        The `output` component of the vector.
        """

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractInputOutputVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return InputOutputVectorArray

    @property
    def type_matrix(self) -> Type[na.InputOutputMatrixArray]:
        return na.InputOutputMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class InputOutputVectorArray(
    AbstractInputOutputVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[InputT, OutputT],
):
    input: InputT = 0
    output: OutputT = 0


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitInputOutputVectorArray(
    AbstractInputOutputVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def input(self) -> na.ArrayLike:
        return self.explicit.input

    @property
    def output(self) -> na.ArrayLike:
        return self.explicit.output


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedInputOutputVectorArray(
    AbstractImplicitInputOutputVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInputOutputVectorSpace(
    AbstractParameterizedInputOutputVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class InputOutputVectorLinearSpace(
    AbstractInputOutputVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
