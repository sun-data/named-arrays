from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractDirectionalMatrixArray',
    'DirectionalMatrixArray',
]

DirectionT = TypeVar('DirectionT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDirectionalMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractDirectionalVectorArray,
):
    @property
    @abc.abstractmethod
    def direction(self) -> na.AbstractVectorArray:
        """
        The `direction` component of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractDirectionalMatrixArray]:
        return AbstractDirectionalMatrixArray

    @property
    def type_explicit(self) -> Type[DirectionalMatrixArray]:
        return DirectionalMatrixArray

    @property
    def type_vector(self) -> Type[na.DirectionalVectorArray]:
        return na.DirectionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class DirectionalMatrixArray(
    na.DirectionalVectorArray,
    AbstractDirectionalMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[DirectionT],
):
    pass
