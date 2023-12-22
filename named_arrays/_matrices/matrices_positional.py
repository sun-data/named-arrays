from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np

__all__ = [
    'AbstractPositionalMatrixArray',
    'PositionalMatrixArray',
]

PositionT = TypeVar('PositionT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPositionalMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractPositionalVectorArray,
):
    @property
    @abc.abstractmethod
    def position(self) -> na.AbstractVectorArray:
        """
        The `position` component of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractPositionalMatrixArray]:
        return AbstractPositionalMatrixArray

    @property
    def type_explicit(self) -> Type[PositionalMatrixArray]:
        return PositionalMatrixArray

    @property
    def type_vector(self) -> Type[na.PositionalVectorArray]:
        return na.PositionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class PositionalMatrixArray(
    na.PositionalVectorArray,
    AbstractPositionalMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[PositionT],
):
    pass
