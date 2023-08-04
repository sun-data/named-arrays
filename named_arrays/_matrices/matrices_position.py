from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np

__all__ = [
    'AbstractPositionMatrixArray',
    'PositionMatrixArray',
]

PositionT = TypeVar('PositionT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPositionMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractPositionVectorArray,
):
    @property
    @abc.abstractmethod
    def position(self) -> na.AbstractVectorArray:
        """
        The `position` component of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractPositionMatrixArray]:
        return AbstractPositionMatrixArray

    @property
    def type_explicit(self) -> Type[PositionMatrixArray]:
        return PositionMatrixArray

    @property
    def type_vector(self) -> Type[na.PositionVectorArray]:
        return na.PositionVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        raise NotImplementedError

    @property
    def inverse(self) -> na.AbstractMatrixArray:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class PositionMatrixArray(
    na.PositionVectorArray,
    AbstractPositionMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[PositionT],
):
    pass
