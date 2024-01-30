from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalMatrixArray',
    'TemporalMatrixArray',
]

TimeT = TypeVar('TimeT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractTemporalVectorArray,
):
    @property
    @abc.abstractmethod
    def time(self) -> na.AbstractVectorArray:
        """
        The temporal component of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractTemporalMatrixArray]:
        return AbstractTemporalMatrixArray

    @property
    def type_explicit(self) -> Type[TemporalMatrixArray]:
        return TemporalMatrixArray

    @property
    def type_vector(self) -> Type[na.TemporalVectorArray]:
        return na.TemporalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class TemporalMatrixArray(
    na.TemporalVectorArray,
    AbstractTemporalMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[TimeT],
):
    pass
