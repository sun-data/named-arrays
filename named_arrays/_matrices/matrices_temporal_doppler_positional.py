from __future__ import annotations
from typing import Type
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractTemporalDopplerPositionalMatrixArray",
    "TemporalDopplerPositionalMatrixArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalDopplerPositionalMatrixArray(
    na.AbstractTemporalDopplerPositionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractTemporalDopplerPositionalMatrixArray]:
        return AbstractTemporalDopplerPositionalMatrixArray

    @property
    def type_explicit(self) -> Type[TemporalDopplerPositionalMatrixArray]:
        return TemporalDopplerPositionalMatrixArray

    @property
    def type_vector(self) -> Type[na.TemporalDopplerPositionalVectorArray]:
        return na.TemporalDopplerPositionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:     # pragma: nocover
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class TemporalDopplerPositionalMatrixArray(
    na.TemporalDopplerPositionalVectorArray,
    AbstractTemporalDopplerPositionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass
