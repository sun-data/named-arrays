from __future__ import annotations
from typing import Type

import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalPositionalMatrixArray',
    'TemporalPositionalMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalPositionalMatrixArray(
    na.AbstractTemporalPositionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractTemporalPositionalMatrixArray]:
        return AbstractTemporalPositionalMatrixArray

    @property
    def type_explicit(self) -> Type[TemporalPositionalMatrixArray]:
        return TemporalPositionalMatrixArray

    @property
    def type_vector(self) -> Type[na.TemporalPositionalVectorArray]:
        return na.TemporalPositionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:     # pragma: nocover
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class TemporalPositionalMatrixArray(
    na.TemporalPositionalVectorArray,
    AbstractTemporalPositionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass
