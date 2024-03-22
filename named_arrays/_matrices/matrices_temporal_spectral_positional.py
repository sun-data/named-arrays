from __future__ import annotations
from typing import Type

import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalSpectralPositionalMatrixArray',
    'TemporalSpectralPositionalMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralPositionalMatrixArray(
    na.AbstractTemporalSpectralPositionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractTemporalSpectralPositionalMatrixArray]:
        return AbstractTemporalSpectralPositionalMatrixArray

    @property
    def type_explicit(self) -> Type[TemporalSpectralPositionalMatrixArray]:
        return TemporalSpectralPositionalMatrixArray

    @property
    def type_vector(self) -> Type[na.TemporalSpectralPositionalVectorArray]:
        return na.TemporalSpectralPositionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:     # pragma: nocover
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralPositionalMatrixArray(
    na.TemporalSpectralPositionalVectorArray,
    AbstractTemporalSpectralPositionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass
