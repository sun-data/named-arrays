from __future__ import annotations
from typing import Type

import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalSpectralDirectionalMatrixArray',
    'TemporalSpectralDirectionalMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralDirectionalMatrixArray(
    na.AbstractTemporalSpectralDirectionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractTemporalSpectralDirectionalMatrixArray]:
        return AbstractTemporalSpectralDirectionalMatrixArray

    @property
    def type_explicit(self) -> Type[TemporalSpectralDirectionalMatrixArray]:
        return TemporalSpectralDirectionalMatrixArray

    @property
    def type_vector(self) -> Type[na.TemporalSpectralDirectionalVectorArray]:
        return na.TemporalSpectralDirectionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralDirectionalMatrixArray(
    na.TemporalSpectralDirectionalVectorArray,
    AbstractTemporalSpectralDirectionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass
