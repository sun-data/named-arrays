from __future__ import annotations
from typing import Type

import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalSpectralMatrixArray',
    'TemporalSpectralMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralMatrixArray(
    na.AbstractTemporalSpectralVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractTemporalSpectralMatrixArray]:
        return AbstractTemporalSpectralMatrixArray

    @property
    def type_explicit(self) -> Type[TemporalSpectralMatrixArray]:
        return TemporalSpectralMatrixArray

    @property
    def type_vector(self) -> Type[na.TemporalSpectralVectorArray]:
        return na.TemporalSpectralVectorArray

    @property
    def determinant(self) -> na.ScalarLike:     # pragma: nocover
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralMatrixArray(
    na.TemporalSpectralVectorArray,
    AbstractTemporalSpectralMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass
