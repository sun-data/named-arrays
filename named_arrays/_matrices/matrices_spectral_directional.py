from __future__ import annotations
from typing import Type

import dataclasses
import named_arrays as na

__all__ = [
    'AbstractSpectralDirectionalMatrixArray',
    'SpectralDirectionalMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralDirectionalMatrixArray(
    na.AbstractSpectralDirectionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractSpectralDirectionalMatrixArray]:
        return AbstractSpectralDirectionalMatrixArray

    @property
    def type_explicit(self) -> Type[SpectralDirectionalMatrixArray]:
        return SpectralDirectionalMatrixArray

    @property
    def type_vector(self) -> Type[na.SpectralDirectionalVectorArray]:
        return na.SpectralDirectionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class SpectralDirectionalMatrixArray(
    na.SpectralDirectionalVectorArray,
    AbstractSpectralDirectionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass
