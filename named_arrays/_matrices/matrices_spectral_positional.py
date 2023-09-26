from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np
import astropy.units as u

__all__ = [
    'AbstractSpectralPositionalMatrixArray',
    'SpectralPositionalMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionalMatrixArray(
    na.AbstractSpectralPositionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractSpectralPositionalMatrixArray]:
        return AbstractSpectralPositionalMatrixArray

    @property
    def type_explicit(self) -> Type[SpectralPositionalMatrixArray]:
        return SpectralPositionalMatrixArray

    @property
    def type_vector(self) -> Type[na.SpectralPositionalVectorArray]:
        return na.SpectralPositionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionalMatrixArray(
    na.SpectralPositionalVectorArray,
    AbstractSpectralPositionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass