from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np
import astropy.units as u

__all__ = [
    'AbstractSpectralPositionMatrixArray',
    'SpectralPositionMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionMatrixArray(
    na.AbstractSpectralPositionVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractSpectralPositionMatrixArray]:
        return AbstractSpectralPositionMatrixArray

    @property
    def type_explicit(self) -> Type[SpectralPositionMatrixArray]:
        return SpectralPositionMatrixArray

    @property
    def type_vector(self) -> Type[na.SpectralPositionVectorArray]:
        return na.SpectralPositionVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        return NotImplementedError

    @property
    def inverse(self) -> na.AbstractMatrixArray:
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionMatrixArray(
    na.SpectralPositionVectorArray,
    AbstractSpectralPositionMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass