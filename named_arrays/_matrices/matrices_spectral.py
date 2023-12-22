from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np

__all__ = [
    'AbstractSpectralMatrixArray',
    'SpectralMatrixArray',
]

WavelengthT = TypeVar('WavelengthT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractSpectralVectorArray,
):
    @property
    @abc.abstractmethod
    def wavelength(self) -> na.AbstractVectorArray:
        """
        The `wavelength` component of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractSpectralMatrixArray]:
        return AbstractSpectralMatrixArray

    @property
    def type_explicit(self) -> Type[SpectralMatrixArray]:
        return SpectralMatrixArray

    @property
    def type_vector(self) -> Type[na.SpectralVectorArray]:
        return na.SpectralVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        return np.abs(self.wavelength)


@dataclasses.dataclass(eq=False, repr=False)
class SpectralMatrixArray(
    na.SpectralVectorArray,
    AbstractSpectralMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[WavelengthT],
):
    pass
