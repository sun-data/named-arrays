from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractDopplerMatrixArray",
    "AbstractDopplerMatrixArray",
    "DopplerMatrixArray",
]

WavelengthT = TypeVar('WavelengthT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDopplerMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractDopplerVectorArray,
):
    @property
    @abc.abstractmethod
    def wavelength_rest(self) -> na.AbstractVectorArray:
        """
        The rest wavelength row of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractDopplerMatrixArray]:
        return AbstractDopplerMatrixArray

    @property
    def type_explicit(self) -> Type[DopplerMatrixArray]:
        return DopplerMatrixArray

    @property
    def type_vector(self) -> Type[na.DopplerVectorArray]:
        return na.DopplerVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class DopplerMatrixArray(
    na.DopplerVectorArray,
    AbstractDopplerMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[WavelengthT],
):
    pass
