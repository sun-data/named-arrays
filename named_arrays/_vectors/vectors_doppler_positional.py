from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractDopplerPositionalVectorArray",
    "DopplerPositionalVectorArray",
    "AbstractImplicitDopplerPositionalVectorArray",
]

PositionT = TypeVar("PositionT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDopplerPositionalVectorArray(
    na.AbstractPositionalVectorArray,
    na.AbstractDopplerVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractDopplerPositionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return DopplerPositionalVectorArray

    @property
    def type_matrix(self) -> Type[na.AbstractMatrixArray]:
        return na.DopplerPositionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class DopplerPositionalVectorArray(
    AbstractDopplerPositionalVectorArray,
    na.PositionalVectorArray[PositionT],
    na.DopplerVectorArray[WavelengthT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitDopplerPositionalVectorArray(
    AbstractDopplerPositionalVectorArray,
    na.AbstractImplicitPositionalVectorArray,
    na.AbstractImplicitDopplerVectorArray,
):
    pass
