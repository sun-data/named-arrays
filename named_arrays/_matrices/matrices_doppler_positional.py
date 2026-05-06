from __future__ import annotations
from typing import Type
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractDopplerPositionalMatrixArray",
    "DopplerPositionalMatrixArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDopplerPositionalMatrixArray(
    na.AbstractDopplerPositionalVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractDopplerPositionalMatrixArray]:
        return AbstractDopplerPositionalMatrixArray

    @property
    def type_explicit(self) -> Type[DopplerPositionalMatrixArray]:
        return DopplerPositionalMatrixArray

    @property
    def type_vector(self) -> Type[na.DopplerPositionalVectorArray]:
        return na.DopplerPositionalVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class DopplerPositionalMatrixArray(
    na.DopplerPositionalVectorArray,
    AbstractDopplerPositionalMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass