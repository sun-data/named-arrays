from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractCartesianNdMatrixArray",
    "CartesianNdMatrixArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianNdMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractCartesianNdVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractCartesianNdMatrixArray]:
        return AbstractCartesianNdMatrixArray

    @property
    def type_explicit(self) -> Type[CartesianNdMatrixArray]:
        return CartesianNdMatrixArray

    @property
    def type_vector(self) -> Type[na.CartesianNdVectorArray]:
        return na.CartesianNdVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        return NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class CartesianNdMatrixArray(
    na.CartesianNdVectorArray,
    AbstractCartesianNdMatrixArray,
    na.AbstractExplicitCartesianMatrixArray,
):
    pass
