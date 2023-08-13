from __future__ import annotations
from typing import Type, Sequence
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractCartesianNdVectorArray",
    "CartesianNdVectorArray"
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianNdVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractCartesianNdVectorArray]:
        return AbstractCartesianNdVectorArray

    @property
    def type_explicit(self) -> Type[CartesianNdVectorArray]:
        return CartesianNdVectorArray

    @property
    def type_matrix(self) -> Type[na.CartesianNdMatrixArray]:
        return na.CartesianNdMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class CartesianNdVectorArray(
    AbstractCartesianNdVectorArray,
    na.AbstractExplicitCartesianVectorArray,
):
    components: dict[str, na.ArrayLike] = None

    def __post_init__(self):
        if self.components is None:
            self.components = dict()

    @classmethod
    def from_scalar(
            cls,
            scalar: na.ScalarLike,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> CartesianNdVectorArray:

        if like is None:
            raise ValueError("like argument must be specified for CartesianNdArrays")

        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result
        else:
            raise ValueError("all implementations of from_scalar return NotImplemented")



    @classmethod
    def from_components(cls, components: dict[str, na.ArrayLike]) -> na.CartesianNdMatrixArray:
        return cls(components)
