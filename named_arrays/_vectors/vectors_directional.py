from __future__ import annotations
from typing import Type, Generic, TypeVar
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractDirectionalVectorArray",
    "DirectionalVectorArray",
    "AbstractImplicitDirectionalVectorArray",
    "AbstractParameterizedDirectionalVectorArray",
    "AbstractDirectionalVectorSpace",
    "DirectionalVectorLinearSpace",
]

DirectionT = TypeVar("DirectionT", bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDirectionalVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    @abc.abstractmethod
    def direction(self) -> na.ArrayLike:
        """
        The `direction` component of the vector.
        """

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractDirectionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return DirectionalVectorArray

    @property
    def type_matrix(self) -> Type[na.DirectionalMatrixArray]:
        return na.DirectionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class DirectionalVectorArray(
    AbstractDirectionalVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[DirectionT],
):
    direction: DirectionT = 0


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitDirectionalVectorArray(
    AbstractDirectionalVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def direction(self) -> na.ArrayLike:
        return self.explicit.direction


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedDirectionalVectorArray(
    AbstractImplicitDirectionalVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDirectionalVectorSpace(
    AbstractParameterizedDirectionalVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class DirectionalVectorLinearSpace(
    AbstractDirectionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
