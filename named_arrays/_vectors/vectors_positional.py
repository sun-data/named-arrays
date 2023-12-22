from __future__ import annotations
from typing import Type, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractPositionalVectorArray',
    'PositionalVectorArray',
    'AbstractImplicitPositionalVectorArray',
    'AbstractParameterizedPositionalVectorArray',
    'AbstractPositionalVectorSpace',
    'PositionalVectorLinearSpace',

]

PositionT = TypeVar("PositionT", bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPositionalVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    @abc.abstractmethod
    def position(self) -> na.ArrayLike:
        """
        The `position` component of the vector.
        """

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractPositionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return PositionalVectorArray

    @property
    def type_matrix(self) -> Type[na.PositionalMatrixArray]:
        return na.PositionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class PositionalVectorArray(
    AbstractPositionalVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[PositionT],
):
    position: PositionT = 0

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> PositionalVectorArray:
        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result
        return cls(position=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitPositionalVectorArray(
    AbstractPositionalVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def position(self) -> na.ArrayLike:
        return self.explicit.position


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedPositionalVectorArray(
    AbstractImplicitPositionalVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPositionalVectorSpace(
    AbstractParameterizedPositionalVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class PositionalVectorLinearSpace(
    AbstractPositionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
