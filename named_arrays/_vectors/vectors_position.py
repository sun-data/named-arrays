from __future__ import annotations
from typing import Type, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractPositionVectorArray',
    'PositionVectorArray',
    'AbstractImplicitPositionVectorArray',
    'AbstractParameterizedPositionVectorArray',
    'AbstractPositionVectorSpace',
    'PositionVectorLinearSpace',

]

PositionT = TypeVar("PositionT", bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPositionVectorArray(
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
        return AbstractPositionVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return PositionVectorArray

    @property
    def type_matrix(self) -> Type[na.PositionMatrixArray]:
        return na.PositionMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class PositionVectorArray(
    AbstractPositionVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[PositionT],
):
    position: PositionT = 0

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> PositionVectorArray:
        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result
        return cls(Position=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitPositionVectorArray(
    AbstractPositionVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def position(self) -> na.ArrayLike:
        return self.explicit.position


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedPositionVectorArray(
    AbstractImplicitPositionVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPositionVectorSpace(
    AbstractParameterizedPositionVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class PositionVectorLinearSpace(
    AbstractPositionVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
