from __future__ import annotations
from typing import Type, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalVectorArray',
    'TemporalVectorArray',
    'AbstractImplicitTemporalVectorArray',
    'AbstractParameterizedTemporalVectorArray',
    'AbstractTemporalVectorSpace',
    'TemporalVectorLinearSpace',

]

TimeT = TypeVar("TimeT", bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    @abc.abstractmethod
    def time(self) -> na.ArrayLike:
        """
        The temporal component of the vector.
        """

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractTemporalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return TemporalVectorArray

    @property
    def type_matrix(self) -> Type[na.TemporalMatrixArray]:
        return na.TemporalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class TemporalVectorArray(
    AbstractTemporalVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[TimeT],
):
    time: TimeT = 0

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> TemporalVectorArray:
        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result
        return cls(time=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalVectorArray(
    AbstractTemporalVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def time(self) -> na.ArrayLike:
        return self.explicit.time


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedTemporalVectorArray(
    AbstractImplicitTemporalVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalVectorSpace(
    AbstractParameterizedTemporalVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class TemporalVectorLinearSpace(
    AbstractTemporalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
