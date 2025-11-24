from __future__ import annotations
from typing import Type, TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractTemporalPositionalVectorArray",
    "TemporalPositionalVectorArray",
    "AbstractImplicitTemporalPositionalVectorArray",
    "AbstractParameterizedTemporalPositionalVectorArray",
    "AbstractTemporalPositionalVectorSpace",
    "TemporalPositionalVectorLinearSpace",
    "ExplicitTemporalWcsPositionalVectorArray",
]

TimeT = TypeVar("TimeT", bound=na.ArrayLike)
PositionT = TypeVar("PositionT", bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalPositionalVectorArray(
    na.AbstractPositionalVectorArray,
    na.AbstractTemporalVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractTemporalPositionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return TemporalPositionalVectorArray

    @property
    def type_matrix(self) -> Type[na.TemporalPositionalMatrixArray]:
        return na.TemporalPositionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class TemporalPositionalVectorArray(
    AbstractTemporalPositionalVectorArray,
    na.PositionalVectorArray[PositionT],
    na.TemporalVectorArray
):

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> TemporalPositionalVectorArray:
        return cls(time=scalar, position=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalPositionalVectorArray(
    AbstractTemporalPositionalVectorArray,
    na.AbstractImplicitPositionalVectorArray,
    na.AbstractImplicitTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedTemporalPositionalVectorArray(
    AbstractImplicitTemporalPositionalVectorArray,
    na.AbstractParameterizedPositionalVectorArray,
    na.AbstractParameterizedTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalPositionalVectorSpace(
    AbstractParameterizedTemporalPositionalVectorArray,
    na.AbstractPositionalVectorSpace,
    na.AbstractTemporalVectorSpace
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class TemporalPositionalVectorLinearSpace(
    AbstractTemporalPositionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalWcsPositionalVectorArray(
    AbstractImplicitTemporalPositionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractScalar = dataclasses.MISSING
    crval: na.AbstractPositionalVectorArray = dataclasses.MISSING
    crpix: na.AbstractCartesianNdVectorArray = dataclasses.MISSING
    cdelt: na.AbstractPositionalVectorArray = dataclasses.MISSING
    pc: na.AbstractPositionalMatrixArray = dataclasses.MISSING
    shape_wcs: dict[str, int] = dataclasses.MISSING

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
