from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na
from named_arrays._core import _required

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
    def type_explicit(self) -> Type[TemporalPositionalVectorArray]:
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
    pass


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
    time: na.AbstractExplicitScalarArray = _required()
    crval: na.AbstractPositionalVectorArray = _required()
    crpix: na.AbstractCartesianNdVectorArray = _required()
    cdelt: na.AbstractPositionalVectorArray = _required()
    pc: na.AbstractPositionalMatrixArray = _required()
    shape_wcs: dict[str, int] = _required()

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
