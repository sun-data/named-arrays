from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na
from named_arrays._core import _required

__all__ = [
    "AbstractTemporalSpectralDirectionalVectorArray",
    "TemporalSpectralDirectionalVectorArray",
    "AbstractImplicitTemporalSpectralDirectionalVectorArray",
    "AbstractParameterizedTemporalSpectralDirectionalVectorArray",
    "AbstractTemporalSpectralDirectionalVectorSpace",
    "TemporalSpectralDirectionalVectorLinearSpace",
    "ExplicitTemporalWcsSpectralDirectionalVectorArray",
]

TimeT = TypeVar("TimeT", bound=na.ArrayLike)
DirectionT = TypeVar("DirectionT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralDirectionalVectorArray(
    na.AbstractDirectionalVectorArray,
    na.AbstractSpectralVectorArray,
    na.AbstractTemporalVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractTemporalSpectralDirectionalVectorArray

    @property
    def type_explicit(self) -> Type[TemporalSpectralDirectionalVectorArray]:
        return TemporalSpectralDirectionalVectorArray

    @property
    def type_matrix(self) -> Type[na.TemporalSpectralDirectionalMatrixArray]:
        return na.TemporalSpectralDirectionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralDirectionalVectorArray(
    AbstractTemporalSpectralDirectionalVectorArray,
    na.DirectionalVectorArray[DirectionT],
    na.SpectralVectorArray[WavelengthT],
    na.TemporalVectorArray
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalSpectralDirectionalVectorArray(
    AbstractTemporalSpectralDirectionalVectorArray,
    na.AbstractImplicitDirectionalVectorArray,
    na.AbstractImplicitSpectralVectorArray,
    na.AbstractImplicitTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedTemporalSpectralDirectionalVectorArray(
    AbstractImplicitTemporalSpectralDirectionalVectorArray,
    na.AbstractParameterizedDirectionalVectorArray,
    na.AbstractParameterizedSpectralVectorArray,
    na.AbstractParameterizedTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralDirectionalVectorSpace(
    AbstractParameterizedTemporalSpectralDirectionalVectorArray,
    na.AbstractDirectionalVectorSpace,
    na.AbstractSpectralVectorSpace,
    na.AbstractTemporalVectorSpace
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralDirectionalVectorLinearSpace(
    AbstractTemporalSpectralDirectionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalWcsSpectralDirectionalVectorArray(
    AbstractImplicitTemporalSpectralDirectionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractExplicitScalarArray = _required()
    crval: AbstractTemporalSpectralDirectionalVectorArray = _required()
    crpix: na.AbstractCartesianNdVectorArray[na.AbstractExplicitScalarArray] = _required()
    cdelt: AbstractTemporalSpectralDirectionalVectorArray = _required()
    pc: na.AbstractTemporalSpectralDirectionalMatrixArray = _required()
    shape_wcs: dict[str, int] = _required()

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
