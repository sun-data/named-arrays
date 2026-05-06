from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractTemporalDopplerPositionalVectorArray",
    "TemporalDopplerPositionalVectorArray",
    "AbstractImplicitTemporalDopplerPositionalVectorArray",
    "ExplicitTemporalWcsDopplerPositionalVectorArray",
]

TimeT = TypeVar("TimeT", bound=na.ArrayLike)
PositionT = TypeVar("PositionT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalDopplerPositionalVectorArray(
    na.AbstractPositionalVectorArray,
    na.AbstractDopplerVectorArray,
    na.AbstractTemporalVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractTemporalDopplerPositionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return TemporalDopplerPositionalVectorArray

    @property
    def type_matrix(self) -> Type[na.AbstractMatrixArray]:
        return na.TemporalDopplerPositionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class TemporalDopplerPositionalVectorArray(
    AbstractTemporalDopplerPositionalVectorArray,
    na.PositionalVectorArray[PositionT],
    na.DopplerVectorArray[WavelengthT],
    na.TemporalVectorArray[TimeT]
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalDopplerPositionalVectorArray(
    AbstractTemporalDopplerPositionalVectorArray,
    na.AbstractImplicitPositionalVectorArray,
    na.AbstractImplicitDopplerVectorArray,
    na.AbstractImplicitTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalWcsDopplerPositionalVectorArray(
    AbstractImplicitTemporalDopplerPositionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractScalar = dataclasses.MISSING
    wavelength_rest: na.AbstractScalar = dataclasses.MISSING
    crval: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    crpix: na.AbstractCartesianNdVectorArray = dataclasses.MISSING
    cdelt: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    pc: na.AbstractSpectralPositionalMatrixArray = dataclasses.MISSING
    shape_wcs: dict[str, int] = dataclasses.MISSING

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(
            time=self.time,
            wavelength_rest=self.wavelength_rest,
        )
