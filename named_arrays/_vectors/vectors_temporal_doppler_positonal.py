from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na
from named_arrays._core import _required

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
    na.AbstractDopplerPositionalVectorArray,
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
    na.DopplerPositionalVectorArray[PositionT, WavelengthT],
    na.TemporalVectorArray[TimeT]
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalDopplerPositionalVectorArray(
    AbstractTemporalDopplerPositionalVectorArray,
    na.AbstractImplicitDopplerPositionalVectorArray,
    na.AbstractImplicitTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalWcsDopplerPositionalVectorArray(
    AbstractImplicitTemporalDopplerPositionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractScalar = _required()
    wavelength_rest: na.AbstractScalar = _required()
    crval: na.AbstractSpectralPositionalVectorArray = _required()
    crpix: na.AbstractCartesianNdVectorArray = _required()
    cdelt: na.AbstractSpectralPositionalVectorArray = _required()
    pc: na.AbstractSpectralPositionalMatrixArray = _required()
    shape_wcs: dict[str, int] = _required()

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(
            time=self.time,
            wavelength_rest=self.wavelength_rest,
        )
