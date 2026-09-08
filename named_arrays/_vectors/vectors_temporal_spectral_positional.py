from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na
from named_arrays._core import _required

__all__ = [
    "AbstractTemporalSpectralPositionalVectorArray",
    "TemporalSpectralPositionalVectorArray",
    "AbstractImplicitTemporalSpectralPositionalVectorArray",
    "AbstractParameterizedTemporalSpectralPositionalVectorArray",
    "AbstractTemporalSpectralPositionalVectorSpace",
    "TemporalSpectralPositionalVectorLinearSpace",
    "ExplicitTemporalSpectralWcsPositionalVectorArray",
    "ExplicitTemporalWcsSpectralPositionalVectorArray",
]

TimeT = TypeVar("TimeT", bound=na.ArrayLike)
PositionT = TypeVar("PositionT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralPositionalVectorArray(
    na.AbstractPositionalVectorArray,
    na.AbstractSpectralVectorArray,
    na.AbstractTemporalVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractTemporalSpectralPositionalVectorArray

    @property
    def type_explicit(self) -> Type[TemporalSpectralPositionalVectorArray]:
        return TemporalSpectralPositionalVectorArray

    @property
    def type_matrix(self) -> Type[na.TemporalSpectralPositionalMatrixArray]:
        return na.TemporalSpectralPositionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralPositionalVectorArray(
    AbstractTemporalSpectralPositionalVectorArray,
    na.PositionalVectorArray[PositionT],
    na.SpectralVectorArray[WavelengthT],
    na.TemporalVectorArray
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalSpectralPositionalVectorArray(
    AbstractTemporalSpectralPositionalVectorArray,
    na.AbstractImplicitPositionalVectorArray,
    na.AbstractImplicitSpectralVectorArray,
    na.AbstractImplicitTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedTemporalSpectralPositionalVectorArray(
    AbstractImplicitTemporalSpectralPositionalVectorArray,
    na.AbstractParameterizedPositionalVectorArray,
    na.AbstractParameterizedSpectralVectorArray,
    na.AbstractParameterizedTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralPositionalVectorSpace(
    AbstractParameterizedTemporalSpectralPositionalVectorArray,
    na.AbstractPositionalVectorSpace,
    na.AbstractSpectralVectorSpace,
    na.AbstractTemporalVectorSpace
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralPositionalVectorLinearSpace(
    AbstractTemporalSpectralPositionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalSpectralWcsPositionalVectorArray(
    AbstractImplicitTemporalSpectralPositionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractExplicitScalarArray = _required()
    wavelength: na.AbstractExplicitScalarArray = _required()
    crval: na.SpectralPositionalVectorArray[
        na.Cartesian2dVectorArray[na.AbstractExplicitScalarArray, na.AbstractExplicitScalarArray],
        na.AbstractExplicitScalarArray,
    ] = _required()
    crpix: na.CartesianNdVectorArray[na.AbstractExplicitScalarArray] = _required()
    cdelt: na.SpectralPositionalVectorArray[
        na.Cartesian2dVectorArray[na.AbstractExplicitScalarArray, na.AbstractExplicitScalarArray],
        na.AbstractExplicitScalarArray,
    ] = _required()
    pc: na.AbstractSpectralPositionalMatrixArray = _required()
    shape_wcs: dict[str, int] = _required()

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(
            time=self.time,
            wavelength=self.wavelength,
        )


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalWcsSpectralPositionalVectorArray(
    AbstractImplicitTemporalSpectralPositionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractExplicitScalarArray = _required()
    crval: na.SpectralPositionalVectorArray[
        na.Cartesian2dVectorArray[na.AbstractExplicitScalarArray, na.AbstractExplicitScalarArray],
        na.AbstractExplicitScalarArray,
    ] = _required()
    crpix: na.CartesianNdVectorArray[na.AbstractExplicitScalarArray] = _required()
    cdelt: na.SpectralPositionalVectorArray[
        na.Cartesian2dVectorArray[na.AbstractExplicitScalarArray, na.AbstractExplicitScalarArray],
        na.AbstractExplicitScalarArray,
    ] = _required()
    pc: na.AbstractSpectralPositionalMatrixArray = _required()
    shape_wcs: dict[str, int] = _required()

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
