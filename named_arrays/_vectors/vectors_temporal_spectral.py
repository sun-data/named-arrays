from __future__ import annotations
from typing import Type, TypeVar
import dataclasses
import named_arrays as na
from named_arrays._core import _required

__all__ = [
    "AbstractTemporalSpectralVectorArray",
    "TemporalSpectralVectorArray",
    "AbstractImplicitTemporalSpectralVectorArray",
    "AbstractParameterizedTemporalSpectralVectorArray",
    "AbstractTemporalSpectralVectorSpace",
    "TemporalSpectralVectorLinearSpace",
    "ExplicitTemporalWcsSpectralVectorArray",
]

TimeT = TypeVar("TimeT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralVectorArray(
    na.AbstractSpectralVectorArray,
    na.AbstractTemporalVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractTemporalSpectralVectorArray

    @property
    def type_explicit(self) -> Type[TemporalSpectralVectorArray]:
        return TemporalSpectralVectorArray

    @property
    def type_matrix(self) -> Type[na.TemporalSpectralMatrixArray]:
        return na.TemporalSpectralMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralVectorArray(
    AbstractTemporalSpectralVectorArray,
    na.SpectralVectorArray[WavelengthT],
    na.TemporalVectorArray
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitTemporalSpectralVectorArray(
    AbstractTemporalSpectralVectorArray,
    na.AbstractImplicitSpectralVectorArray,
    na.AbstractImplicitTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedTemporalSpectralVectorArray(
    AbstractImplicitTemporalSpectralVectorArray,
    na.AbstractParameterizedSpectralVectorArray,
    na.AbstractParameterizedTemporalVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTemporalSpectralVectorSpace(
    AbstractParameterizedTemporalSpectralVectorArray,
    na.AbstractSpectralVectorSpace,
    na.AbstractTemporalVectorSpace
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class TemporalSpectralVectorLinearSpace(
    AbstractTemporalSpectralVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ExplicitTemporalWcsSpectralVectorArray(
    AbstractImplicitTemporalSpectralVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractExplicitScalarArray = _required()
    crval: AbstractTemporalSpectralVectorArray = _required()
    crpix: na.AbstractCartesianNdVectorArray = _required()
    cdelt: AbstractTemporalSpectralVectorArray = _required()
    pc: na.AbstractTemporalSpectralMatrixArray = _required()
    shape_wcs: dict[str, int] = _required()

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
