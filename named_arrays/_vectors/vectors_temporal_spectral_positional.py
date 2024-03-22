from __future__ import annotations
from typing import Type, TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalSpectralPositionalVectorArray',
    'TemporalSpectralPositionalVectorArray',
    'AbstractImplicitTemporalSpectralPositionalVectorArray',
    'AbstractParameterizedTemporalSpectralPositionalVectorArray',
    'AbstractTemporalSpectralPositionalVectorSpace',
    'TemporalSpectralPositionalVectorLinearSpace',
    'ExplicitTemporalWcsSpectralPositionalVectorArray',
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
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
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

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> TemporalSpectralPositionalVectorArray:
        return cls(time=scalar, wavelength=scalar, position=scalar)


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
class ExplicitTemporalWcsSpectralPositionalVectorArray(
    AbstractImplicitTemporalSpectralPositionalVectorArray,
    na.AbstractWcsVector,
):
    time: na.AbstractScalar = dataclasses.MISSING
    crval: AbstractTemporalSpectralPositionalVectorArray = dataclasses.MISSING
    crpix: na.AbstractCartesianNdVectorArray = dataclasses.MISSING
    cdelt: AbstractTemporalSpectralPositionalVectorArray = dataclasses.MISSING
    pc: na.AbstractTemporalSpectralPositionalMatrixArray = dataclasses.MISSING
    shape_wcs: dict[str, int] = dataclasses.MISSING

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
