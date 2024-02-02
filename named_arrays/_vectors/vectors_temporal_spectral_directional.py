from __future__ import annotations
from typing import Type, TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalSpectralDirectionalVectorArray',
    'TemporalSpectralDirectionalVectorArray',
    'AbstractImplicitTemporalSpectralDirectionalVectorArray',
    'AbstractParameterizedTemporalSpectralDirectionalVectorArray',
    'AbstractTemporalSpectralDirectionalVectorSpace',
    'TemporalSpectralDirectionalVectorLinearSpace',
    'ExplicitTemporalWcsSpectralDirectionalVectorArray',
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
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
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

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> TemporalSpectralDirectionalVectorArray:
        return cls(time=scalar, wavelength=scalar, direction=scalar)


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
    time: na.AbstractScalar = dataclasses.MISSING
    crval: AbstractTemporalSpectralDirectionalVectorArray = dataclasses.MISSING
    crpix: na.AbstractCartesianNdVectorArray = dataclasses.MISSING
    cdelt: AbstractTemporalSpectralDirectionalVectorArray = dataclasses.MISSING
    pc: na.AbstractTemporalSpectralDirectionalMatrixArray = dataclasses.MISSING
    shape_wcs: dict[str, int] = dataclasses.MISSING

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
