from __future__ import annotations
from typing import Type, TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractTemporalSpectralVectorArray',
    'TemporalSpectralVectorArray',
    'AbstractImplicitTemporalSpectralVectorArray',
    'AbstractParameterizedTemporalSpectralVectorArray',
    'AbstractTemporalSpectralVectorSpace',
    'TemporalSpectralVectorLinearSpace',
    'ExplicitTemporalWcsSpectralVectorArray',
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
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
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

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> TemporalSpectralVectorArray:
        return cls(time=scalar, wavelength=scalar)


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
    time: na.AbstractScalar = dataclasses.MISSING
    crval: AbstractTemporalSpectralVectorArray = dataclasses.MISSING
    crpix: na.AbstractCartesianNdVectorArray = dataclasses.MISSING
    cdelt: AbstractTemporalSpectralVectorArray = dataclasses.MISSING
    pc: na.AbstractTemporalSpectralMatrixArray = dataclasses.MISSING
    shape_wcs: dict[str, int] = dataclasses.MISSING

    @property
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        return dict(time=self.time)
