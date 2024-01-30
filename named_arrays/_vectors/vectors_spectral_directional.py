from __future__ import annotations
from typing import Type, TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractSpectralDirectionalVectorArray',
    'SpectralDirectionalVectorArray',
    'AbstractImplicitSpectralDirectionalVectorArray',
    'AbstractParameterizedSpectralDirectionalVectorArray',
    'AbstractSpectralDirectionalVectorSpace',
    'SpectralDirectionalVectorLinearSpace',
]

DirectionT = TypeVar("DirectionT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralDirectionalVectorArray(
    na.AbstractDirectionalVectorArray,
    na.AbstractSpectralVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractSpectralDirectionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return SpectralDirectionalVectorArray

    @property
    def type_matrix(self) -> Type[na.SpectralDirectionalMatrixArray]:
        return na.SpectralDirectionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class SpectralDirectionalVectorArray(
    AbstractSpectralDirectionalVectorArray,
    na.DirectionalVectorArray[DirectionT],
    na.SpectralVectorArray[WavelengthT],
):

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> SpectralDirectionalVectorArray:
        return cls(wavelength=scalar, direction=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitSpectralDirectionalVectorArray(
    AbstractSpectralDirectionalVectorArray,
    na.AbstractImplicitDirectionalVectorArray,
    na.AbstractImplicitSpectralVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedSpectralDirectionalVectorArray(
    AbstractImplicitSpectralDirectionalVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralDirectionalVectorSpace(
    AbstractParameterizedSpectralDirectionalVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class SpectralDirectionalVectorLinearSpace(
    AbstractSpectralDirectionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
