from __future__ import annotations
from typing import Type, TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractSpectralPositionalVectorArray',
    'SpectralPositionalVectorArray',
    'AbstractImplicitSpectralPositionalVectorArray',
    'AbstractParameterizedSpectralPositionalVectorArray',
    'AbstractSpectralPositionalVectorSpace',
    'SpectralPositionalVectorLinearSpace',
]

PositionT = TypeVar("PositionT", bound=na.ArrayLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionalVectorArray(
    na.AbstractPositionalVectorArray,
    na.AbstractSpectralVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractSpectralPositionalVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return SpectralPositionalVectorArray

    @property
    def type_matrix(self) -> Type[na.SpectralPositionalMatrixArray]:
        return na.SpectralPositionalMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionalVectorArray(
    AbstractSpectralPositionalVectorArray,
    na.PositionalVectorArray[PositionT],
    na.SpectralVectorArray[WavelengthT],
):

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> SpectralPositionalVectorArray:
        return cls(wavelength=scalar, position=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitSpectralPositionalVectorArray(
    AbstractSpectralPositionalVectorArray,
    na.AbstractImplicitPositionalVectorArray,
    na.AbstractImplicitSpectralVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedSpectralPositionalVectorArray(
    AbstractImplicitSpectralPositionalVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionalVectorSpace(
    AbstractParameterizedSpectralPositionalVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionalVectorLinearSpace(
    AbstractSpectralPositionalVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
