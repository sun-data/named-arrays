from __future__ import annotations
from typing import Type
from typing_extensions import Self
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractSpectralPositionVectorArray',
    'SpectralPositionVectorArray',
    'AbstractImplicitSpectralPositionVectorArray',
    'AbstractParameterizedSpectralPositionVectorArray',
    'AbstractSpectralPositionVectorSpace',
    'SpectralPositionVectorLinearSpace',
]

@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionVectorArray(
    na.AbstractPositionVectorArray,
    na.AbstractSpectralVectorArray,
):

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractSpectralPositionVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return SpectralPositionVectorArray

    @property
    def type_matrix(self) -> Type[na.SpectralPositionMatrixArray]:
        return na.SpectralPositionMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionVectorArray(
    AbstractSpectralPositionVectorArray,
    na.PositionVectorArray,
    na.SpectralVectorArray,
):

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> SpectralPositionVectorArray:
        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result
        return cls(wavelength=scalar, position=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitSpectralPositionVectorArray(
    AbstractSpectralPositionVectorArray,
    na.AbstractImplicitPositionVectorArray,
    na.AbstractImplicitSpectralVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedSpectralPositionVectorArray(
    AbstractImplicitSpectralPositionVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionVectorSpace(
    AbstractParameterizedSpectralPositionVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionVectorLinearSpace(
    AbstractSpectralPositionVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass