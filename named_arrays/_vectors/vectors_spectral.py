from __future__ import annotations
from typing import Type, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractSpectralVectorArray',
    'SpectralVectorArray',
    'AbstractImplicitSpectralVectorArray',
    'AbstractParameterizedSpectralVectorArray',
    'AbstractSpectralVectorSpace',
    'SpectralVectorLinearSpace',

]

WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    @abc.abstractmethod
    def wavelength(self) -> na.ArrayLike:
        """
        The `wavelength` component of the vector.
        """

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractSpectralVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return SpectralVectorArray

    @property
    def type_matrix(self) -> Type[na.SpectralMatrixArray]:
        return na.SpectralMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class SpectralVectorArray(
    AbstractSpectralVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[WavelengthT],
):
    wavelength: WavelengthT = 0

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> SpectralVectorArray:
        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result
        return cls(wavelength=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitSpectralVectorArray(
    AbstractSpectralVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def wavelength(self) -> na.ArrayLike:
        return self.explicit.wavelength


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedSpectralVectorArray(
    AbstractImplicitSpectralVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralVectorSpace(
    AbstractParameterizedSpectralVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class SpectralVectorLinearSpace(
    AbstractSpectralVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass
