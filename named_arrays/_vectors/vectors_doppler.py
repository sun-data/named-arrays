from __future__ import annotations
from typing import Type, Generic, TypeVar
import abc
import dataclasses
import astropy.units as u
import astropy.constants
import named_arrays as na

__all__ = [
    "AbstractDopplerVectorArray",
    "DopplerVectorArray",
    "AbstractImplicitDopplerVectorArray",
]

WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDopplerVectorArray(
    na.AbstractSpectralVectorArray,
):
    """An interface describing a vector with a Doppler-shifted wavelength component."""

    @property
    @abc.abstractmethod
    def wavelength_rest(self) -> na.ArrayLike:
        """
        The rest wavelength.
        """

    @property
    def type_abstract(self) -> Type[na.AbstractArray]:
        return AbstractDopplerVectorArray

    @property
    def type_explicit(self) -> Type[na.AbstractExplicitArray]:
        return DopplerVectorArray

    @property
    def type_matrix(self) -> Type[na.DopplerMatrixArray]:
        return na.DopplerMatrixArray

    @property
    def velocity(self) -> na.ArrayLike:
        """The line-of-sight velocity of the wave emitter."""
        w = self.wavelength
        w0 = self.wavelength_rest
        result = astropy.constants.c * (w / w0 - 1)
        return result.to(u.km / u.s)


@dataclasses.dataclass(eq=False, repr=False)
class DopplerVectorArray(
    AbstractDopplerVectorArray,
    na.SpectralVectorArray,
    Generic[WavelengthT],
):
    """A vector with a Doppler-shifted wavelength component."""

    wavelength_rest: WavelengthT = 0
    """The rest wavelength."""

    @classmethod
    def from_velocity(
        cls,
        velocity: na.ArrayLike,
        wavelength_rest: na.ArrayLike,
        **kwargs,
    ):
        """
        Create a new instance of this class given a line-of-sight velocity
        and a rest wavelength.

        Parameters
        ----------
        velocity
            The line-of-sight velocity of the wave emitter.
        wavelength_rest
            The rest wavelength.
        kwargs
            Additional keyword arguments passed to the constructor of this class.
        """
        return cls(
            wavelength=(1 + velocity / astropy.constants.c) * wavelength_rest,
            wavelength_rest=wavelength_rest,
            **kwargs,
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitDopplerVectorArray(
    AbstractDopplerVectorArray,
    na.AbstractImplicitSpectralVectorArray,
):
    """A vector with an implicit Doppler-shifted wavelength component."""

    @property
    def wavelength_rest(self) -> na.ArrayLike:
        return self.explicit.wavelength_rest
