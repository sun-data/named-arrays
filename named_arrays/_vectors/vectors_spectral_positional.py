from __future__ import annotations
from typing import Type, TypeVar, Sequence
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractSpectralPositionalVectorArray",
    "SpectralPositionalVectorArray",
    "AbstractImplicitSpectralPositionalVectorArray",
    "AbstractParameterizedSpectralPositionalVectorArray",
    "AbstractSpectralPositionalVectorSpace",
    "SpectralPositionalVectorLinearSpace",
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

    def volume_cell(self, axis: None | str | Sequence[str]) -> na.AbstractScalar:
        """
        The volume of each voxel of the logically-rectangular grid formed by
        this array: the wavelength bin width times the area of each position
        cell.

        The wavelength and position axes are determined from `axis` by which
        component of this array they belong to, so their order does not matter.

        Parameters
        ----------
        axis
            The grid axes: the axis of changing wavelength together with the two
            axes of changing position.
            If :obj:`None`, all the axes of this array are used.
        """
        axis = na.axis_normalized(self, axis)

        shape_wavelength = na.shape(self.wavelength)
        shape_position = na.shape(self.position)

        # split `axis` by which component each axis belongs to, ordering each
        # group by the component's own axes so the result does not depend on the
        # order in which the axes were given.
        axis_wavelength = tuple(a for a in shape_wavelength if a in axis)
        axis_position = tuple(
            a for a in shape_position if a in axis and a not in shape_wavelength
        )

        volume_wavelength = self.wavelength.explicit.volume_cell(axis_wavelength)
        volume_position = na.as_named_array(
            self.position.explicit.volume_cell(axis_position)
        )

        # if the position varies with wavelength, its cell area spans the
        # wavelength edges; collapse it onto the wavelength cell centers so it
        # aligns with the wavelength bin widths.
        for a in axis_wavelength:
            if a in na.shape(volume_position):
                volume_position = volume_position.cell_centers(a)

        return volume_wavelength * volume_position


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionalVectorArray(
    AbstractSpectralPositionalVectorArray,
    na.PositionalVectorArray[PositionT],
    na.SpectralVectorArray[WavelengthT],
):
    pass


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
