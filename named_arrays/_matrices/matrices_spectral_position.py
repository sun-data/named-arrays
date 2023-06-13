from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na
import numpy as np
import astropy.units as u

__all__ = [
    'AbstractSpectralPositionMatrixArray',
    'SpectralPositionMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralPositionMatrixArray(
    na.AbstractSpectralPositionVectorArray,
):

    @property
    def type_abstract(self) -> Type[AbstractSpectralPositionMatrixArray]:
        return AbstractSpectralPositionMatrixArray

    @property
    def type_explicit(self) -> Type[SpectralPositionMatrixArray]:
        return SpectralPositionMatrixArray

    @property
    def type_vector(self) -> Type[na.SpectralPositionVectorArray]:
        return na.SpectralPositionVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        # if not self.is_square:
        #     raise ValueError("can only compute determinant of square matrices")
        # xx, xy = self.x.components.values()
        # yx, yy = self.y.components.values()
        # return xx * yy - xy * yx
        return

    @property
    def inverse(self) -> na.AbstractMatrixArray:
        # if not self.is_square:
        #     raise ValueError("can only compute inverse of square matrices")
        # type_matrix = self.x.type_matrix
        # type_row = self.type_vector
        # c1, c2 = self.x.components
        # xx, xy = self.x.components.values()
        # yx, yy = self.y.components.values()
        # result = type_matrix.from_components({
        #     c1: type_row(x=yy, y=-xy),
        #     c2: type_row(x=-yx, y=xx),
        # })
        # result = result / self.determinant
        # return result
        return


@dataclasses.dataclass(eq=False, repr=False)
class SpectralPositionMatrixArray(
    na.SpectralPositionVectorArray,
    AbstractSpectralPositionMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass