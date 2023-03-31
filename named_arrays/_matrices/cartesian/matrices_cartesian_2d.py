from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractCartesian2dMatrixArray',
    'Cartesian2dMatrixArray',
]

XT = TypeVar('XT', bound=na.AbstractVectorArray)
YT = TypeVar('YT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dMatrixArray(
    na.AbstractCartesianMatrixArray,
    na.AbstractCartesian2dVectorArray,
):
    @property
    @abc.abstractmethod
    def x(self) -> na.AbstractVectorArray:
        """
        The `x` row of the matrix.
        """

    @property
    @abc.abstractmethod
    def y(self) -> na.AbstractVectorArray:
        """
        The `y` row of the matrix.
        """

    @property
    def type_array_abstract(self) -> Type[AbstractCartesian2dMatrixArray]:
        return AbstractCartesian2dMatrixArray

    @property
    def type_explicit(self) -> Type[Cartesian2dMatrixArray]:
        return Cartesian2dMatrixArray

    @property
    def type_vector(self) -> Type[na.Cartesian2dVectorArray]:
        return na.Cartesian2dVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        if not self.is_square:
            raise ValueError("can only compute determinant of square matrices")
        xx, xy = self.x.components.values()
        yx, yy = self.y.components.values()
        return xx * yy - xy * yx

    @property
    def inverse(self) -> na.AbstractMatrixArray:
        if not self.is_square:
            raise ValueError("can only compute inverse of square matrices")
        type_matrix = self.x.type_matrix
        type_row = self.type_vector
        c1, c2 = self.x.components
        xx, xy = self.x.components.values()
        yx, yy = self.y.components.values()
        result = type_matrix.from_components({
            c1: type_row(x=yy, y=-xy),
            c2: type_row(x=-yx, y=xx),
        })
        result = result / self.determinant
        return result


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dMatrixArray(
    na.Cartesian2dVectorArray,
    AbstractCartesian2dMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[XT, YT],
):
    pass



