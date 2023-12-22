from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import numpy as np
import named_arrays as na

__all__ = [
    'AbstractCartesian2dMatrixArray',
    'Cartesian2dMatrixArray',
    'AbstractImplicitCartesian2dMatrixArray',
    'Cartesian2dIdentityMatrixArray',
    'AbstractCartesian2dRotationMatrixArray',
    'Cartesian2dRotationMatrixArray'
]

XT = TypeVar('XT', bound=na.AbstractVectorArray)
YT = TypeVar('YT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dMatrixArray(
    na.AbstractCartesian2dVectorArray,
    na.AbstractCartesianMatrixArray,
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
    def type_abstract(self) -> Type[AbstractCartesian2dMatrixArray]:
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


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dMatrixArray(
    na.Cartesian2dVectorArray,
    AbstractCartesian2dMatrixArray,
    na.AbstractExplicitMatrixArray,
    Generic[XT, YT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitCartesian2dMatrixArray(
    na.AbstractImplicitCartesian2dVectorArray,
    AbstractCartesian2dMatrixArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dIdentityMatrixArray(
    AbstractImplicitCartesian2dMatrixArray
):

    @property
    def explicit(self):
        return na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(x=1, y=0),
            y=na.Cartesian2dVectorArray(x=0, y=1),
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dRotationMatrixArray(
    AbstractImplicitCartesian2dMatrixArray,
):

    @property
    @abc.abstractmethod
    def angle(self) -> na.ScalarLike:
        """
        The angle of the rotation matrix.
        """


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dRotationMatrixArray(
    AbstractCartesian2dRotationMatrixArray,
):
    angle: na.ScalarLike = dataclasses.MISSING

    @property
    def explicit(self) -> na.Cartesian2dMatrixArray:
        a = self.angle
        return na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(x=np.cos(a), y=-np.sin(a)),
            y=na.Cartesian2dVectorArray(x=np.sin(a), y=np.cos(a)),
        )
