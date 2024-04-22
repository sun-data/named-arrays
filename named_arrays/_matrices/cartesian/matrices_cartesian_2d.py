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

    def power(
        self,
        exponent: float | na.AbstractScalar,
    ) -> Cartesian2dMatrixArray:
        """
        Compute this matrix raised to the power of a given exponent

        Parameters
        ----------
        exponent
            The power to raise this matrix to.
        """

        z = exponent
        y = z - 1

        a, b = self.x.components.values()
        c, d = self.y.components.values()

        chi = a + d
        delta = a * d - b * c

        t1 = chi / 2
        t2 = np.emath.sqrt(np.square(chi) / 4 - delta)

        e1 = t1 + t2
        e2 = t1 - t2

        d = e1 - e2

        identity = na.Cartesian2dIdentityMatrixArray()

        a1 = (e1 ** z - e2 ** z) / d * self
        a2 = e1 * e2 * (e1 ** y - e2 ** y) / d * identity

        return a1 - a2


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
    na.AbstractImplicitMatrixArray,
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

    def power(
        self,
        exponent: float | na.AbstractScalar,
    ) -> AbstractCartesian2dMatrixArray:
        return self


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
