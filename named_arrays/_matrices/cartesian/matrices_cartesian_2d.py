from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import numpy as np
import named_arrays as na
from named_arrays._core import _required

__all__ = [
    "AbstractCartesian2dMatrixArray",
    "Cartesian2dMatrixArray",
    "AbstractImplicitCartesian2dMatrixArray",
    "Cartesian2dIdentityMatrixArray",
    "AbstractCartesian2dRotationMatrixArray",
    "Cartesian2dRotationMatrixArray",
]

XT = TypeVar('XT', bound=na.AbstractVectorArray, covariant=True)
YT = TypeVar('YT', bound=na.AbstractVectorArray, covariant=True)


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

        a, b = self.x.components.values()
        c, d = self.y.components.values()

        chi = a + d
        delta = a * d - b * c

        t1 = chi / 2
        discriminant = np.square(chi) / 4 - delta

        # astropy does not implement numpy.emath, so given a Quantity it
        # warns and then drops the units, which are applied here instead
        unit = na.unit(discriminant)
        if unit is None:
            t2 = np.emath.sqrt(discriminant)
        else:
            t2 = np.emath.sqrt(discriminant.to_value(unit)) * unit ** (1 / 2)

        # the eigenvalue of the greater magnitude, and the step from it to
        # the other one
        greater = np.abs(t1 + t2) >= np.abs(t1 - t2)
        e1 = np.where(greater, t1 + t2, t1 - t2)
        e2 = np.where(greater, t1 - t2, t1 + t2)
        h = e2 - e1

        p1 = e1 ** z
        p2 = e2 ** z

        # The divided difference of the power at the two eigenvalues. Where
        # they are close, the difference of their powers cancels, so it is
        # found from the logarithm of their ratio instead, and where they are
        # equal, it is its limit, the derivative of the power.
        with np.errstate(divide="ignore", invalid="ignore"):
            # each of these divides by zero, or takes the logarithm of a
            # negative number, only where it is not used
            difference_far = (p2 - p1) / h
            difference_near = p1 * np.expm1(z * np.log1p(h / e1)) / h
        difference = np.where(
            np.abs(h) < np.abs(e1) / 2,
            difference_near,
            difference_far,
        )
        difference = np.where(h == 0, z * e1 ** (z - 1), difference)

        identity = na.Cartesian2dIdentityMatrixArray()

        # the polynomial which interpolates the power at the eigenvalues in
        # the form of Newton, which still holds where they are equal
        return difference * self + (p1 - difference * e1) * identity


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dMatrixArray(
    na.Cartesian2dVectorArray[XT, YT],
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
    angle: na.ScalarLike = _required()

    @property
    def explicit(self) -> na.Cartesian2dMatrixArray:
        a = self.angle
        return na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(x=np.cos(a), y=-np.sin(a)),
            y=na.Cartesian2dVectorArray(x=np.sin(a), y=np.cos(a)),
        )
