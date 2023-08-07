from __future__ import annotations
from typing import TypeVar, Generic, Type
import abc
import dataclasses
import numpy as np
import named_arrays as na

__all__ = [
    'AbstractCartesian3dMatrixArray',
    'Cartesian3dMatrixArray',
    'AbstractImplicitCartesian3dMatrixArray',
    'Cartesian3dIdentityMatrixArray',
    'AbstractCartesian3dRotationMatrixArray',
    'Cartesian3dXRotationMatrixArray',
    'Cartesian3dYRotationMatrixArray',
    'Cartesian3dZRotationMatrixArray',
]

XT = TypeVar('XT', bound=na.AbstractVectorArray)
YT = TypeVar('YT', bound=na.AbstractVectorArray)
ZT = TypeVar('ZT', bound=na.AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian3dMatrixArray(
    na.AbstractCartesian3dVectorArray,
    na.AbstractCartesian2dMatrixArray,
):
    @property
    @abc.abstractmethod
    def z(self) -> na.AbstractVectorArray:
        """
        The `z` row of the matrix.
        """

    @property
    def type_abstract(self) -> Type[AbstractCartesian3dMatrixArray]:
        return AbstractCartesian3dMatrixArray

    @property
    def type_explicit(self) -> Type[Cartesian3dMatrixArray]:
        return Cartesian3dMatrixArray

    @property
    def type_vector(self) -> Type[na.Cartesian3dVectorArray]:
        return na.Cartesian3dVectorArray

    @property
    def determinant(self) -> na.ScalarLike:
        if not self.is_square:
            raise ValueError("can only compute determinant of square matrices")
        a, b, c = self.x.components.values()
        d, e, f = self.y.components.values()
        g, h, i = self.z.components.values()
        return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h)

    @property
    def inverse(self) -> na.AbstractMatrixArray:
        if not self.is_square:
            raise ValueError("can only compute inverse of square matrices")
        type_matrix = self.x.type_matrix
        type_row = self.type_vector
        c1, c2, c3 = self.x.components
        a, b, c = self.x.components.values()
        d, e, f = self.y.components.values()
        g, h, i = self.z.components.values()
        result = type_matrix.from_components({
            c1: type_row(
                x=(e * i - f * h),
                y=-(b * i - c * h),
                z=(b * f - c * e),
            ),
            c2: type_row(
                x=-(d * i - f * g),
                y=(a * i - c * g),
                z=-(a * f - c * d),
            ),
            c3: type_row(
                x=(d * h - e * g),
                y=-(a * h - b * g),
                z=(a * e - b * d),
            )
        })
        result = result / self.determinant
        return result


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dMatrixArray(
    na.Cartesian3dVectorArray,
    AbstractCartesian3dMatrixArray,
    na.Cartesian2dMatrixArray,
    Generic[XT, YT, ZT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitCartesian3dMatrixArray(
    AbstractCartesian3dMatrixArray,
    na.AbstractImplicitCartesian2dMatrixArray,
    na.AbstractImplicitCartesian3dVectorArray,
):

    @property
    def z(self) -> na.ArrayLike:
        return self.explicit.z


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dIdentityMatrixArray(
    AbstractImplicitCartesian3dMatrixArray,
    na.Cartesian2dIdentityMatrixArray,
):

    @property
    def explicit(self) -> na.Cartesian3dMatrixArray:
        return Cartesian3dMatrixArray(
            x=na.Cartesian3dVectorArray(x=1, y=0, z=0),
            y=na.Cartesian3dVectorArray(x=0, y=1, z=0),
            z=na.Cartesian3dVectorArray(x=0, y=0, z=1),
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian3dRotationMatrixArray(
    AbstractImplicitCartesian3dMatrixArray,
    na.AbstractCartesian2dRotationMatrixArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dXRotationMatrixArray(
    AbstractCartesian3dRotationMatrixArray,
):
    angle: na.ScalarLike = dataclasses.MISSING

    @property
    def explicit(self) -> Cartesian3dMatrixArray:
        a = self.angle
        return Cartesian3dMatrixArray(
            x=na.Cartesian3dVectorArray(x=1, y=0, z=0),
            y=na.Cartesian3dVectorArray(x=0, y=np.cos(a), z=-np.sin(a)),
            z=na.Cartesian3dVectorArray(x=0, y=np.sin(a), z=np.cos(a)),
        )


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dYRotationMatrixArray(
    AbstractCartesian3dRotationMatrixArray,
):
    angle: na.ScalarLike = dataclasses.MISSING

    @property
    def explicit(self) -> Cartesian3dMatrixArray:
        a = self.angle
        return Cartesian3dMatrixArray(
            x=na.Cartesian3dVectorArray(x=np.cos(a), y=0, z=np.sin(a)),
            y=na.Cartesian3dVectorArray(x=0, y=1, z=0),
            z=na.Cartesian3dVectorArray(x=-np.sin(a), y=0, z=np.cos(a)),
        )


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dZRotationMatrixArray(
    AbstractCartesian3dRotationMatrixArray,
):
    angle: na.ScalarLike = dataclasses.MISSING

    @property
    def explicit(self) -> Cartesian3dMatrixArray:
        a = self.angle
        return Cartesian3dMatrixArray(
            x=na.Cartesian3dVectorArray(x=np.cos(a), y=-np.sin(a), z=0),
            y=na.Cartesian3dVectorArray(x=np.sin(a), y=np.cos(a), z=0),
            z=na.Cartesian3dVectorArray(x=0, y=0, z=1),
        )
