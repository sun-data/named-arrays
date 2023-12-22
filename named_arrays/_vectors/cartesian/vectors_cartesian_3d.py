from __future__ import annotations
from typing import TypeVar, Type, Generic
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractCartesian3dVectorArray',
    'Cartesian3dVectorArray',
    'AbstractImplicitCartesian3dVectorArray',
    'AbstractCartesian3dVectorRandomSample',
    'Cartesian3dVectorUniformRandomSample',
    'Cartesian3dVectorNormalRandomSample',
    'AbstractParameterizedCartesian3dVectorArray',
    'Cartesian3dVectorArrayRange',
    'AbstractCartesian3dVectorSpace',
    'Cartesian3dVectorLinearSpace',
    'Cartesian3dVectorStratifiedRandomSpace',
    'Cartesian3dVectorLogarithmicSpace',
    'Cartesian3dVectorGeometricSpace',
]

XT = TypeVar('XT', bound=na.ArrayLike)
YT = TypeVar('YT', bound=na.ArrayLike)
ZT = TypeVar('ZT', bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian3dVectorArray(
    na.AbstractCartesian2dVectorArray,
):

    @property
    @abc.abstractmethod
    def z(self: Self) -> na.ArrayLike:
        """
        The `z` component of the vector.
        """

    @property
    def xy(self) -> na.Cartesian2dVectorArray:
        return na.Cartesian2dVectorArray(
            x=self.x,
            y=self.y,
        )

    @property
    def type_abstract(self: Self) -> Type[AbstractCartesian3dVectorArray]:
        return AbstractCartesian3dVectorArray

    @property
    def type_explicit(self: Self) -> Type[Cartesian3dVectorArray]:
        return Cartesian3dVectorArray

    @property
    def type_matrix(self) -> Type[na.Cartesian3dMatrixArray]:
        return na.Cartesian3dMatrixArray

    @property
    def explicit(self) -> Cartesian3dVectorArray:
        return super().explicit

    def cross(
            self,
            other: AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        r"""
        Compute the vector product :math:`\mathbf{a} \times \mathbf{b}`

        Parameters
        ----------
        other
            the right-hand operand of the cross product operation
        """
        a = self
        b = other
        if isinstance(other, na.AbstractCartesian3dVectorArray):
            if len(self.cartesian_nd.components) != 3:
                raise ValueError("all components of `self` must be scalars")
            if len(other.cartesian_nd.components) != 3:
                raise ValueError("all components of `other` must be scalars")
            return self.type_explicit(
                x=+(a.y * b.z - a.z * b.y),
                y=-(a.x * b.z - a.z * b.x),
                z=+(a.x * b.y - a.y * b.x),
            )
        else:
            raise TypeError(
                f"`other` must be an instance of `{na.AbstractCartesian3dVectorArray.__name__}`, "
                f"got `{type(other).__name__}`"
            )


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorArray(
    AbstractCartesian3dVectorArray,
    na.Cartesian2dVectorArray,
    Generic[XT, YT, ZT],
):
    x: XT = 0
    y: YT = 0
    z: ZT = 0

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> Cartesian3dVectorArray:
        result = super().from_scalar(scalar, like=like)
        result.z = scalar
        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitCartesian3dVectorArray(
    AbstractCartesian3dVectorArray,
    na.AbstractImplicitCartesian2dVectorArray,
):

    @property
    def x(self) -> na.ArrayLike:
        return self.explicit.x

    @property
    def y(self) -> na.ArrayLike:
        return self.explicit.y

    @property
    def z(self) -> na.ArrayLike:
        return self.explicit.z


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian3dVectorRandomSample(
    AbstractImplicitCartesian3dVectorArray,
    na.AbstractCartesian2dVectorRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorUniformRandomSample(
    AbstractCartesian3dVectorRandomSample,
    na.Cartesian2dVectorUniformRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorNormalRandomSample(
    AbstractCartesian3dVectorRandomSample,
    na.Cartesian2dVectorNormalRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedCartesian3dVectorArray(
    AbstractImplicitCartesian3dVectorArray,
    na.AbstractParameterizedCartesian2dVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorArrayRange(
    AbstractParameterizedCartesian3dVectorArray,
    na.Cartesian2dVectorArrayRange,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian3dVectorSpace(
    AbstractParameterizedCartesian3dVectorArray,
    na.AbstractCartesian2dVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorLinearSpace(
    AbstractCartesian3dVectorSpace,
    na.Cartesian2dVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorStratifiedRandomSpace(
    Cartesian3dVectorLinearSpace,
    na.Cartesian2dVectorStratifiedRandomSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorLogarithmicSpace(
    AbstractCartesian3dVectorSpace,
    na.Cartesian2dVectorLogarithmicSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian3dVectorGeometricSpace(
    AbstractCartesian3dVectorSpace,
    na.Cartesian2dVectorGeometricSpace,
):
    pass
