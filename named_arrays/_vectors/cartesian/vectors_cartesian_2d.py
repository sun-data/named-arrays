from __future__ import annotations
from typing import TypeVar, Type, Generic
import numpy as np
from typing_extensions import Self
import abc
import dataclasses
import astropy.units as u
import named_arrays as na

__all__ = [
    'AbstractCartesian2dVectorArray',
    'Cartesian2dVectorArray',
    'AbstractImplicitCartesian2dVectorArray',
    'AbstractCartesian2dVectorRandomSample',
    'Cartesian2dVectorUniformRandomSample',
    'Cartesian2dVectorNormalRandomSample',
    'AbstractParameterizedCartesian2dVectorArray',
    'Cartesian2dVectorArrayRange',
    'AbstractCartesian2dVectorSpace',
    'Cartesian2dVectorLinearSpace',
    'Cartesian2dVectorStratifiedRandomSpace',
    'Cartesian2dVectorLogarithmicSpace',
    'Cartesian2dVectorGeometricSpace',
]

XT = TypeVar('XT', bound=na.ArrayLike)
YT = TypeVar('YT', bound=na.ArrayLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dVectorArray(
    na.AbstractCartesianVectorArray,
):

    @property
    @abc.abstractmethod
    def x(self: Self) -> na.ArrayLike:
        """
        The `x` component of the vector.
        """

    @property
    @abc.abstractmethod
    def y(self: Self) -> na.ArrayLike:
        """
        The `y` component of the vector.
        """

    @property
    def type_abstract(self: Self) -> Type[AbstractCartesian2dVectorArray]:
        return AbstractCartesian2dVectorArray

    @property
    def type_explicit(self: Self) -> Type[Cartesian2dVectorArray]:
        return Cartesian2dVectorArray

    @property
    def type_matrix(self) -> Type[na.Cartesian2dMatrixArray]:
        return na.Cartesian2dMatrixArray


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorArray(
    AbstractCartesian2dVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[XT, YT],
):
    x: XT = 0
    y: YT = 0

    @classmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.AbstractScalar,
            like: None | na.AbstractExplicitVectorArray = None,
    ) -> Cartesian2dVectorArray:
        result = super().from_scalar(scalar, like=like)
        if result is not NotImplemented:
            return result

        return cls(x=scalar, y=scalar)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitCartesian2dVectorArray(
    AbstractCartesian2dVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def x(self) -> na.ArrayLike:
        return self.explicit.x

    @property
    def y(self) -> na.ArrayLike:
        return self.explicit.y


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dVectorRandomSample(
    AbstractImplicitCartesian2dVectorArray,
    na.AbstractCartesianVectorRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorUniformRandomSample(
    AbstractCartesian2dVectorRandomSample,
    na.AbstractCartesianVectorUniformRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorNormalRandomSample(
    AbstractCartesian2dVectorRandomSample,
    na.AbstractCartesianVectorNormalRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedCartesian2dVectorArray(
    AbstractImplicitCartesian2dVectorArray,
    na.AbstractParameterizedCartesianVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorArrayRange(
    AbstractParameterizedCartesian2dVectorArray,
    na.AbstractCartesianVectorArrayRange,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dVectorSpace(
    AbstractParameterizedCartesian2dVectorArray,
    na.AbstractCartesianVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorLinearSpace(
    AbstractCartesian2dVectorSpace,
    na.AbstractCartesianVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorStratifiedRandomSpace(
    Cartesian2dVectorLinearSpace,
    na.AbstractCartesianVectorStratifiedRandomSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorLogarithmicSpace(
    AbstractCartesian2dVectorSpace,
    na.AbstractCartesianVectorLogarithmicSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorGeometricSpace(
    AbstractCartesian2dVectorSpace,
    na.AbstractCartesianVectorGeometricSpace,
):
    pass
