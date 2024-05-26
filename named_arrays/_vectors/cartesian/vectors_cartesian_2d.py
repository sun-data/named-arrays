from __future__ import annotations
from typing import TypeVar, Type, Generic, Sequence
import math
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

    def volume_cell(self, axis: None | tuple[str, str]) -> na.AbstractScalar:

        if axis is None:
            if self.ndim != 2:
                raise ValueError(
                    f"If {axis=}, then {self.ndim=} must be two-dimensional"
                )
            axis = self.axes

        if not set(axis).issubset(self.shape):
            raise ValueError(
                f"{axis=} should be a subset of {self.shape=}."
            )

        a1, a2 = axis

        slices = [
            {a1: slice(None, ~0), a2: slice(None, ~0)},
            {a1: slice(+1, None), a2: slice(None, ~0)},
            {a1: slice(+1, None), a2: slice(+1, None)},
            {a1: slice(None, ~0), a2: slice(+1, None)},
        ]

        array = self.broadcasted
        x = array.x
        y = array.y

        if not isinstance(x, na.AbstractScalar):    # pragma: nocover
            raise TypeError(
                f"{type(self.x)=} must be a scalar."
            )

        if not isinstance(y, na.AbstractScalar):    # pragma: nocover
            raise TypeError(
                f"{type(self.y)=} must be a scalar."
            )

        x = [x[s] for s in slices]
        y = [y[s] for s in slices]

        result = 0
        n = len(slices)
        for i in range(n):
            result = result + y[i] * (x[i - 1] - x[(i + 1) % n])

        return result / 2


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
    def volume_cell(self, axis: None | Sequence[str]) -> na.AbstractScalar:

        components = self.components

        axis = na.axis_normalized(self, axis)
        if len(axis) != len(components):
            raise ValueError(
                f"{axis=} must have exactly two elements"
            )

        shape_random = self.shape_random
        if set(axis).issubset(shape_random):
            start = na.asanyarray(self.start, like=self)
            stop = na.asanyarray(self.stop, like=self)
            span = stop - start
            size = math.prod(shape_random[ax] for ax in axis)
            result = math.prod(span.components.values()) / size
        else:
            result = super().volume_cell(axis)

        return result


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
    def volume_cell(self, axis: None | Sequence[str]) -> na.AbstractScalar:

        components = self.components

        axis = na.axis_normalized(self, axis)
        if len(axis) != len(components):
            raise ValueError(
                f"{axis=} must have exactly two elements"
            )

        if set(axis).issubset(self.axis.components.values()):
            result = self.step
            result = math.prod(result.components.values())
        else:
            result = super().volume_cell(axis)

        return result


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
