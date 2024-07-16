from __future__ import annotations
from typing import TypeVar, Type, Generic
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
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

    def volume_cell(
        self,
        axis: None | tuple[str, str, str],
    ) -> na.AbstractScalar:

        shape = self.shape

        if axis is None:
            axis = tuple(shape)
        else:
            if not set(axis).issubset(shape):
                raise ValueError(
                    f"{axis=} should be a subset of {self.shape=}."
                )

        result = 0

        ax, ay, az = axis

        axes_face = [
            (ax, ay, az),
            (ay, az, ax),
            (az, ax, ay),
        ]

        for axis_face in axes_face:
            a1, a2, a3 = axis_face
            face = [
                self[{a1: slice(None, ~0), a2: slice(None, ~0)}],
                self[{a1: slice(+1, None), a2: slice(None, ~0)}],
                self[{a1: slice(+1, None), a2: slice(+1, None)}],
                self[{a1: slice(None, ~0), a2: slice(+1, None)}],
            ]
            triangles = [
                [face[0], face[1], face[2]],
                [face[2], face[3], face[0]],
            ]
            for triangle in triangles:
                v1, v2, v3 = triangle
                vol = v1 @ v2.cross(v3)
                result = result - vol[{a3: slice(None, ~0)}]
                result = result + vol[{a3: slice(+1, None)}]

        result = result / 6

        return result

    @classmethod
    def _sold_angle(
        cls,
        a: na.AbstractCartesian3dVectorArray,
        b: na.AbstractCartesian3dVectorArray,
        c: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:

        numerator = a @ b.cross(c)

        a_ = a.length
        b_ = b.length
        c_ = c.length

        d0 = a_ * b_ * c_
        d1 = (a @ b) * c_
        d2 = (a @ c) * b_
        d3 = (b @ c) * a_
        denomerator = d0 + d1 + d2 + d3

        unit = numerator.unit

        if unit is not None:
            numerator = numerator.to(unit).value
            denomerator = denomerator.to(unit).value

        angle = 2 * np.arctan2(numerator, denomerator)

        return angle << u.sr

    def solid_angle_cell(
        self,
        axis: None | tuple[str, str] = None,
    ) -> na.AbstractScalar:
        r"""
        Compute the solid angle of each cell formed by interpreting this
        array as a logically-rectangular 2D grid of vertices.

        Note that this method is usually only used for sorted arrays

        Parameters
        ----------
        axis
            The two axes defining the logically-rectangular 2D grid.
            If :obj:`None` (the default), :attr:`axes` is used and must have
            only two elements.

        Notes
        -----
        The solid angle :math:`\Omega` of a triangle formed by the vertices
        :math:`\vec{a}`, :math:`\vec{b}`, and :math:`\vec{c}` is given by
        :cite:t:`Eriksson1990` as

        .. math::

            \tan \left( \frac{1}{2} \Omega \right)
                = \frac{\vec{a} \cdot (\vec{b} \times \vec{c})}
                    {a b c + (\vec{a} \cdot \vec{b}) c + (\vec{a} \cdot \vec{c}) b + (\vec{b} \cdot \vec{c}) a}.

        Each rectangular cell is decomposed into two triangles and then the
        solid angle of each triangle is computed.
        """

        shape = self.shape

        if axis is None:
            axis = tuple(shape)
        else:
            if not set(axis).issubset(shape):   # pragma: nocover
                raise ValueError(
                    f"{axis=} should be a subset of {self.shape=}."
                )

        ax, ay = axis

        s0 = slice(None, ~0)
        s1 = slice(+1, None)

        a = self[{ax: s0, ay: s0}]
        b = self[{ax: s1, ay: s0}]
        c = self[{ax: s1, ay: s1}]
        d = self[{ax: s0, ay: s1}]

        angle_1 = self._sold_angle(a, b, c)
        angle_2 = self._sold_angle(c, d, a)

        return angle_1 + angle_2

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
