from __future__ import annotations
from typing import TypeVar, Generic, Iterator, Type
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import numba
import astropy.units as u
import named_arrays as na
from ._transformations import (
    AbstractTranslation,
    Translation,
    AbstractLinearTransformation,
    LinearTransformation,
    AffineTransformation, MatrixT, AbstractTransformation,
)


__all__ = [
    "Cartesian3dTranslation",
    "AbstractCartesian3dLinearTransformation",
    "AbstractCartesian3dRotation",
    "Cartesian3dRotationX",
    "Cartesian3dRotationY",
    "Cartesian3dRotationZ",
]


def _dot_3d(
    a: na.AbstractCartesian3dVectorArray,
    b: na.AbstractCartesian3dVectorArray,
) -> na.AbstractScalarArray:
    """
    Compute the dot product between two 3-dimensional vectors.

    Parameters
    ----------
    a
        The first vector operand.
    b
        The second vector operand.
    """

    ax = a.x
    ay = a.y
    az = a.z

    bx = b.x
    by = b.y
    bz = b.z

    unit_a = na.unit(ax)
    unit_b = na.unit(bx)

    if unit_a is not None:
        ax = (ax << unit_a).value
        ay = (ay << unit_a).value
        az = (az << unit_a).value

    if unit_b is not None:
        bx = (bx << unit_b).value
        by = (by << unit_b).value
        bz = (bz << unit_b).value

    result = _dot_3d_numexpr(ax, ay, az, bx, by, bz)

    # shape = na.shape_broadcasted(ax, ay, az, bx, by, bz)
    #
    # result = _dot_3d_ndarray(
    #     ax.ndarray_aligned(shape),
    #     ay.ndarray_aligned(shape),
    #     az.ndarray_aligned(shape),
    #     bx.ndarray_aligned(shape),
    #     by.ndarray_aligned(shape),
    #     bz.ndarray_aligned(shape),
    # )
    #
    # result = na.ScalarArray(result, axes=tuple(shape))

    if unit_a is not None:
        if unit_b is not None:
            unit = unit_a * unit_b
        else:
            unit = unit_a
    else:
        if unit_b is not None:
            unit = unit_b
        else:
            unit = None

    return result << unit


def _dot_3d_ndarray(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> np.ndarray:

    ax, ay, az, bx, by, bz = np.broadcast_arrays(ax, ay, az, bx, by, bz)

    shape = ax.shape

    result = _dot_3d_njit(
        ax.reshape(-1),
        ay.reshape(-1),
        az.reshape(-1),
        bx.reshape(-1),
        by.reshape(-1),
        bz.reshape(-1),
    )

    result = result.reshape(shape)

    return result


@numba.njit(parallel=True, fastmath=True)
def _dot_3d_njit(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> np.ndarray:

    num, = ax.shape

    result = np.empty(num)

    for i in numba.prange(num):
        result[i] = ax[i] * bx[i] + ay[i] * by[i] + az[i] * bz[i]

    return result


@numba.vectorize(
    ["float64(float64,float64,float64,float64,float64,float64)"],
    cache=True,
    target="parallel",
    fastmath=True,
)
def _dot_3d_vectorized(
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
) -> float:
    return ax * bx + ay * by + az * bz


def _dot_3d_numexpr(
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
) -> float:
    return na.numexpr.evaluate("ax * bx + ay * by + az * bz")


def _matvec_3d(
    a: na.AbstractCartesian3dMatrixArray,
    b: na.AbstractCartesian3dVectorArray,
) -> na.Cartesian3dVectorArray:
    """
    Matrix-vector dot product for 3-dimensional Cartesian vectors.

    Parameters
    ----------
    a
        Matrix operand.
    b
        Vector operand.
    """
    return na.Cartesian3dVectorArray(
        x=_dot_3d(a.x, b),
        y=_dot_3d(a.y, b),
        z=_dot_3d(a.z, b),
    )

def _vecmat_3d(
    a: na.AbstractCartesian3dVectorArray,
    b_T: na.AbstractCartesian3dMatrixArray,
) -> na.Cartesian3dVectorArray:
    """
    Vector-matrix dot product for 3-dimensional Cartesian vectors.

    Parameters
    ----------
    a
        Vector operand.
    b
        Transposed matrix operand.

        This is provided instead of the untransposed version so that repeated
        calls to this function don't invoke multiple calls to :func:`transpose`.
    """
    return na.Cartesian3dVectorArray(
        x=_dot_3d(a, b_T.x),
        y=_dot_3d(a, b_T.y),
        z=_dot_3d(a, b_T.z),
    )


def _transpose_3d(
    a: na.AbstractCartesian3dMatrixArray,
) -> na.Cartesian3dMatrixArray:
    """
    Take the transpose of a 3-dimensional Cartesian matrix.

    Parameters
    ----------
    a
        The matrix to transpose.
    """

    ax = a.x
    ay = a.y
    az = a.z

    return na.Cartesian3dMatrixArray(
        x=na.Cartesian3dVectorArray(ax.x, ay.x, az.x),
        y=na.Cartesian3dVectorArray(ax.y, ay.y, az.y),
        z=na.Cartesian3dVectorArray(ax.z, ay.z, az.z),
    )


def _matmul_3d(
    a: na.AbstractCartesian3dMatrixArray,
    b: na.AbstractCartesian3dMatrixArray,
) -> na.Cartesian3dMatrixArray:
    """
    Matrix product for 3-dimensional Cartesian matrices

    Parameters
    ----------
    a
        Left matrix operand.
    b
        Right matrix operand.
    """

    b_T = _transpose_3d(b)

    return na.Cartesian3dMatrixArray(
        x=_vecmat_3d(a.x, b_T),
        y=_vecmat_3d(a.y, b_T),
        z=_vecmat_3d(a.z, b_T),
    )


@dataclasses.dataclass(eq=False)
class AbstractCartesian3dTransformation(
    AbstractTransformation,
):
    """An arbitrary transformation of a 3D vector."""


@dataclasses.dataclass(eq=False)
class Cartesian3dTranslation(
    AbstractCartesian3dTransformation,
    AbstractTranslation,
):
    """
    A translation in a 3D Cartesian space.
    """

    x: na.ScalarLike = 0 * u.mm
    """The :math:`x` component of this translation."""

    y: na.ScalarLike = 0 * u.mm
    """The :math:`y` component of this translation."""

    z: na.ScalarLike = 0 * u.mm
    """The :math:`z` component of this translation."""

    @property
    def type_concrete(self) -> Type[Cartesian3dTranslation]:
        return Cartesian3dTranslation

    @classmethod
    def from_vector(
        cls: Type[Self],
        vector: na.AbstractCartesian3dVectorArray,
    ) -> Self:
        return cls(
            x=vector.x,
            y=vector.y,
            z=vector.z,
        )

    @property
    def vector(self) -> na.Cartesian3dVectorArray:
        return na.Cartesian3dVectorArray(self.x, self.y, self.z)

    def __call__(
        self,
        a: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        return na.Cartesian3dVectorArray(
            x=self.x + a.x,
            y=self.y + a.y,
            z=self.z + a.z,
        )

    @property
    def inverse(self: Self) -> Self:
        return Cartesian3dTranslation(
            x=-self.x,
            y=-self.y,
            z=-self.z,
        )

    def __matmul__(
        self,
        other: Cartesian3dTranslation,
    ) -> Cartesian3dTranslation:
        if isinstance(other, Cartesian3dTranslation):
            return Cartesian3dTranslation(
                x=self.x + other.x,
                y=self.y + other.y,
                z=self.z + other.z,
            )
        else:
            return NotImplemented


@dataclasses.dataclass(eq=False)
class AbstractCartesian3dLinearTransformation(
    AbstractCartesian3dTransformation,
    AbstractLinearTransformation,
):
    """
    An interface describing an arbitrary linear transformation of a 3D Cartesian vector.
    """

    @property
    def type_concrete(self) -> Type[Cartesian3dLinearTransformation]:
        return Cartesian3dLinearTransformation

    @property
    @abc.abstractmethod
    def matrix(self) -> na.AbstractCartesian3dMatrixArray:
        """
        The underlying matrix representation of this transformation.
        """

    def __call__(
        self,
        a: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        return _matvec_3d(self.matrix, a)

    def __matmul__(
        self,
        other: AbstractCartesian3dLinearTransformation | Cartesian3dTranslation,
    ) -> Cartesian3dLinearTransformation | AffineTransformation:
        print("here")
        if isinstance(other, AbstractCartesian3dOrthogonalTransformation):
            return Cartesian3dOrthogonalTransformation(
                _matmul_3d(self.matrix, other.matrix),
            )
        elif isinstance(other, Cartesian3dTranslation):
            return AffineTransformation(
                transformation_linear=self,
                translation=Cartesian3dTranslation.from_vector(
                    self.matrix @ other.vector,
                ),
            )
        else:
            return NotImplemented

    def __rmatmul__(
            self,
            other: Cartesian3dTranslation,
    ) -> AffineTransformation:
        if isinstance(other, Cartesian3dTranslation):
            return AffineTransformation(
                transformation_linear=self,
                translation=other,
            )
        else:
            return NotImplemented


@dataclasses.dataclass(eq=False)
class Cartesian3dLinearTransformation(
    AbstractCartesian3dLinearTransformation,
):
    """
    An arbitrary linear transformation of a 3D Cartesian vector.
    """
    matrix: na.AbstractCartesian3dMatrixArray = dataclasses.MISSING

    @classmethod
    def from_matrix(
        cls,
        m: na.AbstractCartesian3dMatrixArray,
    ) -> Cartesian3dLinearTransformation:
        return cls(matrix=m)


@dataclasses.dataclass(eq=False)
class AbstractCartesian3dOrthogonalTransformation(
    AbstractCartesian3dLinearTransformation,
):
    """
    An orthogonal transformation of a 3-dimensional Cartesian vector.
    """

    @property
    def type_concrete(self) -> Type[Cartesian3dOrthogonalTransformation]:
        return Cartesian3dOrthogonalTransformation

    @property
    def inverse(self: Self) -> Self:
        return Cartesian3dOrthogonalTransformation(
            matrix=_transpose_3d(self.matrix),
        )
    #
    # def __matmul__(
    #     self,
    #     other: AbstractCartesian3dTransformation | AbstractCartesian3dOrthogonalTransformation,
    # ) -> Cartesian3dOrthogonalTransformation | Cartesian3dLinearTransformation | AffineTransformation:
    #     if isinstance(other, AbstractCartesian3dOrthogonalTransformation):
    #         return self.type_concrete.from_matrix()


@dataclasses.dataclass(eq=False)
class Cartesian3dOrthogonalTransformation(
    AbstractCartesian3dOrthogonalTransformation,
    Cartesian3dLinearTransformation,
):
    """A concrete implementation of a 3-dimensional orthogonal transformation."""


@dataclasses.dataclass(eq=False)
class AbstractCartesian3dRotation(
    AbstractCartesian3dOrthogonalTransformation,
):
    """
    An interface describing an arbitrary rotation in a 3D Cartesian space.
    """

    angle: na.ScalarLike = 0 * u.deg
    """The angle of rotation."""

    @classmethod
    @abc.abstractmethod
    def _matrix_type(cls) -> type[na.AbstractCartesian3dRotationMatrixArray]:
        """to be used by subclasses to specify which rotation matrix to use"""

    @property
    def matrix(self) -> na.AbstractCartesian3dRotationMatrixArray:
        return self._matrix_type()(self.angle).to(u.one)

    def __call__(
        self,
        a: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        return _matvec_3d(self.matrix, a)

    @property
    def inverse(self: Self) -> Self:
        return dataclasses.replace(self, angle=-self.angle)


@dataclasses.dataclass(eq=False)
class Cartesian3dRotationX(
    AbstractCartesian3dRotation
):
    """A rotation about the $x$ axis."""
    def _matrix_type(cls):
        return na.Cartesian3dXRotationMatrixArray


@dataclasses.dataclass(eq=False)
class Cartesian3dRotationY(
    AbstractCartesian3dRotation
):
    """A rotation about the $y$ axis."""
    def _matrix_type(cls):
        return na.Cartesian3dYRotationMatrixArray


@dataclasses.dataclass(eq=False)
class Cartesian3dRotationZ(
    AbstractCartesian3dRotation
):
    """A rotation about the $z$ axis."""
    def _matrix_type(cls):
        return na.Cartesian3dZRotationMatrixArray
