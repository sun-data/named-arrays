from __future__ import annotations
from typing import TypeVar, Generic, Iterator
from typing_extensions import Self
import abc
import dataclasses
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractTransformation",
    "AbstractTranslation",
    "Translation",
    "AbstractLinearTransformation",
    "LinearTransformation",
    "AbstractAffineTransformation",
    "AffineTransformation",
    "AbstractTransformationList",
    "TransformationList",
]

VectorT = TypeVar("VectorT", bound="na.AbstractVectorArray")
MatrixT = TypeVar("MatrixT", bound="na.AbstractMatrixArray")
TransformationT = TypeVar("TransformationT", bound="AbstractTransformation")
LinearTransformationT = TypeVar("LinearTransformationT", bound="AbstractLinearTransformation")
TranslationT = TypeVar("TranslationT", bound="AbstractTranslation")


@dataclasses.dataclass(eq=False)
class AbstractTransformation(
    abc.ABC
):
    """
    An interface for an arbitrary vector transform
    """

    @property
    @abc.abstractmethod
    def shape(self) -> dict[str, int]:
        """The shape of the transformation."""

    @abc.abstractmethod
    def __call__(
            self,
            a: na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:
        """
        apply the transformation to the given vector

        Parameters
        ----------
        a
            vector to transform
        """

    @property
    @abc.abstractmethod
    def inverse(self: Self) -> Self:
        """
        a new transformation that reverses the effect of this transformation
        """

    @abc.abstractmethod
    def __matmul__(
            self,
            other: AbstractTransformation,
    ) -> AbstractTransformation:
        """
        Compose multiple transformations into a single transformation.

        Parameters
        ----------
        other
            another transformation to compose with this one

        Examples
        --------
        Compose two transformations

        .. jupyter-execute::

            import astropy.units as u
            import named_arrays as na

            t1 = na.transformations.Translation(
                vector=na.Cartesian2dVectorArray(x=5) * u.mm,
            )
            t2 = na.transformations.LinearTransformation(
                matrix=na.Cartesian2dRotationArray(53 * u.deg)
            )

            t_composed = t1 @ t2
            t_composed

        use the composed transformation to transform a vector

        .. jupyter-execute::

            v = na.Cartesian3dVectorArray(1, 2, 3) * u.mm

            t_composed(v)

        transform the same vector by applying each transformation separately
        and note that it's the same result as using the composed transformation

        .. jupyter-execute::

            t1(t2(v))
        """


@dataclasses.dataclass(eq=False)
class IdentityTransformation(
    AbstractTransformation,
):
    """
    The identity transformation just returns its inputs.
    """

    @property
    def shape(self) -> dict[str, int]:
        return dict()

    def __call__(
        self,
        a: na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:
        return a

    @property
    def inverse(self: Self) -> Self:
        return self

    def __matmul__(self, other: TransformationT) -> TransformationT:
        return other

    def __rmatmul__(self, other: TransformationT) -> TransformationT:
        return other


@dataclasses.dataclass(eq=False)
class AbstractTranslation(
    AbstractTransformation,
):
    @property
    @abc.abstractmethod
    def vector(self) -> na.AbstractVectorArray:
        """
        the vector representing the translation
        """

    @property
    def shape(self) -> dict[str, int]:
        return self.vector.shape

    def __call__(self, a: na.AbstractVectorArray) -> na.AbstractVectorArray:
        return a + self.vector

    @property
    def inverse(self: Self) -> Self:
        return Translation(vector=-self.vector)

    def __matmul__(self, other: AbstractTranslation) -> AbstractTranslation:
        if isinstance(other, AbstractTranslation):
            return Translation(
                vector=self.vector + other.vector
            )
        else:
            return NotImplemented


@dataclasses.dataclass(eq=False)
class Translation(
    AbstractTranslation,
    Generic[VectorT]
):
    """
    A translation-only vector transformation.

    Examples
    --------

    Translate a vector using a single transformation

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na

        vector = na.Cartesian3dVectorArray(
            x=12 * u.mm,
            y=12 * u.mm,
        )

        transformation = na.transformations.Translation(vector)

        square = na.Cartesian3dVectorArray(
            x=na.ScalarArray([-10, 10, 10, -10, -10] * u.mm, axes="vertex"),
            y=na.ScalarArray([-10, -10, 10, 10, -10] * u.mm, axes="vertex"),
        )

        square_transformed = transformation(square)

        with astropy.visualization.quantity_support():
            plt.figure();
            plt.gca().set_aspect("equal");
            na.plt.plot(square, label="original");
            na.plt.plot(square_transformed, label="translated");
            plt.legend();

    |

    Translate a vector using an array of transformations

    .. jupyter-execute::

        vector_2 = na.Cartesian3dVectorArray(
            x=na.ScalarArray([12, -12] * u.mm, axes="transform"),
            y=9 * u.mm,
        )

        transformation_2 = na.transformations.Translation(vector_2)

        square_transformed_2 = transformation_2(square)

        with astropy.visualization.quantity_support():
            plt.figure();
            plt.gca().set_aspect("equal");
            na.plt.plot(square, label="original");
            na.plt.plot(square_transformed_2, axis="vertex", label="translated");
            plt.legend();
    """
    vector: VectorT = dataclasses.MISSING


@dataclasses.dataclass(eq=False)
class Cartesian3dTranslation(
    AbstractTranslation
):
    x: na.ScalarLike = 0 * u.mm
    y: na.ScalarLike = 0 * u.mm
    z: na.ScalarLike = 0 * u.mm

    @property
    def vector(self) -> na.Cartesian3dVectorArray:
        return na.Cartesian3dVectorArray(self.x, self.y, self.z)


@dataclasses.dataclass(eq=False)
class AbstractLinearTransformation(
    AbstractTransformation,
):
    @property
    @abc.abstractmethod
    def matrix(self) -> na.AbstractMatrixArray:
        """
        the matrix representing the linear transformation
        """

    @property
    def shape(self) -> dict[str, int]:
        return self.matrix.shape

    def __call__(self, a: na.AbstractVectorArray) -> na.AbstractVectorArray:
        return self.matrix @ a

    @property
    def inverse(self: Self) -> Self:
        return LinearTransformation(matrix=self.matrix.inverse)

    def __matmul__(
            self,
            other: AbstractLinearTransformation | AbstractTranslation,
    ) -> LinearTransformation | AffineTransformation:
        if isinstance(other, AbstractLinearTransformation):
            return LinearTransformation(
                self.matrix @ other.matrix,
            )
        elif isinstance(other, AbstractTranslation):
            return AffineTransformation(
                transformation_linear=self,
                translation=Translation(self.matrix @ other.vector),
            )
        else:
            return NotImplemented

    def __rmatmul__(
            self,
            other: AbstractTranslation,
    ) -> AffineTransformation:
        if isinstance(other, AbstractTranslation):
            return AffineTransformation(
                transformation_linear=self,
                translation=other,
            )


@dataclasses.dataclass(eq=False)
class LinearTransformation(
    AbstractLinearTransformation,
    Generic[MatrixT],
):
    """
    A vector transformation represented by a matrix multiplication

    Examples
    --------

    Rotate a vector using a linear transformation

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na

        angle = 53 * u.deg
        matrix = na.Cartesian2dRotationMatrixArray(angle)

        transformation = na.transformations.LinearTransformation(matrix)

        square = na.Cartesian2dVectorArray(
            x=na.ScalarArray([-10, 10, 10, -10, -10] * u.mm, axes="vertex"),
            y=na.ScalarArray([-10, -10, 10, 10, -10] * u.mm, axes="vertex"),
        )

        square_transformed = transformation(square)

        with astropy.visualization.quantity_support():
            plt.figure();
            plt.gca().set_aspect("equal");
            na.plt.plot(square, label="original");
            na.plt.plot(square_transformed, label="rotated");
            plt.legend();

    |

    Rotate a vector using an array of transformations

    .. jupyter-execute::

        angle_2 = na.ScalarArray([30, 45] * u.deg, axes="transform")
        matrix_2 = na.Cartesian2dRotationMatrixArray(angle_2)

        transformation_2 = na.transformations.LinearTransformation(matrix_2)

        square_transformed_2 = transformation_2(square)

        with astropy.visualization.quantity_support():
            plt.figure();
            plt.gca().set_aspect("equal");
            na.plt.plot(square, label="original");
            na.plt.plot(square_transformed_2, axis="vertex", label="rotated");
            plt.legend();
    """
    matrix: MatrixT = dataclasses.MISSING


@dataclasses.dataclass(eq=False)
class AbstractCartesian3dRotation(
    AbstractLinearTransformation
):
    angle: na.ScalarLike = 0 * u.deg

    @classmethod
    @abc.abstractmethod
    def _matrix_type(cls) -> type[na.AbstractCartesian3dRotationMatrixArray]:
        """to be used by subclasses to specify which rotation matrix to use"""

    @property
    def matrix(self) -> na.AbstractCartesian3dRotationMatrixArray:
        return self._matrix_type()(self.angle)


@dataclasses.dataclass(eq=False)
class Cartesian3dRotationX(
    AbstractCartesian3dRotation
):
    def _matrix_type(cls):
        return na.Cartesian3dXRotationMatrixArray


@dataclasses.dataclass(eq=False)
class Cartesian3dRotationY(
    AbstractCartesian3dRotation
):
    def _matrix_type(cls):
        return na.Cartesian3dYRotationMatrixArray


@dataclasses.dataclass(eq=False)
class Cartesian3dRotationZ(
    AbstractCartesian3dRotation
):
    def _matrix_type(cls):
        return na.Cartesian3dZRotationMatrixArray


@dataclasses.dataclass(eq=False)
class AbstractAffineTransformation(
    AbstractTransformation,
):

    @property
    @abc.abstractmethod
    def transformation_linear(self) -> AbstractLinearTransformation:
        """
        The linear transformation component of this affine transformation
        """

    @property
    @abc.abstractmethod
    def translation(self) -> AbstractTranslation:
        """
        the translation component of this affine transformation
        """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            self.transformation_linear.shape,
            self.translation.shape,
        )

    def __call__(self, a: na.AbstractVectorArray) -> na.AbstractVectorArray:
        return self.translation(self.transformation_linear(a))

    @property
    def inverse(self):
        transformation_linear = self.transformation_linear.inverse
        translation = self.translation.inverse
        return AffineTransformation(
            transformation_linear=transformation_linear,
            translation=Translation(transformation_linear(translation.vector)),
        )

    def __matmul__(
            self,
            other: AbstractAffineTransformation | AbstractLinearTransformation | AbstractTranslation
    ) -> AffineTransformation:
        transformation_linear = self.transformation_linear
        translation = self.translation
        if isinstance(other, AbstractAffineTransformation):
            return AffineTransformation(
                transformation_linear=transformation_linear @ other.transformation_linear,
                translation=Translation(transformation_linear.matrix @ other.translation.vector + translation.vector),
            )
        elif isinstance(other, AbstractLinearTransformation):
            return AffineTransformation(
                transformation_linear=transformation_linear @ other,
                translation=translation,
            )
        elif isinstance(other, AbstractTranslation):
            return AffineTransformation(
                transformation_linear=transformation_linear,
                translation=Translation(transformation_linear.matrix @ other.vector + translation.vector)
            )
        else:
            return NotImplemented

    def __rmatmul__(
            self,
            other: AbstractLinearTransformation | AbstractTranslation,
    ) -> AffineTransformation:
        if isinstance(other, AbstractLinearTransformation):
            return AffineTransformation(
                transformation_linear=other @ self.transformation_linear,
                translation=Translation(other.matrix @ self.translation.vector),
            )
        elif isinstance(other, AbstractTranslation):
            return AffineTransformation(
                transformation_linear=self.transformation_linear,
                translation=other @ self.translation,
            )
        else:
            return NotImplemented


@dataclasses.dataclass
class AffineTransformation(
    AbstractAffineTransformation,
    Generic[LinearTransformationT, TranslationT],
):
    transformation_linear: LinearTransformationT = dataclasses.MISSING
    translation: TranslationT = dataclasses.MISSING


@dataclasses.dataclass
class AbstractTransformationList(
    AbstractTransformation,
):
    @property
    @abc.abstractmethod
    def transformations(self) -> list[AbstractTransformation]:
        """
        the underlying list of transformations to compose together
        """

    @property
    @abc.abstractmethod
    def intrinsic(self) -> bool:
        """
        flag controlling whether the transformation should be applied to the
        coordinates or the coordinate system
        """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(*[t.shape for t in self.transformations])

    def __iter__(self) -> Iterator[AbstractTransformation]:
        if self.intrinsic:
            return iter(self.transformations)
        else:
            return reversed(self.transformations)

    @property
    def composed(self) -> AbstractTransformation:
        transformations = list(self)
        result = IdentityTransformation()
        for t in transformations:
            result = t @ result
        return result

    def __call__(self, a: na.AbstractVectorArray) -> na.AbstractVectorArray:
        return self.composed(a)

    @property
    def inverse(self) -> AbstractTransformation:
        return self.composed.inverse

    def __matmul__(self, other: AbstractTransformation):
        return self.composed @ other

    def __rmatmul__(self, other: AbstractTransformation):
        return other @ self.composed


@dataclasses.dataclass
class TransformationList(
    AbstractTransformationList
):
    transformations: list[AbstractTransformation] = dataclasses.MISSING
    intrinsic: bool = True
