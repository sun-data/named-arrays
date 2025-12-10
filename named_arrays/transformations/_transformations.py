from __future__ import annotations
from typing import TypeVar, Generic, Iterator, Type
from typing_extensions import Self
import abc
import dataclasses
import numba
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractTransformation",
    "IdentityTransformation",
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
    An interface for an arbitrary vector transformations.
    """

    @property
    @abc.abstractmethod
    def type_concrete(self) -> Type[AbstractTransformation]:
        """
        The default concrete implementation of this transformation.

        This is used to create new instances of this transformation.
        """

    @property
    @abc.abstractmethod
    def shape(self) -> dict[str, int]:
        """The logical shape of the arrays comprising this transformation."""

    @abc.abstractmethod
    def __call__(
        self,
        a: na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:
        """
        Apply this transformation to the given vector.

        Parameters
        ----------
        a
            vector to transform
        """

    @property
    @abc.abstractmethod
    def inverse(self: Self) -> Self:
        """
        A new transformation that reverses the effect of this transformation.
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
            Another transformation to compose with this one.

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
    def type_concrete(self) -> Type[IdentityTransformation]:
        return IdentityTransformation

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
    Generic[VectorT],
):
    """
    An interface describing an arbitrary translation of a vector.
    """

    @property
    def type_concrete(self) -> Type[Translation]:
        return Translation

    @property
    @abc.abstractmethod
    def vector(self) -> VectorT:
        """A vector representing this translation."""

    @classmethod
    @abc.abstractmethod
    def from_vector(cls, v: VectorT) -> AbstractTranslation[VectorT]:
        """Create a new instance of this transformation from a vector."""

    @property
    def shape(self) -> dict[str, int]:
        return self.vector.shape

    def __call__(self, a: na.AbstractVectorArray) -> na.AbstractVectorArray:
        return a + self.vector

    @property
    def inverse(self: Self) -> Self:
        return self.type_concrete.from_vector(-self.vector)

    def __matmul__(self, other: AbstractTranslation) -> AbstractTranslation:
        if isinstance(other, AbstractTranslation):
            return self.type_concrete.from_vector(self.vector + other.vector)
        else:
            return NotImplemented


@dataclasses.dataclass(eq=False)
class Translation(
    AbstractTranslation[VectorT],
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

        vector = na.Cartesian2dVectorArray(
            x=12 * u.mm,
            y=12 * u.mm,
        )

        transformation = na.transformations.Translation(vector)

        square = na.Cartesian2dVectorArray(
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

        vector_2 = na.Cartesian2dVectorArray(
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
    """A vector representing the translation."""

    @classmethod
    def from_vector(cls, v: VectorT) -> Translation[VectorT]:
        return cls(vector=v)


@dataclasses.dataclass(eq=False)
class AbstractLinearTransformation(
    AbstractTransformation,
    Generic[MatrixT],
):
    """
    An interface describing an arbitrary linear transformation.
    """

    @property
    def type_concrete(self) -> Type[LinearTransformation]:
        return LinearTransformation

    @property
    @abc.abstractmethod
    def matrix(self) -> MatrixT:
        """
        The matrix representation of this linear transformation.
        """

    @property
    def shape(self) -> dict[str, int]:
        return self.matrix.shape

    def __call__(self, a: na.AbstractVectorArray) -> na.AbstractVectorArray:
        return self.matrix @ a

    @property
    def inverse(self: Self) -> Self:
        return self.type_concrete(matrix=self.matrix.inverse)

    def __matmul__(
            self,
            other: AbstractLinearTransformation | AbstractTranslation,
    ) -> LinearTransformation | AffineTransformation:
        if isinstance(other, AbstractLinearTransformation):
            return self.type_concrete.from_matrix(self.matrix @ other.matrix)
        elif isinstance(other, AbstractTranslation):
            return AffineTransformation(
                transformation_linear=self,
                translation=other.type_concrete.from_vector(self(other.vector)),
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
        else:
            return NotImplemented


@dataclasses.dataclass(eq=False)
class LinearTransformation(
    AbstractLinearTransformation,
    Generic[MatrixT],
):
    """
    A vector transformation represented by a matrix multiplication.

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

    @classmethod
    def from_matrix(cls, m: MatrixT) -> LinearTransformation[MatrixT]:
        return cls(matrix=m)


@dataclasses.dataclass(eq=False)
class AbstractAffineTransformation(
    AbstractTransformation,
):
    """An interface describing an arbitrary affine transformation."""

    @property
    def type_concrete(self) -> Type[AffineTransformation]:
        return AffineTransformation

    @property
    @abc.abstractmethod
    def transformation_linear(self) -> AbstractLinearTransformation:
        """
        The linear transformation component of this affine transformation.
        """

    @property
    @abc.abstractmethod
    def translation(self) -> AbstractTranslation:
        """
        The translation component of this affine transformation.
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
            translation=translation.from_vector(
                v=transformation_linear(translation.vector),
            ),
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
                translation=translation.from_vector(
                    translation(transformation_linear(other.translation.vector)),
                ),
            )
        elif isinstance(other, AbstractLinearTransformation):
            return AffineTransformation(
                transformation_linear=transformation_linear @ other,
                translation=translation,
            )
        elif isinstance(other, AbstractTranslation):
            return AffineTransformation(
                transformation_linear=transformation_linear,
                translation=translation.from_vector(
                    translation(transformation_linear(other.vector)),
                ),
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
                translation=self.translation.from_vector(
                    other(self.translation.vector),
                ),
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
    """
    A general affine transformation.

    This is a composition of a linear transformation and a translation.
    """

    transformation_linear: LinearTransformationT = dataclasses.MISSING
    """The linear component of this affine transformation."""

    translation: TranslationT = dataclasses.MISSING
    """The translation component of this affine transformation."""


@dataclasses.dataclass
class AbstractTransformationList(
    AbstractTransformation,
):
    """An interface describing a sequence of transformations."""

    @property
    def type_concrete(self) -> Type[TransformationList]:
        return TransformationList

    @property
    @abc.abstractmethod
    def transformations(self) -> list[AbstractTransformation]:
        """
        The underlying list of transformations to compose together.
        """

    @property
    @abc.abstractmethod
    def intrinsic(self) -> bool:
        """
        A flag controlling whether the transformation should be applied to the
        coordinates or the coordinate system.
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
        """
        The composed version of the transformation.

        This is a single transformation representing the entire sequence of
        transformations.
        """
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
    """An arbitrary sequence of transformations."""

    transformations: list[AbstractTransformation] = dataclasses.MISSING
    """The underlying list of transformations to compose together."""

    intrinsic: bool = True
    """
    If :obj:`True`, the transformation will be applied to the coordinates.
    If :obj:`False`, the transformation will be applied to the coordinate system.
    """
