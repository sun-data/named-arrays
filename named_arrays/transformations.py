from __future__ import annotations
from typing import TypeVar, Generic, Iterator
from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractTransformation",
]

VectorT = TypeVar("VectorT", bound="na.AbstractVectorArray")
MatrixT = TypeVar("MatrixT", bound="na.AbstractMatrixArray")
LinearTransformationT = TypeVar("LinearTransformationT", bound="AbstractLinearTransformation")
TranslationT = TypeVar("TranslationT", bound="AbstractTranslation")


@dataclasses.dataclass(eq=False)
class AbstractTransformation(
    abc.ABC
):

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
        """


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
    vector: VectorT = dataclasses.MISSING


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
    matrix: MatrixT = dataclasses.MISSING


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

    def __iter__(self) -> Iterator[AbstractTransformation]:
        if self.intrinsic:
            return iter(self.transformations)
        else:
            return reversed(self.transformations)

    @property
    def composed(self) -> AbstractTransformation:
        transformations = iter(self)
        result = next(transformations)
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
