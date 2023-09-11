import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractTestAbstractTransformation",
]

translations = [
    na.transformations.Translation(na.Cartesian2dVectorArray(1, 2) * u.mm),
    na.transformations.Translation(na.Cartesian2dVectorLinearSpace(
        start=-10 * u.mm,
        stop=10 * u.mm,
        axis="t",
        num=4,
    )),
]

transformations_linear = [
    na.transformations.LinearTransformation(na.Cartesian2dIdentityMatrixArray()),
    na.transformations.LinearTransformation(na.Cartesian2dRotationMatrixArray(43 * u.deg)),
]

transformations_affine = [
    na.transformations.AffineTransformation(transformation_linear, translation)
    for transformation_linear in transformations_linear
    for translation in translations
]

transformations_basic = translations + transformations_linear + transformations_affine

transformations_list = [
    na.transformations.TransformationList([t1, t2], intrinsic=intrinsic)
    for t1 in transformations_basic[::2]
    for t2 in transformations_basic[1::2]
    for intrinsic in (False, True)
]

transformations = transformations_basic + transformations_list


class AbstractTestAbstractTransformation(
    abc.ABC,
):

    @pytest.mark.parametrize(
        argnames="x",
        argvalues=[
            na.Cartesian2dVectorArray(5, 5) * u.mm,
        ]
    )
    class TestVectorOperations:

        def test__call__(
                self,
                a: na.transformations.AbstractTransformation,
                x: na.AbstractVectorArray,
        ):
            result = a(x)
            assert isinstance(result, na.AbstractVectorArray)

        def test_inverse(
                self,
                a: na.transformations.AbstractTransformation,
                x: na.AbstractVectorArray,
        ):
            assert np.allclose(a.inverse(a(x)), x)
            assert np.allclose(a(a.inverse(x)), x)

        @pytest.mark.parametrize("b", transformations)
        def test__matmul__(
                self,
                a: na.transformations.AbstractTransformation,
                b: na.transformations.AbstractTransformation,
                x: na.AbstractVectorArray,
        ):
            assert np.allclose((a @ a)(x), a(a(x)))


class AbstractTestAbstractTranslation(
    AbstractTestAbstractTransformation,
):
    def test_vector(self, a: na.transformations.AbstractTranslation):
        assert isinstance(a.vector, na.AbstractVectorArray)


@pytest.mark.parametrize("a", translations)
class TestTranslation(
    AbstractTestAbstractTranslation,
):
    pass


class AbstractTestAbstractLinearTransformation(
    AbstractTestAbstractTransformation,
):
    def test_matrix(self, a: na.transformations.AbstractLinearTransformation):
        assert isinstance(a.matrix, na.AbstractMatrixArray)


@pytest.mark.parametrize("a", transformations_linear)
class TestLinearTransformation(
    AbstractTestAbstractLinearTransformation
):
    pass


class AbstractTestAbstractAffineTransformation(
    AbstractTestAbstractTransformation,
):
    def test_transformation_linear(self, a: na.transformations.AbstractAffineTransformation):
        assert isinstance(a.transformation_linear, na.transformations.AbstractLinearTransformation)

    def test_translation(self, a: na.transformations.AbstractAffineTransformation):
        assert isinstance(a.translation, na.transformations.Translation)


@pytest.mark.parametrize("a", transformations_affine)
class TestAffineTransformation(
    AbstractTestAbstractAffineTransformation
):
    pass


class AbstractTestAbstractTransformationList(
    AbstractTestAbstractTransformation,
):

    def test_transformations(self, a: na.transformations.AbstractTransformationList):
        for t in a.transformations:
            assert isinstance(t, na.transformations.AbstractTransformation)

    def test_intrinsic(self, a: na.transformations.AbstractTransformationList):
        assert isinstance(a.intrinsic, bool)

    def test__iter__(self, a: na.transformations.AbstractTransformationList):
        for transformation in a:
            assert isinstance(transformation, na.transformations.AbstractTransformation)

    def test_composed(self, a: na.transformations.AbstractTransformationList):
        result = a.composed
        assert isinstance(result, na.transformations.AbstractTransformation)
        assert not isinstance(result, na.transformations.AbstractTransformationList)


@pytest.mark.parametrize("a", transformations_list)
class TestTransformationList(
    AbstractTestAbstractTransformationList
):
    pass



