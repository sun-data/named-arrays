import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractTestAbstractTransformation",
]

identities = [
    na.transformations.IdentityTransformation(),
]

translations = [
    na.transformations.Translation(na.Cartesian3dVectorArray(1, 2) * u.mm),
    na.transformations.Translation(na.Cartesian3dVectorLinearSpace(
        start=-10 * u.mm,
        stop=10 * u.mm,
        axis="t",
        num=4,
    )),
]

translations_cartesian_3d = [
    na.transformations.Cartesian3dTranslation(x=1 * u.mm, y=2 * u.mm),
    na.transformations.Cartesian3dTranslation(
        x=na.linspace(-1, 1, axis="t", num=4) * u.mm,
    )
]

transformations_linear = [
    na.transformations.LinearTransformation(na.Cartesian3dIdentityMatrixArray()),
    na.transformations.LinearTransformation(na.Cartesian3dXRotationMatrixArray(43 * u.deg)),
]

angles = [
    53 * u.deg,
    na.linspace(0, 180, axis="angle", num=5) * u.deg
]

rotations_x = [na.transformations.Cartesian3dRotationX(angle) for angle in angles]
rotations_y = [na.transformations.Cartesian3dRotationY(angle) for angle in angles]
rotations_z = [na.transformations.Cartesian3dRotationZ(angle) for angle in angles]


transformations_affine = [
    na.transformations.AffineTransformation(transformation_linear, translation)
    for transformation_linear in transformations_linear
    for translation in translations
]

transformations_basic = identities + translations + transformations_linear + transformations_affine

transformations_list = [
    na.transformations.TransformationList([]),
]

transformations_list += [
    na.transformations.TransformationList([t1, t2], intrinsic=intrinsic)
    for t1 in transformations_basic[::2]
    for t2 in transformations_basic[1::2]
    for intrinsic in (False, True)
]

transformations = transformations_basic + transformations_list


class AbstractTestAbstractTransformation(
    abc.ABC,
):

    def test_shape(self, a: na.transformations.AbstractTransformation):
        result = a.shape
        assert isinstance(result, dict)
        for k in result:
            assert isinstance(k, str)
            assert isinstance(result[k], int)

    @pytest.mark.parametrize(
        argnames="x",
        argvalues=[
            na.Cartesian3dVectorArray(10, 20, 30) * u.mm,
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


@pytest.mark.parametrize("a", identities)
class TestIdentityTransformation(
    AbstractTestAbstractTransformation,
):
    pass


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


@pytest.mark.parametrize("a", translations_cartesian_3d)
class TestCartesian3dTranslation(
    AbstractTestAbstractTranslation
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


class AbstractTestAbstractCartesian3dRotation(
    AbstractTestAbstractLinearTransformation,
):
    pass


@pytest.mark.parametrize("a", rotations_x)
class TestCartesian3dRotationX(
    AbstractTestAbstractCartesian3dRotation,
):
    pass


@pytest.mark.parametrize("a", rotations_y)
class TestCartesian3dRotationY(
    AbstractTestAbstractCartesian3dRotation,
):
    pass


@pytest.mark.parametrize("a", rotations_z)
class TestCartesian3dRotationZ(
    AbstractTestAbstractCartesian3dRotation,
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

    class TestVectorOperations(
        AbstractTestAbstractTransformation.TestVectorOperations
    ):
        def test__call__(
                self,
                a: na.transformations.AbstractTransformationList,
                x: na.AbstractVectorArray,
        ):
            result = a(x)
            result_expected = x
            for transformation in a:
                result_expected = transformation(result_expected)
            assert np.allclose(result, result_expected)


@pytest.mark.parametrize("a", transformations_list)
class TestTransformationList(
    AbstractTestAbstractTransformationList
):
    pass



