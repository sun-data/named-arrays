import pytest
import astropy.units as u
import named_arrays as na
from ._transformations_test import (
    AbstractTestAbstractTransformation,
    AbstractTestAbstractTranslation,
    AbstractTestAbstractLinearTransformation,
)


translations_cartesian_3d = [
    na.transformations.Cartesian3dTranslation(x=1 * u.mm, y=2 * u.mm),
    na.transformations.Cartesian3dTranslation(
        x=na.linspace(-1, 1, axis="t", num=4) * u.mm,
    )
]


angles = [
    53 * u.deg,
    na.linspace(0, 180, axis="angle", num=5) * u.deg
]

rotations_x = [na.transformations.Cartesian3dRotationX(angle) for angle in angles]
rotations_y = [na.transformations.Cartesian3dRotationY(angle) for angle in angles]
rotations_z = [na.transformations.Cartesian3dRotationZ(angle) for angle in angles]


class AbstractTestAbstractCartesian3dTransformation(
    AbstractTestAbstractTransformation,
):
    pass


@pytest.mark.parametrize("a", translations_cartesian_3d)
class TestCartesian3dTranslation(
    AbstractTestAbstractTranslation
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
