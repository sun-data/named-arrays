import pytest
import numpy as np
import astropy.units as u
import named_arrays as na


@pytest.mark.parametrize(
    argnames="n",
    argvalues=[
        10,
        (11 * u.photon).astype(int),
        na.ScalarArray(12),
        (na.arange(1, 10, axis="x") << u.photon).astype(int),
        na.Cartesian2dVectorArray(10, 11),
    ],
)
@pytest.mark.parametrize(
    argnames="p",
    argvalues=[
        0.5,
        na.ScalarArray(0.51),
        na.linspace(0.4, 0.5, axis="p", num=5),
        na.UniformUncertainScalarArray(0.5, width=0.1),
        na.Cartesian2dVectorArray(0.5, 0.6),
    ],
)
@pytest.mark.parametrize(
    argnames="shape_random",
    argvalues=[
        None,
        dict(_s=6),
    ],
)
@pytest.mark.parametrize(
    argnames="seed",
    argvalues=[
        None,
        42,
    ],
)
def test_binomial(
    n: int | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    p: float | na.AbstractScalar | na.AbstractVectorArray,
    shape_random: None | dict[str, int],
    seed: None | int,
):
    result = na.random.binomial(
        n=n,
        p=p,
        shape_random=shape_random,
        seed=seed,
    )

    assert na.unit(result) == na.unit(n)

    assert np.all(result >= 0)
    assert np.all(result <= n)


@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        0.5,
        na.ScalarArray(0.51),
        na.linspace(0.4, 0.5, axis="p", num=5),
        na.UniformUncertainScalarArray(0.5, width=0.1),
        na.Cartesian2dVectorArray(0.5, 0.6),
    ],
)
@pytest.mark.parametrize(
    argnames="scale",
    argvalues=[
        10,
        (11 * u.photon).astype(int),
        na.ScalarArray(12),
        (na.arange(1, 10, axis="x") << u.photon).astype(int),
        na.Cartesian2dVectorArray(10, 11),
    ],
)
@pytest.mark.parametrize(
    argnames="shape_random",
    argvalues=[
        None,
        dict(_s=6),
    ],
)
@pytest.mark.parametrize(
    argnames="seed",
    argvalues=[
        None,
        42,
    ],
)
def test_gamma(
    shape: float | na.AbstractScalar | na.AbstractVectorArray,
    scale: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    shape_random: None | dict[str, int],
    seed: None | int,
):
    result = na.random.gamma(
        shape=shape,
        scale=scale,
        shape_random=shape_random,
        seed=seed,
    )

    assert na.unit(result) == na.unit(scale)
    assert np.all(result >= 0)
