import pytest
import numpy as np
import astropy.units as u
import named_arrays as na


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        2,
        2 * u.mm,
        na.ScalarArray(2) * u.mm,
        na.NormalUncertainScalarArray(2, width=1, num_distribution=11),
        na.Cartesian2dVectorArray(2, 2),
        na.Cartesian2dVectorArray(
            na.Cartesian2dVectorArray(5, 7),
            na.NormalUncertainScalarArray(2, width=1, num_distribution=11),
        )
    ],
)
@pytest.mark.parametrize(
    argnames="b",
    argvalues=[
        3,
        na.linspace(3, 4, axis="b", num=4),
        na.ScalarLinearSpace(3, 4, axis="b", num=4),
        na.Cartesian2dVectorLinearSpace(
            start=3,
            stop=4,
            axis=na.Cartesian2dVectorArray("a", "b"),
            num=4,
        )
    ],
)
def test_evaluate(
    a: na.AbstractArray,
    b: na.AbstractArray,
):

    if na.unit(a) is not None or na.unit(b) is not None:
        with pytest.raises(ValueError):
            na.numexpr.evaluate("a * b")
        return

    result = na.numexpr.evaluate("a * b")

    result_expected  = a * b

    assert type(result) == type(na.as_named_array(result_expected))

    assert np.all(result == result_expected)
