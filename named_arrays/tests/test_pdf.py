import numpy as np
from typing import Sequence
import pytest
import scipy.special as sp
import named_arrays as na


@pytest.mark.parametrize(
    argnames="mean",
    argvalues=[
        0,
        na.ScalarArray(2),
        na.linspace(0,1, axis="y", num=11),
        na.NormalUncertainScalarArray(0, width=1,)
    ]
)
@pytest.mark.parametrize(
    argnames="std",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="q",
    argvalues=[
        25,
    ]
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
        None,
        "x",
        ("x",),
        ("x", "y"),
    ]
)
def test_pdf_argpercentile_gaussian(
    mean: float | na.AbstractScalar,
    std: float | na.AbstractScalar,
    q: float | na.AbstractScalar,
    axis: None | str | Sequence[str],
):
    x = na.linspace(
        start=mean - 5 * std,
        stop=mean + 5 * std,
        axis="x",
        num=100001,
    )
    xc = x.cell_centers("x")

    a = np.exp(-np.square((xc - mean) / std) / 2) / np.sqrt(2 * np.pi) / std

    _axis = na.axis_normalized(a, axis=axis)

    if len(_axis) != 1:
        with pytest.raises(ValueError):
            na.pdf.argpercentile(a, q=q, axis=_axis)
        return

    _axis = _axis[0]

    result = na.pdf.argpercentile(a, q=q, axis=axis)

    array_reduced = na.interp(
        x=1 + result[_axis],
        xp=na.indices(x.shape)[_axis],
        fp=x,
    )

    p = q / 100

    array_reduced_expected = mean + std * np.sqrt(2) * sp.erfinv(2 * p - 1)

    assert np.allclose(array_reduced, array_reduced_expected, rtol=1e-2)


@pytest.mark.parametrize(
    argnames="a,q,axis,result_expected",
    argvalues=[
        (
            na.ScalarArray.ones(dict(x=3)),
            50,
            "x",
            dict(x=1.5),
        ),
        (
            na.ScalarArray.ones(dict(x=4)),
            50,
            "x",
            dict(x=2),
        ),
    ],
)
def test_pdf_argpercentile(
    a: na.AbstractScalar,
    q: float | na.AbstractScalar,
    axis: None | str | Sequence[str],
    result_expected: dict[str, na.AbstractScalar],
):
    result = na.pdf.argpercentile(a, q=q, axis=axis)

    assert result.keys() == result_expected.keys()

    for ax in result:
        assert np.all(result[ax] == result_expected[ax])


@pytest.mark.parametrize(
    argnames="x,f,q,axis,result_expected",
    argvalues=[
        (
            na.linspace(-1, 1, axis="x", num=4),
            na.ScalarArray.ones(dict(x=3)),
            25,
            "x",
            -0.5,
        ),
    ],
)
def test_percentile(
    x: na.AbstractScalar,
    f: na.AbstractScalar,
    q: float | na.AbstractScalar,
    axis: None | str | Sequence[str],
    result_expected: dict[str, na.AbstractScalar],
):
    result = na.pdf.percentile(
        x=x,
        f=f,
        q=q,
        axis=axis,
    )

    assert np.all(result == result_expected)


@pytest.mark.parametrize(
    argnames="x,f,axis,result_expected",
    argvalues=[
        (
            na.linspace(-1, 1, axis="x", num=4),
            na.ScalarArray.ones(dict(x=3)),
            "x",
            0,
        ),
    ],
)
def test_median(
    x: na.AbstractScalar,
    f: na.AbstractScalar,
    axis: None | str | Sequence[str],
    result_expected: dict[str, na.AbstractScalar],
):
    result = na.pdf.median(
        x=x,
        f=f,
        axis=axis,
    )

    assert np.allclose(result, result_expected)


@pytest.mark.parametrize(
    argnames="x,f,axis,result_expected",
    argvalues=[
        (
            na.linspace(-1, 1, axis="x", num=4),
            na.ScalarArray.ones(dict(x=3)),
            "x",
            1,
        ),
    ],
)
def test_iqr(
    x: na.AbstractScalar,
    f: na.AbstractScalar,
    axis: None | str | Sequence[str],
    result_expected: dict[str, na.AbstractScalar],
):
    result = na.pdf.iqr(
        x=x,
        f=f,
        axis=axis,
    )

    assert np.allclose(result, result_expected)

