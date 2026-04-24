from typing import Sequence
import pytest
import numpy as np
import scipy.stats
import named_arrays as na


@pytest.mark.parametrize(
    argnames="x, y, axis, where",
    argvalues=[
        (
            na.random.normal(0, 1, dict(x=101)),
            na.random.normal(0, 1, dict(x=101)),
            "x",
            True,
        ),
        (
            na.random.normal(0, 1, dict(x=101, y=11)),
            na.random.normal(0, 1, dict(x=101, y=11)),
            "x",
            True,
        ),
        (
            na.random.normal(0, 1, dict(x=101)),
            na.random.normal(0, 1, dict(x=101)),
            "x",
            na.random.normal(0, 1, dict(x=101)) > 0.2,
        ),
        (
            na.random.normal(0, 1, dict(x=101, y=11)),
            na.random.normal(0, 1, dict(x=101, y=11)),
            "x",
            na.random.normal(0, 1, dict(x=101, y=11)) > 0.2,
        ),
        (
            na.random.normal(0, 1, dict(x=11, y=101)),
            na.random.normal(0, 1, dict(x=11, y=101)),
            "y",
            na.random.normal(0, 1, dict(x=11, y=101)) > 0.2,
        ),
    ],
)
def test_pearsonr(
    x: na.AbstractScalarArray,
    y: na.AbstractScalarArray,
    axis: None | str | Sequence[str],
    where: bool | na.AbstractScalarArray,
):
    result = na.stats.pearsonr(x, y, axis=axis, where=where)

    shape = na.shape_broadcasted(x, y, where)
    _x = x.broadcast_to(shape)
    _y = y.broadcast_to(shape)
    _where = na.broadcast_to(where, shape)

    for i in na.ndindex(shape, axis_ignored=axis):

        result_expected = scipy.stats.pearsonr(_x[i][_where[i]].ndarray, _y[i][_where[i]].ndarray)

        assert np.allclose(result[i].ndarray, result_expected.statistic)


@pytest.mark.parametrize(
    argnames="x, y, axis",
    argvalues=[
        (
            na.random.normal(0, 1, dict(x=101)),
            na.random.normal(0, 1, dict(x=101)),
            "x",
        ),
        (
            na.random.normal(0, 1, dict(x=101, y=11)),
            na.random.normal(0, 1, dict(x=101, y=11)),
            "x",
        ),
        (
            na.random.normal(0, 1, dict(x=11, y=101)),
            na.random.normal(0, 1, dict(x=11, y=101)),
            "y",
        ),
    ],
)
def test_spearmanr(
    x: na.AbstractScalarArray,
    y: na.AbstractScalarArray,
    axis: None | str | Sequence[str],
):
    result = na.stats.spearmanr(x, y, axis=axis)

    shape = na.shape_broadcasted(x, y)
    _x = x.broadcast_to(shape)
    _y = y.broadcast_to(shape)

    for i in na.ndindex(shape, axis_ignored=axis):

        result_expected = scipy.stats.spearmanr(_x[i].ndarray, _y[i].ndarray)

        assert np.allclose(result[i].ndarray, result_expected.statistic)
