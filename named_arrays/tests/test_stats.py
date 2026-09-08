from typing import Sequence
import pytest
import numpy as np
import scipy.stats
import named_arrays as na


@pytest.mark.parametrize(
    argnames="x, y, axis, where",
    argvalues=[
        (
            na.random.uniform(0, 100, dict(x=101)).astype(int),
            na.random.uniform(0, 100, dict(x=101)).astype(int),
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
            na.random.uniform(0, 10, dict(x=101)).astype(int),
            na.random.uniform(0, 10, dict(x=101)).astype(int),
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


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "invalid"])
@pytest.mark.parametrize(
    argnames="a, axis, where",
    argvalues=[
        (
            na.random.uniform(0, 10, dict(x=101)).astype(int),
            "x",
            True,
        ),
        (
            na.random.uniform(0, 10, dict(x=11, y=101)).astype(int),
            "y",
            True,
        ),
        (
            na.random.uniform(0, 10, dict(x=101, y=11)).astype(int),
            "x",
            na.random.uniform(0, 1, dict(x=101, y=11)) > 0.2,
        ),
        (
            na.random.uniform(0, 10, dict(x=11, y=13)).astype(int),
            ("x", "y"),
            True,
        ),
        (
            na.random.uniform(0, 10, dict(x=11, y=13)).astype(int),
            None,
            na.random.uniform(0, 1, dict(x=11, y=13)) > 0.2,
        ),
    ],
)
def test_rankdata(
    a: na.AbstractScalarArray,
    axis: None | str | Sequence[str],
    where: bool | na.AbstractScalarArray,
    method: str,
):
    if method not in ("average", "min", "max", "dense"):
        with pytest.raises(ValueError):
            na.stats.rankdata(a, axis=axis, method=method, where=where)
        return

    result = na.stats.rankdata(a, axis=axis, method=method, where=where)

    # The result always lies along the flattened version of `axis`, so combine
    # the ranking axes on the inputs and compare each remaining slice to scipy.
    shape = na.shape_broadcasted(a, where)
    _a = a.broadcast_to(shape)
    _where = na.broadcast_to(where, shape)

    axis_normalized = na.axis_normalized(_a, axis)
    axis_flat = na.flatten_axes(axis_normalized)
    _a = _a.combine_axes(axis_normalized, axis_flat)
    _where = _where.combine_axes(axis_normalized, axis_flat)

    for i in na.ndindex(_a.shape, axis_ignored=axis_flat):

        mask = _where[i].ndarray
        result_expected = np.full(mask.shape, np.nan)
        result_expected[mask] = scipy.stats.rankdata(_a[i].ndarray[mask], method=method)

        assert np.allclose(result[i].ndarray, result_expected, equal_nan=True)
