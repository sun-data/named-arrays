import pytest
import numpy as np
import matplotlib.axes
import matplotlib.animation
import named_arrays as na

_num_t = 11
_num_x = 12
_num_y = 13


@pytest.mark.parametrize(
    argnames="TXY",
    argvalues=[
        (
            na.linspace(-1, 1, axis="t", num=_num_t),
            na.linspace(-2, 2, axis="x", num=_num_x),
            na.linspace(-1, 1, axis="y", num=_num_y),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="C",
    argvalues=[
        na.random.uniform(-1, 1, shape_random=dict(t=_num_t, x=_num_x, y=_num_y)),
    ],
)
@pytest.mark.parametrize(
    argnames="axis_time",
    argvalues=["t"],
)
def test_pcolormovie(
    TXY: tuple[
        na.AbstractScalarArray,
        na.AbstractScalarArray,
        na.AbstractScalarArray,
    ],
    C: na.AbstractScalarArray,
    axis_time: str,
):
    result = na.plt.pcolormovie(
        *TXY,
        C=C,
        axis_time=axis_time,
    )
    assert isinstance(result, matplotlib.animation.FuncAnimation)
    assert isinstance(result.to_jshtml(), str)


@pytest.mark.parametrize(
    argnames="xlabel,ax",
    argvalues=[
        ("foo", None),
        ("foo", na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_xlabel(
    xlabel: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_xlabel(xlabel, ax=ax)
    result = na.plt.get_xlabel(ax)
    assert np.all(result == xlabel)


@pytest.mark.parametrize(
    argnames="ylabel,ax",
    argvalues=[
        ("foo", None),
        ("foo", na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_ylabel(
    ylabel: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_ylabel(ylabel, ax=ax)
    result = na.plt.get_ylabel(ax)
    assert np.all(result == ylabel)


@pytest.mark.parametrize(
    argnames="label,ax",
    argvalues=[
        ("foo", None),
        ("foo", na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_title(
    label: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_title(label, ax=ax)
    result = na.plt.get_title(ax)
    assert np.all(result == label)


@pytest.mark.parametrize(
    argnames="value,ax",
    argvalues=[
        ("log", None),
        ("log", na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_xscale(
    value: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_xscale(value, ax=ax)
    result = na.plt.get_xscale(ax)
    assert np.all(result == value)


@pytest.mark.parametrize(
    argnames="value,ax",
    argvalues=[
        ("log", None),
        ("log", na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_yscale(
    value: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_yscale(value, ax=ax)
    result = na.plt.get_yscale(ax)
    assert np.all(result == value)


@pytest.mark.parametrize(
    argnames="aspect,ax",
    argvalues=[
        (1, None),
        (1, na.plt.subplots(ncols=3)[1]),
        (2, na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_aspect(
    aspect: str | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_aspect(aspect, ax=ax)
    result = na.plt.get_aspect(ax)
    assert np.all(result == aspect)


@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_transAxes(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.transAxes(ax)
    assert isinstance(result, na.AbstractArray)
    assert result.shape == na.shape(ax)


@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_transData(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.transData(ax)
    assert isinstance(result, na.AbstractArray)
    assert result.shape == na.shape(ax)
