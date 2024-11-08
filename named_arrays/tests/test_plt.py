import pytest
import numpy as np
import matplotlib.axes
import matplotlib.animation
import astropy.units as u
import named_arrays as na

_num_t = 11
_num_w = 12
_num_x = 13
_num_y = 14


@pytest.mark.parametrize(
    argnames="W",
    argvalues=[
        na.linspace(-1, 1, axis="w", num=_num_w) * u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="X",
    argvalues=[
        na.linspace(-2, 2, axis="x", num=_num_x),
    ],
)
@pytest.mark.parametrize(
    argnames="Y",
    argvalues=[
        na.linspace(-1, 1, axis="y", num=_num_y),
    ],
)
@pytest.mark.parametrize(
    argnames="C",
    argvalues=[
        na.random.uniform(-1, 1, shape_random=dict(w=_num_w, x=_num_x, y=_num_y)),
    ],
)
def testspectralmesh(
    W: na.AbstractScalar,
    X: na.AbstractScalar,
    Y: na.AbstractScalar,
    C: na.AbstractScalar,
):
    result_1 = na.plt.pspectralmesh(
        W,
        X,
        Y,
        C=C,
        axis_wavelength="w",
    )
    result_2 = na.plt.pspectralmesh(
        W,
        na.Cartesian2dVectorArray(X, Y),
        C=C,
        axis_wavelength="w",
    )
    result_3 = na.plt.pspectralmesh(
        na.SpectralPositionalVectorArray(
            wavelength=W,
            position=na.Cartesian2dVectorArray(X, Y),
        ),
        C=C,
        axis_wavelength="w",
    )
    result_4 = na.plt.pspectralmesh(
        C=na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=W,
                position=na.Cartesian2dVectorArray(X, Y),
            ),
            outputs=C,
        ),
        axis_wavelength="w",
    )

    assert np.all(result_1 == result_2)
    assert np.all(result_1 == result_3)
    assert np.all(result_1 == result_4)


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


@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_twinx(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.twinx(ax)
    for r in np.nditer(result.ndarray, flags=("refs_ok",)):
        assert isinstance(r.item(), matplotlib.axes.Axes)


@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_twiny(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.twiny(ax)
    for r in np.nditer(result.ndarray, flags=("refs_ok",)):
        assert isinstance(r.item(), matplotlib.axes.Axes)


@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_invert_xaxis(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.invert_xaxis(ax)


@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_invert_yaxis(
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.invert_yaxis(ax)
