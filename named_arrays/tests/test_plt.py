import pytest
import numpy as np
import matplotlib.axes
import matplotlib.animation
import matplotlib.text
import matplotlib.backend_bases
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na

_num_t = 11
_num_w = 12
_num_x = 13
_num_y = 14


@pytest.mark.parametrize(
    argnames="y",
    argvalues=[
        2,
    ],
)
@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_axhline(
    y: float | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.axhline(
        y=y,
        ax=ax,
    )

    for r in result.ndarray.flat:
        assert isinstance(r, plt.Line2D)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        2,
    ],
)
@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_axvline(
    x: float | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.axvline(
        x=x,
        ax=ax,
    )

    for r in result.ndarray.flat:
        assert isinstance(r, plt.Line2D)


@pytest.mark.parametrize(
    argnames="ymin",
    argvalues=[
        2,
    ],
)
@pytest.mark.parametrize(
    argnames="ymax",
    argvalues=[
        3,
    ],
)
@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_axhspan(
    ymin: float | na.AbstractScalar,
    ymax: float | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.axhspan(
        ymin=ymin,
        ymax=ymax,
        ax=ax,
    )

    for r in result.ndarray.flat:
        assert isinstance(r, plt.Rectangle)


@pytest.mark.parametrize(
    argnames="xmin",
    argvalues=[
        2,
    ],
)
@pytest.mark.parametrize(
    argnames="xmax",
    argvalues=[
        3,
    ],
)
@pytest.mark.parametrize(
    argnames="ax",
    argvalues=[
        None,
        na.plt.subplots(ncols=3)[1]
    ]
)
def test_axvspan(
    xmin: float | na.AbstractScalar,
    xmax: float | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    result = na.plt.axvspan(
        xmin=xmin,
        xmax=xmax,
        ax=ax,
    )

    for r in result.ndarray.flat:
        assert isinstance(r, plt.Rectangle)


@pytest.mark.parametrize(
    argnames="text",
    argvalues=[
        "foo",
    ],
)
@pytest.mark.parametrize(
    argnames="xy,xytext,components",
    argvalues=[
        (
            na.Cartesian2dVectorArray(1, 1),
            None,
            None,
        ),
        (
            na.Cartesian2dVectorArray(1, 1),
            na.Cartesian2dVectorArray(2, 1),
            None,
        ),
        (
            na.Cartesian2dVectorArray(na.Cartesian2dVectorArray(1, 2), 1),
            None,
            ("x.y", "y"),
        ),
    ],
)
def test_annotate(
    text: str | na.AbstractScalarArray,
    xy: na.AbstractVectorArray,
    xytext: None | na.AbstractVectorArray,
    components: None | tuple[str, str],
):

    fig, ax = plt.subplots()

    result = na.plt.annotate(
        text=text,
        xy=xy,
        xytext=xytext,
        components=components,
    )

    for element in result.ndarray.flat:
        assert isinstance(element, matplotlib.text.Annotation)

    plt.close(fig)


@pytest.mark.parametrize(
    argnames="a,b,components",
    argvalues=[
        (
            na.Cartesian2dVectorArray(0.2, 0.5) * u.mm,
            na.Cartesian2dVectorArray(0.8, 0.5) * u.mm,
            None,
        ),
        (
            na.Cartesian2dVectorArray(
                x=na.linspace(0.1, 0.5, axis="d", num=3),
                y=0.2,
            ) * u.mm,
            na.Cartesian2dVectorArray(
                x=0.9,
                y=na.linspace(0.4, 0.9, axis="d", num=3),
            ) * u.mm,
            None,
        ),
        (
            na.Cartesian2dVectorArray(na.Cartesian2dVectorArray(0.2, 0.3), 0.5) * u.mm,
            na.Cartesian2dVectorArray(na.Cartesian2dVectorArray(0.8, 0.7), 0.5) * u.mm,
            ("x.y", "y"),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="offset",
    argvalues=[
        0,
        0.0 * u.mm,
        0.15 * u.mm,
        na.ScalarArray(np.array([0.1, -0.1, 0.1]), axes="d") * u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="label",
    argvalues=[
        None,
        "foo",
    ],
)
@pytest.mark.parametrize(
    argnames="rotate",
    argvalues=[
        True,
        False,
    ],
)
def test_dimension(
    a: na.AbstractCartesian2dVectorArray,
    b: na.AbstractCartesian2dVectorArray,
    components: None | tuple[str, str],
    offset: float | na.AbstractScalar,
    label: None | str,
    rotate: bool,
):
    fig, ax = plt.subplots()

    result = na.plt.dimension(
        a=a,
        b=b,
        offset=offset,
        label=label,
        rotate=rotate,
        components=components,
        ax=ax,
    )

    for element in result.ndarray.flat:
        assert isinstance(element, matplotlib.text.Annotation)

    plt.close(fig)


def test_dimension_default_ax():
    fig, ax = plt.subplots()

    result = na.plt.dimension(
        a=na.Cartesian2dVectorArray(0.2, 0.5),
        b=na.Cartesian2dVectorArray(0.8, 0.5),
    )

    for element in result.ndarray.flat:
        assert isinstance(element, matplotlib.text.Annotation)

    plt.close(fig)


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
def test_rgbmesh(
    W: na.AbstractScalar,
    X: na.AbstractScalar,
    Y: na.AbstractScalar,
    C: na.AbstractScalar,
):
    result_1 = na.plt.rgbmesh(
        W,
        X,
        Y,
        C=C,
        axis_wavelength="w",
    )
    result_2 = na.plt.rgbmesh(
        W,
        na.Cartesian2dVectorArray(X, Y),
        C=C,
        axis_wavelength="w",
    )
    result_3 = na.plt.rgbmesh(
        na.SpectralPositionalVectorArray(
            wavelength=W,
            position=na.Cartesian2dVectorArray(X, Y),
        ),
        C=C,
        axis_wavelength="w",
    )
    result_4 = na.plt.rgbmesh(
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
    argnames="t,x,y,C,axis_time,ax",
    argvalues=[
        (
            na.random.uniform(-1, 1, shape_random=dict(t=_num_t)),
            na.linspace(-2, 2, axis="x", num=_num_x),
            na.linspace(-1, 1, axis="y", num=_num_y),
            na.random.uniform(-1, 1, shape_random=dict(t=_num_t, x=_num_x, y=_num_y)),
            "t",
            None,
        ),
        (
            na.random.uniform(-1, 1, shape_random=dict(t=_num_t, c=2)),
            na.linspace(-2, 2, axis="x", num=_num_x),
            na.linspace(-1, 1, axis="y", num=_num_y),
            na.random.uniform(-1, 1, shape_random=dict(t=_num_t, x=_num_x, y=_num_y)),
            "t",
            na.plt.subplots(axis_rows="c", nrows=2)[1],
        ),
    ],
)
def test_pcolormovie(
    t: na.AbstractArray,
    x: na.AbstractArray,
    y: na.AbstractArray,
    C: na.AbstractScalarArray,
    axis_time: str,
    ax: None | matplotlib.axes.Axes | na.AbstractArray,
):
    result = na.plt.pcolormovie(
        t, x, y,
        C=C,
        axis_time=axis_time,
        ax=ax,
    )
    assert isinstance(result, matplotlib.animation.FuncAnimation)
    assert isinstance(result.to_jshtml(), str)


@pytest.mark.parametrize(
    argnames="T",
    argvalues=[
        na.linspace(-1, 1, axis="t", num=_num_t) * u.s,
    ],
)
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
        na.random.uniform(
            low=-1,
            high=1,
            shape_random=dict(t=_num_t, w=_num_w, x=_num_x, y=_num_y),
        ),
    ],
)
def test_rgbmovie(
    T: na.AbstractScalar,
    W: na.AbstractScalar,
    X: na.AbstractScalar,
    Y: na.AbstractScalar,
    C: na.AbstractScalar,
):
    ani_1, cbar_1 = na.plt.rgbmovie(
        T,
        W,
        X,
        Y,
        C=C,
        axis_time="t",
        axis_wavelength="w",
    )
    ani_2, cbar_2 = na.plt.rgbmovie(
        T,
        na.SpectralPositionalVectorArray(
            wavelength=W,
            position=na.Cartesian2dVectorArray(X, Y),
        ),
        C=C,
        axis_time="t",
        axis_wavelength="w",
    )
    ani_3, cbar_3 = na.plt.rgbmovie(
        T,
        W,
        na.Cartesian2dVectorArray(X, Y),
        C=C,
        axis_time="t",
        axis_wavelength="w",
    )
    ani_4, cbar_4 = na.plt.rgbmovie(
        na.TemporalSpectralPositionalVectorArray(
            time=T,
            wavelength=W,
            position=na.Cartesian2dVectorArray(X, Y),
        ),
        C=C,
        axis_time="t",
        axis_wavelength="w",
    )
    ani_5, cbar_5 = na.plt.rgbmovie(
        C=na.FunctionArray(
            inputs=na.TemporalSpectralPositionalVectorArray(
                time=T,
                wavelength=W,
                position=na.Cartesian2dVectorArray(X, Y),
            ),
            outputs=C,
        ),
        axis_time="t",
        axis_wavelength="w",
    )

    assert isinstance(ani_1, matplotlib.animation.FuncAnimation)
    assert isinstance(ani_2, matplotlib.animation.FuncAnimation)
    assert isinstance(ani_3, matplotlib.animation.FuncAnimation)
    assert isinstance(ani_4, matplotlib.animation.FuncAnimation)
    assert isinstance(ani_5, matplotlib.animation.FuncAnimation)

    assert isinstance(ani_1.to_jshtml(), str)
    assert isinstance(ani_2.to_jshtml(), str)
    assert isinstance(ani_3.to_jshtml(), str)
    assert isinstance(ani_4.to_jshtml(), str)
    assert isinstance(ani_5.to_jshtml(), str)

    assert np.all(cbar_1 == cbar_2)
    assert np.all(cbar_1 == cbar_3)
    assert np.all(cbar_1 == cbar_4)
    assert np.all(cbar_1 == cbar_5)


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
    argnames="left,right,ax",
    argvalues=[
        (4, 5, na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_xlim(
    left: float | na.AbstractScalar,
    right: float | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_xlim(left, right, ax=ax, emit=False)
    left_saved, right_saved = na.plt.get_xlim(ax)
    if left is not None:
        assert np.all(left_saved == left)
    if right is not None:
        assert np.all(right_saved == right)


@pytest.mark.parametrize(
    argnames="bottom,top,ax",
    argvalues=[
        (1, 2, na.plt.subplots(ncols=3)[1]),
    ]
)
def test_set_ylim(
    bottom: float | na.AbstractScalar,
    top: float | na.AbstractScalar,
    ax: None | matplotlib.axes.Axes | na.AbstractScalar,
):
    na.plt.set_ylim(bottom, top, ax=ax)
    b, t = na.plt.get_ylim(ax)
    if bottom is not None:
        assert np.all(b == bottom)
    if top is not None:
        assert np.all(t == top)


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


_num_slicer_t = 3
_num_slicer_w = 5
_num_slicer_x = 8
_num_slicer_y = 7

_kwargs_slicer = dict(
    axis_time="t",
    axis_wavelength="w",
    axis_x="x",
    axis_y="y",
)


def _hypercube_function(units: bool) -> na.FunctionArray:
    """A moving Gaussian blob with a spatially-varying Doppler shift."""
    time = na.linspace(0, 2, axis="t", num=_num_slicer_t)
    wavelength = na.linspace(-2, 2, axis="w", num=_num_slicer_w)
    x = na.linspace(-1, 1, axis="x", num=_num_slicer_x)
    y = na.linspace(-1, 1, axis="y", num=_num_slicer_y)

    center = 0.5 * time - 0.5
    blob = np.exp(-(np.square(x - center) + np.square(y)) / np.square(0.5))
    line = np.exp(-np.square(wavelength - x) / 2)
    outputs = 1 + blob * line

    if units:
        time = time * u.s
        wavelength = 500 * u.nm + wavelength * u.nm
        outputs = outputs * u.photon

    return na.FunctionArray(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=time,
            wavelength=wavelength,
            position=na.Cartesian2dVectorArray(x, y),
        ),
        outputs=outputs,
    )


def _hypercube_wavelength(function: na.FunctionArray) -> na.AbstractScalar:
    return na.as_named_array(function.inputs.wavelength).broadcast_to(
        shape={"w": _num_slicer_w},
    )


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _hypercube_function(units=False),
        _hypercube_function(units=True),
    ],
)
def test_hypercube_slicer(function: na.FunctionArray):
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)

    assert isinstance(slicer.fig, plt.Figure)
    assert slicer.mode == "intensity"
    assert slicer.index_time == 0
    assert slicer.index_x == _num_slicer_x // 2
    assert slicer.index_y == _num_slicer_y // 2

    result = slicer.ax_image.images[0].get_array()
    expected = function.outputs[dict(t=0)].sum("w")
    expected = expected.value.ndarray_aligned(("y", "x"))
    assert result.shape == (_num_slicer_y, _num_slicer_x)
    np.testing.assert_allclose(result, expected)

    plt.close(slicer.fig)


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _hypercube_function(units=False),
        _hypercube_function(units=True),
    ],
)
def test_hypercube_slicer_set_time(function: na.FunctionArray):
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)

    slicer.set_time(1)
    assert slicer.index_time == 1

    result = slicer.ax_image.images[0].get_array()
    expected = function.outputs[dict(t=1)].sum("w")
    expected = expected.value.ndarray_aligned(("y", "x"))
    np.testing.assert_allclose(result, expected)

    result_slit = slicer.ax_slit_vertical.images[0].get_array()
    expected_slit = function.outputs[dict(t=1, x=slicer.index_x)]
    expected_slit = expected_slit.value.ndarray_aligned(("y", "w"))
    np.testing.assert_allclose(result_slit, expected_slit)

    slicer.set_time(_num_slicer_t)
    assert slicer.index_time == 0

    slicer.set_time(-1)
    assert slicer.index_time == _num_slicer_t - 1

    plt.close(slicer.fig)


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _hypercube_function(units=False),
        _hypercube_function(units=True),
    ],
)
def test_hypercube_slicer_set_point(function: na.FunctionArray):
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)

    index_x = 2
    index_y = 3
    slicer.set_point(index_x, index_y)
    assert slicer.index_x == index_x
    assert slicer.index_y == index_y

    outputs = function.outputs

    result_vertical = slicer.ax_slit_vertical.images[0].get_array()
    expected_vertical = outputs[dict(t=0, x=index_x)]
    expected_vertical = expected_vertical.value.ndarray_aligned(("y", "w"))
    assert result_vertical.shape == (_num_slicer_y, _num_slicer_w)
    np.testing.assert_allclose(result_vertical, expected_vertical)

    result_horizontal = slicer.ax_slit_horizontal.images[0].get_array()
    expected_horizontal = outputs[dict(t=0, y=index_y)]
    expected_horizontal = expected_horizontal.value.ndarray_aligned(("w", "x"))
    assert result_horizontal.shape == (_num_slicer_w, _num_slicer_x)
    np.testing.assert_allclose(result_horizontal, expected_horizontal)

    spectrum = outputs[dict(t=0, x=index_x, y=index_y)]
    wavelength = _hypercube_wavelength(function)
    line = slicer.ax_spectrum.lines[0]
    np.testing.assert_allclose(
        line.get_xdata(),
        wavelength.value.ndarray_aligned(("w",)),
    )
    np.testing.assert_allclose(
        line.get_ydata(),
        spectrum.value.ndarray_aligned(("w",)),
    )

    expected_mean = (wavelength * spectrum).sum("w") / spectrum.sum("w")
    result_mean = slicer._line_spectrum.get_xdata()[0]
    np.testing.assert_allclose(result_mean, expected_mean.value.ndarray)

    crosshair_x = slicer._crosshair_vertical.get_xdata()[0]
    crosshair_y = slicer._crosshair_horizontal.get_ydata()[0]
    assert crosshair_x == index_x
    assert crosshair_y == index_y

    slicer.set_point(-100, 100)
    assert slicer.index_x == 0
    assert slicer.index_y == _num_slicer_y - 1

    plt.close(slicer.fig)


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _hypercube_function(units=False),
        _hypercube_function(units=True),
    ],
)
def test_hypercube_slicer_set_mode(function: na.FunctionArray):
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)

    slicer.set_mode("doppler")
    assert slicer.mode == "doppler"

    outputs = function.outputs[dict(t=0)]
    wavelength = _hypercube_wavelength(function)
    expected = (wavelength * outputs).sum("w") / outputs.sum("w")
    expected = expected.value.ndarray_aligned(("y", "x"))

    result = slicer.ax_image.images[0].get_array()
    np.testing.assert_allclose(result, expected)

    slicer.set_mode("intensity")
    assert slicer.mode == "intensity"

    result = slicer.ax_image.images[0].get_array()
    expected = function.outputs[dict(t=0)].sum("w")
    expected = expected.value.ndarray_aligned(("y", "x"))
    np.testing.assert_allclose(result, expected)

    with pytest.raises(ValueError):
        slicer.set_mode("invalid")

    plt.close(slicer.fig)


def test_hypercube_slicer_wavelength_index():
    function = _hypercube_function(units=False)
    function = na.FunctionArray(
        inputs=function.inputs.position,
        outputs=function.outputs,
    )
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)

    slicer.set_mode("doppler")

    outputs = function.outputs[dict(t=0)]
    wavelength = na.ScalarArray(np.arange(_num_slicer_w), axes=("w",))
    expected = (wavelength * outputs).sum("w") / outputs.sum("w")
    expected = expected.value.ndarray_aligned(("y", "x"))

    result = slicer.ax_image.images[0].get_array()
    np.testing.assert_allclose(result, expected)

    plt.close(slicer.fig)


def test_hypercube_slicer_invalid_axis():
    function = _hypercube_function(units=False)
    with pytest.raises(ValueError):
        na.plt.HypercubeSlicer(function, axis_time="invalid")


def test_hypercube_slicer_invalid_mode():
    function = _hypercube_function(units=False)
    with pytest.raises(ValueError):
        na.plt.HypercubeSlicer(function, **_kwargs_slicer, mode="invalid")


def test_hypercube_slicer_events():
    function = _hypercube_function(units=True)
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)
    fig = slicer.fig
    fig.canvas.draw()

    event_scroll = matplotlib.backend_bases.MouseEvent(
        name="scroll_event",
        canvas=fig.canvas,
        x=0,
        y=0,
        button="up",
    )
    fig.canvas.callbacks.process("scroll_event", event_scroll)
    assert slicer.index_time == 1

    event_scroll_down = matplotlib.backend_bases.MouseEvent(
        name="scroll_event",
        canvas=fig.canvas,
        x=0,
        y=0,
        button="down",
    )
    fig.canvas.callbacks.process("scroll_event", event_scroll_down)
    assert slicer.index_time == 0

    event_key_right = matplotlib.backend_bases.KeyEvent(
        name="key_press_event",
        canvas=fig.canvas,
        key="right",
    )
    fig.canvas.callbacks.process("key_press_event", event_key_right)
    assert slicer.index_time == 1

    event_key_left = matplotlib.backend_bases.KeyEvent(
        name="key_press_event",
        canvas=fig.canvas,
        key="left",
    )
    fig.canvas.callbacks.process("key_press_event", event_key_left)
    assert slicer.index_time == 0

    event_key_m = matplotlib.backend_bases.KeyEvent(
        name="key_press_event",
        canvas=fig.canvas,
        key="m",
    )
    fig.canvas.callbacks.process("key_press_event", event_key_m)
    assert slicer.mode == "doppler"

    fig.canvas.callbacks.process("key_press_event", event_key_m)
    assert slicer.mode == "intensity"

    index_x = 4
    index_y = 2
    x_display, y_display = slicer.ax_image.transData.transform(
        (index_x, index_y),
    )
    event_click = matplotlib.backend_bases.MouseEvent(
        name="button_press_event",
        canvas=fig.canvas,
        x=x_display,
        y=y_display,
        button=matplotlib.backend_bases.MouseButton.LEFT,
    )
    fig.canvas.callbacks.process("button_press_event", event_click)
    assert slicer.index_x == index_x
    assert slicer.index_y == index_y

    plt.close(fig)


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _hypercube_function(units=False),
        _hypercube_function(units=True),
    ],
)
def test_hypercube_slicer_norm_fixed(function: na.FunctionArray):
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)

    outputs = function.outputs

    clim_image = slicer.ax_image.images[0].get_clim()
    clim_vertical = slicer.ax_slit_vertical.images[0].get_clim()
    clim_horizontal = slicer.ax_slit_horizontal.images[0].get_clim()
    ylim_spectrum = slicer.ax_spectrum.get_ylim()

    expected_image = np.nanpercentile(
        outputs.sum("w").value.ndarray,
        (0.1, 99.9),
    )
    np.testing.assert_allclose(clim_image, expected_image)

    expected_slit = np.nanpercentile(outputs.value.ndarray, (0.1, 99.9))
    np.testing.assert_allclose(clim_vertical, expected_slit)
    assert clim_vertical == clim_horizontal

    # stepping through time must not change any color scale or the
    # vertical limits of the spectrum panel
    slicer.set_time(1)
    assert slicer.ax_image.images[0].get_clim() == clim_image
    assert slicer.ax_slit_vertical.images[0].get_clim() == clim_vertical
    assert slicer.ax_slit_horizontal.images[0].get_clim() == clim_horizontal
    assert slicer.ax_spectrum.get_ylim() == ylim_spectrum

    # selecting a new point must not change any color scale, but must
    # recompute the vertical limits of the spectrum panel from the extrema
    # over time of the new point's spectra
    index_x = 0
    index_y = 0
    slicer.set_point(index_x, index_y)
    assert slicer.ax_image.images[0].get_clim() == clim_image
    assert slicer.ax_slit_vertical.images[0].get_clim() == clim_vertical

    spectra = outputs[dict(x=index_x, y=index_y)].value.ndarray
    lo = spectra.min()
    hi = spectra.max()
    pad = 0.1 * (hi - lo)
    np.testing.assert_allclose(
        slicer.ax_spectrum.get_ylim(),
        (lo - pad, hi + pad),
    )

    plt.close(slicer.fig)


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        _hypercube_function(units=False),
        _hypercube_function(units=True),
    ],
)
def test_hypercube_slicer_colorbar(function: na.FunctionArray):
    slicer = na.plt.HypercubeSlicer(function, **_kwargs_slicer)
    fig = slicer.fig

    # four panels plus two colorbar axes
    assert len(fig.axes) == 6

    label_intensity = slicer.cax_image.get_ylabel()

    num_axes = len(fig.axes)
    num_children = len(fig.get_children())
    for _ in range(5):
        slicer.set_mode("doppler")
        slicer.set_mode("intensity")
    assert len(fig.axes) == num_axes
    assert len(fig.get_children()) == num_children
    assert slicer.cax_image.get_ylabel() == label_intensity

    # the colorbar of the image panel must track the norm, colormap, and
    # label of the current mode
    slicer.set_mode("doppler")
    fig.canvas.draw()
    assert slicer.cax_image.get_ylabel() != label_intensity

    outputs = function.outputs
    wavelength = _hypercube_wavelength(function)
    center = (wavelength * outputs).sum() / outputs.sum()
    moment = (wavelength * outputs).sum("w") / outputs.sum("w")
    deviation = np.abs(moment - center).value.ndarray
    center = float(center.value.ndarray)
    halfrange = float(np.nanpercentile(deviation, 99.9))

    clim = slicer.ax_image.images[0].get_clim()
    np.testing.assert_allclose(clim, (center - halfrange, center + halfrange))
    np.testing.assert_allclose(
        slicer._colorbar_image.mappable.get_clim(),
        clim,
    )

    # the shared slit colorbar reflects the norm of both slit panels
    np.testing.assert_allclose(
        slicer._colorbar_slit.mappable.get_clim(),
        slicer.ax_slit_horizontal.images[0].get_clim(),
    )

    plt.close(fig)
