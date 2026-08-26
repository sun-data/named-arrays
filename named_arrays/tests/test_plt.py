import pytest
import numpy as np
import matplotlib.axes
import matplotlib.animation
import matplotlib.text
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
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


def _square(y: float) -> na.Cartesian3dVectorArray:
    """A unit square in the plane of constant `y`."""
    return na.Cartesian3dVectorArray(
        x=na.ScalarArray(np.array([-1.0, 1, 1, -1]), axes=("wire",)) * u.mm,
        y=na.ScalarArray(np.array([y] * 4, dtype=float), axes=("wire",)) * u.mm,
        z=na.ScalarArray(np.array([-1.0, -1, 1, 1]), axes=("wire",)) * u.mm,
    )


def test_fill_3d():
    """
    A polygon filled on a 3D axes is drawn as a 3D collection.

    :meth:`matplotlib.axes.Axes.fill` has no 3D counterpart, and on a 3D axes
    it reads the third coordinate as another polygon, giving flat patches in
    the plane of the page.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    result = na.plt.fill(
        _square(0),
        ax=ax,
        axis="wire",
        components=("x", "y", "z"),
    )

    assert isinstance(
        result[dict()].ndarray,
        mpl_toolkits.mplot3d.art3d.Poly3DCollection,
    )
    assert not ax.patches
    plt.close(fig)


def test_fill_3d_occludes():
    """
    A polygon filled on a 3D axes hides what is behind it.

    This is the point of drawing one as a collection: it takes part in the
    depth sorting of the axes, where a flat patch would not.
    """
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=0, azim=-90)

    for y, color in ((1, "tab:red"), (-1, "tab:blue")):
        na.plt.fill(
            _square(y),
            ax=ax,
            axis="wire",
            components=("x", "y", "z"),
            color=color,
        )

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255
    plt.close(fig)

    red, _, blue = image[image.shape[0] // 2, image.shape[1] // 2]
    assert blue > red


def test_fill_2d():
    """Filling on a 2D axes is unchanged."""
    fig, ax = plt.subplots()

    na.plt.fill(
        _square(0),
        ax=ax,
        axis="wire",
        components=("x", "z"),
    )

    assert len(ax.patches) == 1
    plt.close(fig)


def _diagonal() -> na.Cartesian3dVectorArray:
    """A line crossing the origin, sampled at five points."""
    t = na.linspace(-2, 2, axis="t", num=5)
    return na.Cartesian3dVectorArray(x=t, y=t, z=0 * t) * u.mm


def test_line_collection_3d():
    """A line drawn on a 3D axes becomes one collection per segment."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    result = na.plt.line_collection(
        _diagonal(),
        ax=ax,
        axis="t",
        components=("x", "y", "z"),
    )

    # one artist per gap between samples, rather than one for the whole line
    assert na.shape(result) == dict(t=4)
    assert len(ax.collections) == 4
    assert not ax.lines

    for artist in result.ndarray.flat:
        assert isinstance(artist, mpl_toolkits.mplot3d.art3d.Line3DCollection)

    plt.close(fig)


def test_line_collection_2d():
    """On a 2D axes it is an ordinary line collection, and it sets the limits."""
    fig, ax = plt.subplots()

    result = na.plt.line_collection(
        _diagonal(),
        ax=ax,
        axis="t",
        components=("x", "y"),
    )

    assert na.shape(result) == dict(t=4)
    assert len(ax.collections) == 4
    assert not ax.lines

    # the axes has been told how big the line is
    assert ax.get_xlim()[0] < -2
    assert ax.get_xlim()[1] > 2

    plt.close(fig)


def test_line_collection_3d_is_depth_sorted():
    """
    A line drawn this way is sorted into a 3D scene by its depth.

    This is the point of drawing one as a collection. A 3D axes sorts only its
    collections and patches, so a line drawn by :func:`named_arrays.plt.plot`
    keeps the zorder it was given and is placed either in front of every filled
    surface or behind all of them.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # looking along the y axis, so the line at negative y is the nearer one
    ax.view_init(elev=0, azim=-90)

    def line(y: float) -> na.Cartesian3dVectorArray:
        t = na.linspace(-2, 2, axis="t", num=5)
        return na.Cartesian3dVectorArray(x=t, y=0 * t + y, z=0 * t) * u.mm

    near = na.plt.line_collection(
        line(-4), ax=ax, axis="t", components=("x", "y", "z")
    )
    far = na.plt.line_collection(
        line(4), ax=ax, axis="t", components=("x", "y", "z")
    )
    plotted = na.plt.plot(line(0), ax=ax, axis="t", components=("x", "y", "z"))

    fig.canvas.draw()

    zorder_near = [a.get_zorder() for a in near.ndarray.flat]
    zorder_far = [a.get_zorder() for a in far.ndarray.flat]
    zorder_plotted = [a.get_zorder() for a in np.atleast_1d(plotted.ndarray).flat]

    plt.close(fig)

    # every segment of the nearer line is drawn after every segment of the
    # further one, which the axes worked out for itself
    assert min(zorder_near) > max(zorder_far)

    # while a plotted line is left where it started, underneath both of them
    assert all(z < min(zorder_far) for z in zorder_plotted)


def test_line_collection_kwargs_singular():
    """The spellings a line takes are accepted for a collection."""
    fig, ax = plt.subplots()

    result = na.plt.line_collection(
        _diagonal(),
        ax=ax,
        axis="t",
        components=("x", "y"),
        color="tab:red",
        linewidth=2,
        linestyle="--",
    )

    artist = result[dict(t=0)].ndarray
    assert artist.get_linewidth()[0] == 2
    assert tuple(artist.get_color()[0]) == matplotlib.colors.to_rgba("tab:red")

    plt.close(fig)


def test_line_collection_transformation():
    """The line is moved before it is drawn, as with the other plotting functions."""
    fig, ax = plt.subplots()

    shift = 10 * u.mm
    result = na.plt.line_collection(
        _diagonal(),
        ax=ax,
        axis="t",
        components=("x", "y"),
        transformation=na.transformations.Cartesian3dTranslation(x=shift),
    )

    segment = result[dict(t=0)].ndarray.get_segments()[0]

    # the line started at -2 mm and has been moved along by ten
    assert segment[0][0] == pytest.approx(-2 + shift.to_value(u.mm))

    plt.close(fig)
