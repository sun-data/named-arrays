import pytest
import numpy as np
import scipy.spatial
import astropy.units as u
import named_arrays as na

_axis = "point"

_square = na.Cartesian2dVectorArray(
    x=na.ScalarArray(np.array([1, -1, 1, -1, 0, 0.5, -0.2]), axes=_axis),
    y=na.ScalarArray(np.array([1, -1, -1, 1, 0, -0.3, 0.4]), axes=_axis),
)
"""The corners of a square, in no order, and three points inside it."""

_square_hull = na.Cartesian2dVectorArray(
    x=na.ScalarArray(np.array([-1, 1, 1, -1]), axes="vertex"),
    y=na.ScalarArray(np.array([-1, -1, 1, 1]), axes="vertex"),
)
"""The hull of :data:`_square`, counterclockwise from its lower left corner."""


def test_convex_hull() -> None:
    result = na.geometry.convex_hull(_square, axis=_axis)
    assert np.allclose(result, _square_hull)


def test_convex_hull_ties() -> None:
    """
    Among the points with the least x-coordinate, the hull starts from the
    one with the least y-coordinate, whatever order they are given in.
    """
    points = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([-1, -1, 1, 1]), axes=_axis),
        y=na.ScalarArray(np.array([1, -1, 1, -1]), axes=_axis),
    )
    result = na.geometry.convex_hull(points, axis=_axis)
    assert np.allclose(result, _square_hull)


@pytest.mark.parametrize("axis_vertex", ["corner", _axis])
def test_convex_hull_axis_vertex(axis_vertex: str) -> None:
    """The vertices vary along the axis asked for, which may be the axis the
    points varied along."""
    result = na.geometry.convex_hull(_square, axis=_axis, axis_vertex=axis_vertex)
    expected = na.Cartesian2dVectorArray(
        x=na.ScalarArray(_square_hull.x.ndarray, axes=axis_vertex),
        y=na.ScalarArray(_square_hull.y.ndarray, axes=axis_vertex),
    )
    assert np.allclose(result, expected)


def test_convex_hull_unit() -> None:
    """The vertices keep the unit of the points, and of their x-coordinates
    if the two differ."""
    points = na.Cartesian2dVectorArray(
        x=(_square.x * u.mm).to(u.cm),
        y=_square.y * u.mm,
    )
    result = na.geometry.convex_hull(points, axis=_axis)
    assert np.allclose(result, _square_hull * u.mm)
    assert result.x.unit == u.cm
    assert result.y.unit == u.cm


def test_convex_hull_where() -> None:
    """A point left out is left out of the hull."""
    outlier = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([5]), axes=_axis),
        y=na.ScalarArray(np.array([0]), axes=_axis),
    )
    points = na.concatenate([_square, outlier], axis=_axis)
    where = na.ScalarArray(
        np.array([True] * _square.shape[_axis] + [False]),
        axes=_axis,
    )
    result = na.geometry.convex_hull(points, axis=_axis, where=where)
    assert np.allclose(result, _square_hull)


def test_convex_hull_batch() -> None:
    """
    Each set of points along a batch axis gets a hull of its own, and the
    smaller hulls are padded by repeating their last vertex.
    """
    triangle = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0, 2, 0, 0.5, 0.5, 0.1, 0.1]), axes=_axis),
        y=na.ScalarArray(np.array([0, 0, 2, 0.5, 0.2, 0.1, 1.5]), axes=_axis),
    )
    points = na.stack([_square, triangle], axis="set")
    result = na.geometry.convex_hull(points, axis=_axis)
    expected = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([[-1, 1, 1, -1], [0, 2, 0, 0]]), axes=("set", "vertex")),
        y=na.ScalarArray(np.array([[-1, -1, 1, 1], [0, 0, 2, 2]]), axes=("set", "vertex")),
    )
    assert np.allclose(result, expected)


def test_convex_hull_axes() -> None:
    """The points of a set can vary along more than one axis."""
    points = na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("px", "py"),
        num=5,
    )
    result = na.geometry.convex_hull(points, axis=("px", "py"))
    assert np.allclose(result, _square_hull)


@pytest.mark.parametrize(
    argnames="points,expected",
    argvalues=[
        (
            na.Cartesian2dVectorArray(
                x=na.ScalarArray(np.array([0, 2, 1, 3]), axes=_axis),
                y=na.ScalarArray(np.array([0, 2, 1, 3]), axes=_axis),
            ),
            na.Cartesian2dVectorArray(
                x=na.ScalarArray(np.array([0, 3]), axes="vertex"),
                y=na.ScalarArray(np.array([0, 3]), axes="vertex"),
            ),
        ),
        (
            na.Cartesian2dVectorArray(
                x=na.ScalarArray(np.array([2]), axes=_axis),
                y=na.ScalarArray(np.array([3]), axes=_axis),
            ),
            na.Cartesian2dVectorArray(
                x=na.ScalarArray(np.array([2]), axes="vertex"),
                y=na.ScalarArray(np.array([3]), axes="vertex"),
            ),
        ),
    ],
    ids=["collinear", "single"],
)
def test_convex_hull_degenerate(
    points: na.Cartesian2dVectorArray,
    expected: na.Cartesian2dVectorArray,
) -> None:
    """Collinear points have a hull of their two ends, and a single point is
    its own hull."""
    result = na.geometry.convex_hull(points, axis=_axis)
    assert np.allclose(result, expected)


def test_convex_hull_empty() -> None:
    """A set with no points has a hull of NaN, while the others are found."""
    points = na.stack([_square, _square], axis="set")
    where = na.ScalarArray(np.array([True, False]), axes="set")
    result = na.geometry.convex_hull(points, axis=_axis, where=where)
    assert np.allclose(result[dict(set=0)], _square_hull)
    assert np.all(np.isnan(result[dict(set=1)].x))
    assert np.all(np.isnan(result[dict(set=1)].y))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_convex_hull_random(seed: int) -> None:
    """The vertices are those :class:`scipy.spatial.ConvexHull` finds, in the
    same counterclockwise order."""
    num = 1000
    points = na.Cartesian2dVectorArray(
        x=na.random.normal(0, 1, shape_random={_axis: num}, seed=seed),
        y=na.random.normal(0, 2, shape_random={_axis: num}, seed=seed + 100),
    )
    result = na.geometry.convex_hull(points, axis=_axis)

    xy = np.stack([points.x.ndarray, points.y.ndarray], axis=-1)
    vertices = scipy.spatial.ConvexHull(xy).vertices
    start = np.lexsort((xy[vertices, 1], xy[vertices, 0]))[0]
    vertices = np.roll(vertices, -start)
    expected = na.Cartesian2dVectorArray(
        x=na.ScalarArray(xy[vertices, 0], axes="vertex"),
        y=na.ScalarArray(xy[vertices, 1], axes="vertex"),
    )
    assert np.allclose(result, expected)


def test_convex_hull_counterclockwise() -> None:
    """Every turn around the hull is to the left."""
    points = na.Cartesian2dVectorArray(
        x=na.random.uniform(-1, 1, shape_random={_axis: 500, "set": 4}, seed=3),
        y=na.random.uniform(-1, 1, shape_random={_axis: 500, "set": 4}, seed=4),
    )
    result = na.geometry.convex_hull(points, axis=_axis)
    num = result.shape["vertex"]
    a = result
    b = result[dict(vertex=na.ScalarArray((np.arange(num) + 1) % num, "vertex"))]
    c = result[dict(vertex=na.ScalarArray((np.arange(num) + 2) % num, "vertex"))]
    cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
    assert np.all(cross >= 0)


@pytest.mark.parametrize(
    argnames="kwargs,error",
    argvalues=[
        (dict(points=_square.x, axis=_axis), TypeError),
        (dict(points=_square, axis="foo"), ValueError),
        (
            dict(points=na.stack([_square, _square], "set"), axis=_axis, axis_vertex="set"),
            ValueError,
        ),
    ],
)
def test_convex_hull_errors(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        na.geometry.convex_hull(**kwargs)
