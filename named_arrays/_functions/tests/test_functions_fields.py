"""
The extra fields of a subclass of :class:`named_arrays.FunctionArray` follow
the function through the operations which select, rearrange, or combine its
elements.
"""

from typing import Callable
import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = []

_num_t = 4
_num_w = 3
_num_x = 5


@dataclasses.dataclass(eq=False, repr=False)
class _Images(na.FunctionArray):
    """A sequence of images with an exposure time for each."""

    timedelta: na.AbstractScalar | u.Quantity = 0 * u.s
    """The exposure time of each image."""

    label: str = "images"
    """A field which is not a named array."""


def _images(scale: float = 1) -> _Images:
    """Images along time, channel, and the vertices of a detector axis."""
    outputs = na.ScalarArray(
        ndarray=np.arange(_num_t * _num_w * _num_x, dtype=float).reshape(_num_t, _num_w, _num_x),
        axes=("t", "w", "x"),
    )
    timedelta = na.ScalarArray(
        ndarray=np.arange(1, 1 + _num_t * _num_w, dtype=float).reshape(_num_t, _num_w),
        axes=("t", "w"),
    )
    return _Images(
        inputs=na.Cartesian2dVectorArray(
            x=na.linspace(0, 1, axis="x", num=_num_x + 1) * u.arcsec,
            y=na.linspace(0, 30, axis="t", num=_num_t) * u.s,
        ),
        outputs=scale * outputs * u.DN,
        timedelta=scale * timedelta * u.s,
    )


def _images_center() -> _Images:
    """Images along time and channel only, with no vertex axes."""
    images = _images()[dict(x=0)]
    return images.replace(inputs=images.inputs.y)


@pytest.mark.parametrize(
    argnames="item",
    argvalues=[
        dict(t=1),
        dict(t=slice(1, 3)),
        dict(t=na.ScalarArray(np.array([2, 0]), axes="t")),
        dict(t=0, w=1),
        dict(x=slice(1, 3)),
        dict(t=1, x=2),
    ],
)
class TestGetitem:

    def test_getitem(self, item: dict):
        images = _images()
        result = images[item]
        index = {ax: item[ax] for ax in item if ax in images.timedelta.shape}
        assert isinstance(result, _Images)
        assert np.all(result.timedelta == images.timedelta[index])
        assert set(result.timedelta.shape).issubset(result.shape)
        assert result.label == images.label

    def test_isel(self, item: dict):
        images = _images()
        result = images.isel(**item)
        assert np.all(result.timedelta == images[item].timedelta)


def test_getitem_mask():
    images = _images_center()
    mask = images.outputs > 20 * u.DN
    result = images[na.FunctionArray(images.inputs, mask)]
    expected = na.broadcast_to(images.timedelta, images.outputs.shape)[mask]
    assert np.all(result.timedelta == expected)
    assert result.timedelta.shape == result.outputs.shape


def test_stack():
    a = _images()
    b = _images(scale=10)
    result = np.stack([a, b], axis="s")
    assert isinstance(result, _Images)
    assert np.all(result.timedelta[dict(s=0)] == a.timedelta)
    assert np.all(result.timedelta[dict(s=1)] == b.timedelta)


def test_concatenate():
    a = _images()
    b = _images(scale=10)
    result = np.concatenate([a, b], axis="t")
    assert result.timedelta.shape["t"] == 2 * _num_t
    assert np.all(result.timedelta[dict(t=slice(None, _num_t))] == a.timedelta)
    assert np.all(result.timedelta[dict(t=slice(_num_t, None))] == b.timedelta)


def test_concatenate_constant():
    # an exposure time which is the same for every image of both arrays
    # stays the same, and is not broadcast along the concatenated axis
    a = _images()
    a = a.replace(timedelta=a.timedelta[dict(t=0)])
    b = a.replace(outputs=10 * a.outputs)
    result = np.concatenate([a, b], axis="t")
    assert "t" not in result.timedelta.shape
    assert np.all(result.timedelta == a.timedelta)


def test_concatenate_broadcast():
    # an exposure time which does not vary along time in either array, but
    # differs between them, is broadcast along time
    a = _images()
    a = a.replace(timedelta=a.timedelta[dict(t=0)])
    b = a.replace(timedelta=2 * a.timedelta)
    result = np.concatenate([a, b], axis="t")
    assert result.timedelta.shape["t"] == 2 * _num_t
    assert np.all(result.timedelta[dict(t=0)] == a.timedelta)
    assert np.all(result.timedelta[dict(t=_num_t)] == b.timedelta)


def test_concatenate_scalar():
    # a scalar exposure time is the same for every image
    a = _images()
    b = a.replace(timedelta=7 * u.s)
    result = np.concatenate([a, b], axis="t")
    assert np.all(result.timedelta[dict(t=slice(None, _num_t))] == a.timedelta)
    assert np.all(result.timedelta[dict(t=slice(_num_t, None))] == 7 * u.s)


def test_concatenate_not_array():
    a = _images()
    b = a.replace(timedelta=None)
    with pytest.raises(ValueError, match="timedelta"):
        np.concatenate([a, b], axis="t")


def test_concatenate_other_type():
    # an array without the field can be combined with one whose field is
    # the same for every element, but not with one whose field varies
    a = _images()
    plain = na.FunctionArray(a.inputs, a.outputs)
    constant = a.replace(timedelta=5 * u.s)
    result = np.concatenate([constant, plain], axis="t")
    assert isinstance(result, _Images)
    assert result.timedelta == 5 * u.s
    with pytest.raises(ValueError, match="timedelta"):
        np.concatenate([a, plain], axis="t")


def test_moveaxis():
    images = _images()
    result = np.moveaxis(images, "t", "time")
    assert "time" in result.timedelta.shape
    assert "t" not in result.timedelta.shape
    assert np.all(result.timedelta.ndarray == images.timedelta.ndarray)


def test_repeat():
    images = _images()
    result = np.repeat(images, 2, axis="t")
    assert result.timedelta.shape["t"] == 2 * _num_t
    assert np.all(result.timedelta[dict(t=1)] == images.timedelta[dict(t=0)])


def test_take_along_axis():
    images = _images_center()
    indices = na.ScalarArray(np.array([3, 0]), axes="t")
    result = np.take_along_axis(images, indices, axis="t")
    assert np.all(result.timedelta == images.timedelta[dict(t=indices)])


def test_combine_axes():
    images = _images()
    result = images.combine_axes(("t", "w"), "tw")
    assert result.timedelta.shape == dict(tw=_num_t * _num_w)
    assert np.all(result.timedelta == images.timedelta.combine_axes(("t", "w"), "tw"))


class TestSetitem:

    def test_setitem(self):
        images = _images()
        other = _images(scale=10)
        result = images.copy()
        result[dict(t=0)] = other[dict(t=1)]
        assert np.all(result.timedelta[dict(t=0)] == other.timedelta[dict(t=1)])
        assert np.all(result.timedelta[dict(t=1)] == images.timedelta[dict(t=1)])

    def test_setitem_shared(self):
        # an array which shares its exposure times with another does not
        # change the other when it is assigned to
        images = _images()
        result = images + 0 * u.DN
        assert result.timedelta is images.timedelta
        result[dict(t=0)] = _images(scale=10)[dict(t=1)]
        assert np.all(images.timedelta == _images().timedelta)

    def test_setitem_scalar(self):
        # a value without fields leaves the fields alone
        images = _images()
        result = images.copy()
        result[dict(t=0)] = 0 * u.DN
        assert np.all(result.timedelta == images.timedelta)

    def test_setitem_missing(self):
        images = _images()
        images = images.replace(timedelta=images.timedelta[dict(t=0)])
        with pytest.raises(ValueError, match="does not vary along"):
            images[dict(t=0)] = _images(scale=10)[dict(t=1)]


@pytest.mark.parametrize(
    argnames="func",
    argvalues=[np.sum, np.mean, np.median, np.max],
)
class TestReduction:

    def test_keepdims(self, func: Callable):
        # keeping the axes keeps the inputs, so the fields still match them
        images = _images()
        result = func(images, axis="t")
        assert np.all(result.timedelta == images.timedelta)

    def test_removes_axis(self, func: Callable):
        images = _images()
        with pytest.raises(ValueError, match="timedelta"):
            func(images, axis="t", keepdims=False)

    def test_reduced_first(self, func: Callable):
        images = _images()
        images = dataclasses.replace(images, timedelta=images.timedelta.sum("t"))
        result = func(images, axis="t", keepdims=False)
        assert "t" not in result.shape
        assert np.all(result.timedelta == images.timedelta)

    def test_other_axis(self, func: Callable):
        # the exposure time does not vary along the detector axis
        images = _images_center().replace(timedelta=_images_center().timedelta[dict(w=0)])
        result = func(images, axis="w", keepdims=False)
        assert np.all(result.timedelta == images.timedelta)


def test_percentile():
    images = _images()
    with pytest.raises(ValueError, match="timedelta"):
        np.percentile(images, 50 * u.percent, axis="t")
    result = np.percentile(images, 50 * u.percent, axis="t", keepdims=True)
    assert np.all(result.timedelta == images.timedelta)


def test_integrate():
    images = _images()
    with pytest.raises(ValueError, match="timedelta"):
        images.integrate("t", component="y")
    result = images.integrate("x", component="x")
    assert np.all(result.timedelta == images.timedelta)


def test_unchanged():
    # the operations which keep the shape of the function pass the fields on
    images = _images()
    for result in [
        images + 1 * u.DN,
        -images,
        images.to(u.DN),
        images.explicit,
        np.transpose(images),
        na.broadcast_to(images, images.shape | dict(s=2)),
        images.cell_centers("x"),
    ]:
        assert isinstance(result, _Images)
        assert np.all(result.timedelta == images.timedelta)
        assert result.label == images.label
