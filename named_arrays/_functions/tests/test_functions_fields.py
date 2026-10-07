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


@dataclasses.dataclass(eq=False, repr=False)
class _Masked(na.FunctionArray):
    """A function with a mask of the samples to use, unrelated to images."""

    where: na.AbstractScalar | bool = True
    """Whether to use each sample."""


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
    images = _images()
    return images.replace(
        inputs=images.inputs.y,
        outputs=images.outputs[dict(x=0)],
    )


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


class TestReshape:

    def test_reshape(self):
        images = _images_center()
        # stored in the other order, which must not change which image each
        # exposure time belongs to
        images = images.replace(timedelta=np.transpose(images.timedelta, axes=("w", "t")))
        result = np.reshape(images, dict(tw=_num_t * _num_w))
        assert isinstance(result, _Images)
        assert result.label == images.label
        assert result.timedelta.shape == dict(tw=_num_t * _num_w)
        # in `_images`, the exposure time of each image is one more than its
        # first output divided by the number of pixels
        expected = result.outputs.to_value(u.DN) / _num_x + 1
        assert np.all(result.timedelta.to_value(u.s) == expected)

    def test_reshape_constant(self):
        images = _images_center().replace(timedelta=na.ScalarArray(5 * u.s))
        result = np.reshape(images, dict(tw=_num_t * _num_w))
        assert result.timedelta is images.timedelta

    def test_reshape_outside_axis(self):
        images = _images_center()
        timedelta = images.timedelta.add_axes("line")
        images = images.replace(timedelta=timedelta)
        with pytest.raises(ValueError, match="`timedelta`.*cannot be reshaped"):
            np.reshape(images, dict(tw=_num_t * _num_w))


class TestSetitem:

    def test_setitem(self):
        images = _images()
        other = _images(scale=10)
        result = images.copy()
        result[dict(t=0)] = other[dict(t=1)]
        assert np.all(result.timedelta[dict(t=0)] == other.timedelta[dict(t=1)])
        assert np.all(result.timedelta[dict(t=1)] == images.timedelta[dict(t=1)])

    def test_setitem_view(self):
        # a slice is a view of both the outputs and the fields, so assigning
        # into it writes both through to the array it was taken from
        images = _images()
        other = _images(scale=10)
        view = images[dict(t=slice(0, 2))]
        view[dict(t=0)] = other[dict(t=1)]
        assert np.all(images.outputs[dict(t=0)] == other.outputs[dict(t=1)])
        assert np.all(images.timedelta[dict(t=0)] == other.timedelta[dict(t=1)])

    def test_setitem_shared(self):
        # the result of a ufunc has new outputs, but the same fields as the
        # array it was computed from, so assigning into it changes the fields
        # of that array too, as documented
        images = _images()
        other = _images(scale=10)
        result = images + 0 * u.DN
        assert result.timedelta is images.timedelta
        result[dict(t=0)] = other[dict(t=1)]
        assert np.all(images.timedelta[dict(t=0)] == other.timedelta[dict(t=1)])
        assert np.all(images.outputs == _images().outputs)

    def test_setitem_scalar(self):
        # a value without fields leaves the fields alone
        images = _images()
        result = images.copy()
        result[dict(t=0)] = 0 * u.DN
        assert np.all(result.timedelta == images.timedelta)

    def test_setitem_missing(self):
        images = _images()
        images = images.replace(timedelta=images.timedelta[dict(t=0)])
        outputs = images.outputs.copy()
        with pytest.raises(ValueError, match="does not vary along"):
            images[dict(t=0)] = _images(scale=10)[dict(t=1)]
        # nothing was written
        assert np.all(images.outputs == outputs)

    def test_setitem_missing_equal(self):
        # a field which does not vary along the axis can be given the value it
        # already has
        images = _images()
        images = images.replace(timedelta=images.timedelta[dict(t=0)])
        value = images[dict(t=1)].replace(outputs=_images(scale=10).outputs[dict(t=1)])
        images[dict(t=0)] = value
        assert np.all(images.outputs[dict(t=0)] == value.outputs)

    def test_setitem_extra_axis(self):
        # the value varies along an axis which the field does not have
        images = _images()
        images = images.replace(timedelta=images.timedelta[dict(w=0)])
        with pytest.raises(ValueError, match="of the value varies along"):
            images[dict(t=0)] = _images(scale=10)[dict(t=1)]

    def test_setitem_scalar_field(self):
        # a scalar field of this array cannot hold a value for one image only
        images = _images().replace(timedelta=0 * u.s)
        value = _images(scale=10)[dict(t=1)]
        value = value.replace(timedelta=value.timedelta[dict(w=0)])
        with pytest.raises(ValueError, match="does not vary along"):
            images[dict(t=0)] = value

    def test_setitem_scalar_value(self):
        # a scalar field of the value is the same for every element written
        images = _images()
        value = _images(scale=10)[dict(t=1)].replace(timedelta=7 * u.s)
        images[dict(t=0)] = value
        assert np.all(images.timedelta[dict(t=0)] == 7 * u.s)
        assert np.all(images.timedelta[dict(t=1)] == _images().timedelta[dict(t=1)])

    def test_setitem_plain(self):
        # an array without the fields of the value just takes its outputs
        images = _images()
        plain = na.FunctionArray(images.inputs, images.outputs.copy())
        plain[dict(t=0)] = _images(scale=10)[dict(t=1)]
        assert np.all(plain.outputs[dict(t=0)] == _images(scale=10).outputs[dict(t=1)])

    def test_setitem_simple_differs(self):
        images = _images()
        value = _images(scale=10)[dict(t=1)].replace(label="other")
        with pytest.raises(ValueError, match="`label` of the value is 'other'"):
            images[dict(t=0)] = value

    def test_setitem_scalar_differs(self):
        # a scalar field on both sides cannot hold a value for one image only
        images = _images().replace(timedelta=0 * u.s)
        value = _images(scale=10)[dict(t=1)].replace(timedelta=5 * u.s)
        with pytest.raises(ValueError, match="`timedelta` of the value is"):
            images[dict(t=0)] = value
        images[dict(t=0)] = value.replace(timedelta=0 * u.ms)

    def test_setitem_unknown_axis(self):
        # the outputs raise for an axis which the array does not have, and the
        # fields do not raise first with a misleading message
        images = _images()
        with pytest.raises(ValueError, match="must be a subset"):
            images[dict(t=0, z=0)] = _images(scale=10)[dict(t=1)]

    def test_setitem_mask(self):
        images = _images_center()
        # assigning with a mask also writes the inputs, so they must have
        # every axis of the mask
        images = images.replace(inputs=na.broadcast_to(images.inputs, images.outputs.shape).copy())
        other = images.replace(outputs=10 * images.outputs, timedelta=10 * images.timedelta)
        mask = na.FunctionArray(images.inputs, images.outputs > 20 * u.DN)
        images[mask] = other[mask]
        expected = np.where(mask.outputs, other.timedelta, _images_center().timedelta)
        assert np.all(images.timedelta == expected)


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

    def test_keepdims_outputs_axis(self, func: Callable):
        # keeping the axes keeps only the axes of the inputs, so an axis of
        # the outputs which the inputs do not have is reduced to one element
        images = _images_center()
        with pytest.raises(ValueError, match="keeps only the axes of the inputs"):
            func(images, axis="w")
        images = images.replace(timedelta=images.timedelta.sum("w"))
        result = func(images, axis="w")
        assert np.all(result.timedelta == images.timedelta)

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


class TestGetitemConstant:

    def test_dict(self):
        # a field without the indexed axis is passed on as it is
        images = _images().replace(timedelta=na.ScalarArray(5 * u.s))
        result = images[dict(t=1)]
        assert result.timedelta is images.timedelta

    def test_mask(self):
        images = _images_center().replace(timedelta=na.ScalarArray(5 * u.s))
        result = images[na.FunctionArray(images.inputs, images.outputs > 20 * u.DN)]
        assert result.timedelta is images.timedelta


class TestDebroadcast:

    def _images(self, timedelta: na.AbstractScalar) -> _Images:
        """Images which are the same along time, with the given exposure times."""
        outputs = na.ScalarArray(np.arange(_num_w, dtype=float), axes="w") * u.DN
        return _Images(
            inputs=na.linspace(0, 1, axis="w", num=_num_w),
            outputs=na.broadcast_to(outputs, dict(t=_num_t, w=_num_w)),
            timedelta=timedelta,
        )

    def test_varies(self):
        # the exposure time varies along time, so time is kept
        timedelta = na.ScalarArray(np.arange(1, 1 + _num_t) * u.s, axes="t")
        result = na.debroadcast(self._images(timedelta))
        assert "t" in result.shape
        assert np.all(result.timedelta == timedelta)

    def test_constant(self):
        timedelta = na.broadcast_to(na.ScalarArray(2 * u.s), dict(t=_num_t))
        result = na.debroadcast(self._images(timedelta))
        assert "t" not in result.shape
        assert np.all(result.timedelta == 2 * u.s)


class TestCellCenters:

    def test_cell_centers(self):
        images = _images_center()
        result = images.cell_centers("t")
        assert result.timedelta.shape["t"] == _num_t - 1
        assert np.all(result.timedelta == images.timedelta.cell_centers("t"))

    def test_random(self):
        images = _images_center()
        with pytest.raises(ValueError, match="random=True"):
            images.cell_centers("t", random=True)
        # a field which does not vary along the axis is passed on
        images = images.replace(timedelta=images.timedelta[dict(w=0)])
        result = images.cell_centers("w", random=True, seed=0)
        assert result.timedelta is images.timedelta

    def test_not_inexact(self):
        images = _images_center()
        masked = _Masked(images.inputs, images.outputs, where=images.outputs > 20 * u.DN)
        with pytest.raises(ValueError, match="not floating-point numbers"):
            masked.cell_centers("t")


def test_regrid():
    images = _images_center()
    inputs = na.linspace(0, 30, axis="t", num=2 * _num_t) * u.s
    with pytest.raises(ValueError, match=r"regrid resamples.*timedelta\.mean"):
        images.regrid(inputs, axis="t")


class TestCheck:

    def test_mismatch(self):
        images = _images()
        with pytest.raises(ValueError, match="has 3 elements along 't'"):
            images.replace(timedelta=images.timedelta[dict(t=slice(0, 3))])

    def test_implicit(self):
        # finding the shape of implicit outputs would compute them, so the
        # check waits until they are explicit
        images = _Images(
            inputs=na.linspace(0, 30, axis="t", num=_num_t) * u.s,
            outputs=na.ScalarLinearSpace(0 * u.DN, 1 * u.DN, axis="t", num=_num_t),
            timedelta=_images().timedelta[dict(t=slice(0, 3))],
        )
        with pytest.raises(ValueError, match="has 3 elements along 't'"):
            images.explicit

    def test_outputs_reduced(self):
        # the outputs of a reduction which keeps its axes have one element
        # along them, while the inputs and the fields keep theirs
        images = _images()
        result = images.replace(outputs=np.sum(images.outputs, axis="t", keepdims=True))
        assert np.all(result.timedelta == images.timedelta)


class TestOut:

    def _out(self) -> _Images:
        """An array to write results into, with exposure times of its own."""
        out = _images_center()
        return out.replace(timedelta=99 * u.s + 0 * out.timedelta)

    def test_reduction(self):
        images = _images_center()
        out = self._out()
        out = out.replace(outputs=np.sum(out.outputs, axis="t", keepdims=True))
        result = np.sum(images, axis="t", out=out)
        assert result is out
        assert np.all(result.timedelta == images.timedelta)

    def test_cumulative(self):
        images = _images_center()
        result = np.cumsum(images, axis="w", out=self._out())
        assert np.all(result.timedelta == images.timedelta)

    def test_ufunc(self):
        images = _images_center()
        out = self._out()
        result = np.add(images, 1 * u.DN, out=(out,))
        assert result is out
        assert np.all(out.timedelta == images.timedelta)

    def test_stack(self):
        a = _images_center().replace(timedelta=5 * u.s)
        out = np.stack([self._out(), self._out()], axis="s")
        result = np.stack([a, a], axis="s", out=out)
        assert result is out
        assert out.timedelta == 5 * u.s

    def test_stack_other_type(self):
        # an `out` of a type without the fields does not get them
        a = _images_center()
        out = na.FunctionArray(
            inputs=np.stack([a.inputs, a.inputs], axis="s"),
            outputs=np.stack([a.outputs, a.outputs], axis="s"),
        )
        np.stack([a, a], axis="s", out=out)
        assert not hasattr(out, "timedelta")

    def test_clip(self):
        images = _images_center()
        result = np.clip(images, 0 * u.DN, 10 * u.DN, out=self._out())
        assert np.all(result.timedelta == images.timedelta)

    def test_round(self):
        images = _images_center()
        result = np.round(images, out=self._out())
        assert np.all(result.timedelta == images.timedelta)

    def test_matmul(self):
        images = _images_center()
        out = self._out()
        result = np.matmul(images, 2, out=out)
        assert result is out
        assert np.all(result.timedelta == images.timedelta)

    def test_copyto(self):
        images = _images_center()
        dst = self._out()
        np.copyto(dst=dst, src=images)
        assert np.all(dst.timedelta == images.timedelta)

    def test_copyto_simple_differs(self):
        images = _images_center()
        with pytest.raises(ValueError, match="`label` of the value is 'other'"):
            np.copyto(dst=self._out(), src=images.replace(label="other"))

    def test_copyto_where(self):
        images = _images_center()
        images = images.replace(inputs=na.broadcast_to(images.inputs, images.outputs.shape).copy())
        dst = images.replace(outputs=0 * images.outputs, timedelta=0 * images.timedelta)
        where = na.FunctionArray(images.inputs, images.outputs > 20 * u.DN)
        np.copyto(dst=dst, src=images, where=where)
        expected = np.where(where.outputs, images.timedelta, 0 * u.s)
        assert np.all(dst.timedelta == expected)


class TestCombine:

    def test_concatenate_reversed(self):
        # the result takes the type of the subclass, whichever comes first
        a = _images().replace(timedelta=5 * u.s)
        plain = na.FunctionArray(a.inputs, a.outputs)
        result = np.concatenate([plain, a], axis="t")
        assert isinstance(result, _Images)
        assert result.timedelta == 5 * u.s

    def test_simple_differs(self):
        a = _images()
        b = a.replace(label="other")
        with pytest.raises(ValueError, match="`label` differs"):
            np.stack([a, b], axis="s")

    def test_simple_equal(self):
        a = _images()
        result = np.stack([a, a.replace(outputs=2 * a.outputs)], axis="s")
        assert result.label == a.label

    def test_simple_nan(self):
        # a field which is NaN in every array is the same in every array
        a = _images().replace(timedelta=np.nan * u.s)
        b = a.replace(timedelta=np.nan * u.s)
        result = np.stack([a, b], axis="s")
        assert np.isnan(result.timedelta)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_unrelated_types(self, reverse: bool):
        # neither type has the fields of the other, so the result would lose
        # some, whichever comes first
        a = _images()
        masked = _Masked(a.inputs, a.outputs, where=a.outputs > 20 * u.DN)
        arrays = [a, masked][::-1] if reverse else [a, masked]
        with pytest.raises(TypeError, match="none of them is a subclass"):
            np.concatenate(arrays, axis="t")


def test_reduction_message():
    images = _images()
    with pytest.raises(ValueError, match="a sum or mean of an exposure time"):
        np.sum(images, axis="t", keepdims=False)


def test_asarray():
    images = _images()
    result = na.asarray(images, like=images)
    assert np.all(result.timedelta == images.timedelta)
    assert result.label == images.label


def test_polynomial_mask():
    # the mask of the points used for a fit follows the fit when it is indexed
    inputs = na.linspace(0, 1, axis="x", num=5)
    fit = na.PolynomialFitFunctionArray.from_degree(
        inputs=inputs,
        outputs=2 * inputs,
        degree=1,
        axis_polynomial="x",
        where_polynomial=inputs < 0.8,
    )
    result = fit[dict(x=slice(1, 4))]
    assert np.all(result.where_polynomial == fit.where_polynomial[dict(x=slice(1, 4))])

