import abc
import dataclasses
import math
import numpy as np
import pytest
import named_arrays as na


class AbstractTestIndexable(
    abc.ABC,
):
    """
    Base test class for subclasses of :class:`named_arrays.Indexable`.

    Concrete test classes inherit from this and parametrize an ``indexable``
    argument with instances of the :class:`named_arrays.Indexable` subclass
    under test. Every test derives what it needs from the instance itself
    (shape, axes, ...), so no further parameters are required. The
    ``AbstractTest`` name prefix keeps pytest from collecting this base class
    directly.
    """

    def test_shape(self, indexable: na.Indexable):
        shape = indexable.shape
        assert isinstance(shape, dict)
        for axis in shape:
            assert isinstance(axis, str)
            assert isinstance(shape[axis], int)
        # the property must agree with the top-level ``na.shape``
        assert shape == na.shape(indexable)

    def test_ndim(self, indexable: na.Indexable):
        assert indexable.ndim == len(indexable.shape)

    def test_size(self, indexable: na.Indexable):
        size = indexable.size
        assert isinstance(size, int)
        assert size == math.prod(indexable.shape.values())

    def test_axes(self, indexable: na.Indexable):
        axes = indexable.axes
        assert isinstance(axes, tuple)
        assert set(axes) == set(indexable.shape)
        for axis in axes:
            assert isinstance(axis, str)

    def test_getitem(self, indexable: na.Indexable):
        shape = indexable.shape

        if shape:
            # index one axis by an identity range along itself, which should
            # round-trip the object while preserving its shape
            axis = next(iter(shape))
            item = {axis: na.ScalarArray(np.arange(shape[axis]), axes=axis)}
        else:
            # a zero-dimensional object can only be indexed by an empty dict
            item = {}

        result = indexable[item]

        # a new instance of the same type, original left unchanged (immutable)
        assert type(result) is type(indexable)
        assert result is not indexable
        assert result.shape == shape
        assert indexable.shape == shape

    def test_isel(self, indexable: na.Indexable):
        shape = indexable.shape
        if not shape:
            return
        # ``isel`` is keyword sugar for dict-indexing along named axes
        axis = next(iter(shape))
        item = {axis: na.ScalarArray(np.arange(shape[axis]), axes=axis)}
        result = indexable.isel(**item)
        assert type(result) is type(indexable)
        assert result.shape == indexable[item].shape


@dataclasses.dataclass(eq=False)
class _Point(na.Indexable):
    """A minimal composite type used to validate ``AbstractTestIndexable``."""
    x: na.AbstractScalar
    y: na.AbstractScalar
    label: str = "point"


@pytest.mark.parametrize(
    argnames="indexable",
    argvalues=[
        _Point(
            x=na.arange(0, 5, axis="t"),
            y=na.arange(0, 3, axis="w"),
        ),
        _Point(
            x=na.ScalarArray(2),
            y=na.ScalarArray(3),
        ),
    ],
)
class TestIndexable(
    AbstractTestIndexable,
):
    pass
