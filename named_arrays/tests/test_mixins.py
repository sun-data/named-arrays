import pytest
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import named_arrays.mixins


class AbstractTestCopyable(
    abc.ABC
):

    def test_copy_shallow(self: Self, array: named_arrays.mixins.CopyableMixin):
        array_copy = array.copy_shallow()
        assert isinstance(array_copy, named_arrays.mixins.CopyableMixin)
        assert dataclasses.is_dataclass(array_copy)
        for field in dataclasses.fields(array_copy):
            assert getattr(array, field.name) is getattr(array_copy, field.name)

    def test_copy(self: Self, array: named_arrays.mixins.CopyableMixin):
        array_copy = array.copy()
        assert isinstance(array_copy, named_arrays.mixins.CopyableMixin)
        assert dataclasses.is_dataclass(array_copy)
        for field in dataclasses.fields(array_copy):
            assert np.all(getattr(array, field.name) == getattr(array_copy, field.name))


class AbstractTestNDArrayMethodsMixin(
    abc.ABC,
):

    def test_broadcast_to(
            self: Self,
            array: named_arrays.mixins.NDArrayMethodsMixin,
            shape: dict[str, int],
    ):
        assert np.all(array.broadcast_to(shape) == np.broadcast_to(array, shape))

    def test_reshape(
            self,
            array: named_arrays.mixins.NDArrayMethodsMixin,
            shape: dict[str, int],
    ):
        assert np.all(array.reshape(shape) == np.reshape(array, shape))

    def test_min(
            self: Self,
            array: named_arrays.mixins.NDArrayMethodsMixin,
    ):
        assert np.all(array.min() == np.min(array))

    def test_max(
            self: Self,
            array: named_arrays.mixins.NDArrayMethodsMixin,
    ):
        assert np.all(array.max() == np.max(array))

    def test_sum(
            self: Self,
            array: named_arrays.mixins.NDArrayMethodsMixin,
    ):
        assert np.all(array.sum() == np.sum(array))

    def test_ptp(
            self: Self,
            array: named_arrays.mixins.NDArrayMethodsMixin,
    ):
        if not np.issubdtype(array.dtype, bool):
            assert np.all(array.ptp() == np.ptp(array))
        else:
            with pytest.raises(TypeError, match='numpy boolean subtract, .*'):
                array.ptp()
