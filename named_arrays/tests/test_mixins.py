import abc
import dataclasses
import numpy as np
import named_arrays.mixins


class AbstractTestCopyable(
    abc.ABC
):

    def test_copy_shallow(self, array: named_arrays.mixins.CopyableMixin):
        array_copy = array.copy_shallow()
        assert isinstance(array_copy, named_arrays.mixins.CopyableMixin)
        assert dataclasses.is_dataclass(array_copy)
        for field in dataclasses.fields(array_copy):
            assert getattr(array, field.name) is getattr(array_copy, field.name)

    def test_copy(self, array: named_arrays.mixins.CopyableMixin):
        array_copy = array.copy()
        assert isinstance(array_copy, named_arrays.mixins.CopyableMixin)
        assert dataclasses.is_dataclass(array_copy)
        for field in dataclasses.fields(array_copy):
            assert np.all(getattr(array, field.name) == getattr(array_copy, field.name))
