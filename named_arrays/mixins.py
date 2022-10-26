from typing import Sequence
from typing_extensions import Self

import abc
import dataclasses
import copy
import numpy as np
import numpy.typing as npt

__all__ = [
    'CopyableMixin',
    'NDArrayMethodsMixin'
]


class CopyableMixin(abc.ABC):

    def copy_shallow(self: Self) -> Self:
        return copy.copy(self)

    def copy(self: Self) -> Self:
        return copy.deepcopy(self)

    def __copy__(self: Self) -> Self:
        fields = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        return type(self)(**fields)

    def __deepcopy__(self: Self, memodict={}) -> Self:
        fields = {field.name: copy.deepcopy(getattr(self, field.name)) for field in dataclasses.fields(self)}
        return type(self)(**fields)


@dataclasses.dataclass(eq=False)
class NDArrayMethodsMixin:

    def broadcast_to(
            self: Self,
            shape: dict[str, int],
    ) -> Self:
        return np.broadcast_to(self, shape=shape)

    def reshape(
            self: Self,
            shape: dict[str, int],
    ) -> Self:
        return np.reshape(self, newshape=shape)

    def min(
            self: Self,
            axis: None | str | Sequence[str] = None,
            initial: npt.ArrayLike = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.min(self, axis=axis, initial=initial, where=where)

    def max(
            self: Self,
            axis: None | str | Sequence[str] = None,
            initial: npt.ArrayLike = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.max(self, axis=axis, initial=initial, where=where)

    def sum(
            self: Self,
            axis: None | str | Sequence[str] = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.sum(self, axis=axis, where=where)

    def ptp(
            self: Self,
            axis: None | str | Sequence[str] = None,
    ) -> Self:
        return np.ptp(self, axis=axis)

    def mean(
            self: Self,
            axis: None | str | Sequence[str] = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.mean(self, axis=axis, where=where)

    def std(
            self: Self,
            axis: None | str | Sequence[str] = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.std(self, axis=axis, where=where)

    def all(
            self: Self,
            axis: None | str | Sequence[str] = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.all(self, axis=axis, where=where)

    def any(
            self: Self,
            axis: None | str | Sequence[str] = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.any(self, axis=axis, where=where)

    def rms(
            self: Self,
            axis: None | str | Sequence[str] = None,
            where: Self = np._NoValue,
    ) -> Self:
        return np.sqrt(np.mean(np.square(self), axis=axis, where=where))
