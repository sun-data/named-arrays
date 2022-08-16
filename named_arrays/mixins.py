from __future__ import annotations
from typing import TypeVar, Sequence

import abc
import dataclasses
import copy
import numpy as np
import numpy.typing as npt

__all__ = [
    'CopyableMixin',
    'NDArrayMethodsMixin'
]

CopyableT = TypeVar('CopyableT', bound='Copyable')
NDArrayMethodsMixinT = TypeVar('NDArrayMethodsMixinT', bound='NDArrayMethodsMixin')


class CopyableMixin(abc.ABC):

    def copy_shallow(self: CopyableT) -> CopyableT:
        return copy.copy(self)

    def copy(self: CopyableT) -> CopyableT:
        return copy.deepcopy(self)

    def __copy__(self: CopyableT) -> CopyableT:
        fields = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        return type(self)(**fields)

    def __deepcopy__(self: CopyableT, memodict={}) -> CopyableT:
        fields = {field.name: copy.deepcopy(getattr(self, field.name)) for field in dataclasses.fields(self)}
        return type(self)(**fields)


@dataclasses.dataclass(eq=False)
class NDArrayMethodsMixin:

    def broadcast_to(
            self: NDArrayMethodsMixinT,
            shape: dict[str, int],
    ) -> NDArrayMethodsMixinT:
        return np.broadcast_to(self, shape=shape)

    def reshape(
            self: NDArrayMethodsMixinT,
            shape: dict[str, int],
    ) -> NDArrayMethodsMixinT:
        return np.reshape(self, newshape=shape)

    def min(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            initial: npt.ArrayLike = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.min(self, axis=axis, initial=initial, where=where)

    def max(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            initial: npt.ArrayLike = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.max(self, axis=axis, initial=initial, where=where)

    def sum(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.sum(self, axis=axis, where=where)

    def ptp(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
    ) -> NDArrayMethodsMixinT:
        return np.ptp(self, axis=axis)

    def mean(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.mean(self, axis=axis, where=where)

    def std(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.std(self, axis=axis, where=where)

    def all(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.all(self, axis=axis, where=where)

    def any(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.any(self, axis=axis, where=where)

    def rms(
            self: NDArrayMethodsMixinT,
            axis: None | str | Sequence[str] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.sqrt(np.mean(np.square(self), axis=axis, where=where))
