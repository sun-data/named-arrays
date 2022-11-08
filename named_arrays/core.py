from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic, Sequence, Iterator, Union, Type
from typing_extensions import Self
if TYPE_CHECKING:
    import named_arrays.scalars
    import named_arrays.vectors
    import named_arrays.matrices

import abc
import dataclasses
import copy
import random
import numpy as np
import numpy.typing as npt
import astropy.units as u

import named_arrays.mixins

__all__ = [
    'QuantityLike',
    'get_dtype',
    'unit',
    'broadcast_shapes',
    'shape_broadcasted',
    'ndindex',
    'indices',
    'AbstractArray',
    'ArrayBase',
    'AbstractParameterizedArray',
    'AbstractRandomMixin',
    'AbstractRange',
    'AbstractSymmetricRange',
    'AbstractLinearParameterizedArrayMixin',
    'AbstractArrayRange',
    'AbstractSpace',
    'AbstractLinearSpace',
    'AbstractStratifiedRandomSpace',
    'AbstractLogarithmicSpace',
    'AbstractGeometricSpace',
    'AbstractUniformRandomSample',
    'AbstractNormalRandomSample',
]

QuantityLike = Union[int, float, complex, np.ndarray, u.Quantity]


def get_dtype(
        value: bool | int | float | complex | str | np.ndarray | AbstractArray,
) -> Type:
    if isinstance(value, (np.ndarray, AbstractArray)):
        return value.dtype
    else:
        return np.array(value).dtype


def unit(
        value: bool | int | float | complex | str | np.ndarray | u.Quantity | AbstractArray
) -> None | u.UnitBase:
    if isinstance(value, (u.Quantity, AbstractArray)):
        return value.unit
    else:
        return None


def broadcast_shapes(*shapes: dict[str, int]) -> dict[str, int]:
    result = dict()
    for shape in shapes:
        for axis in reversed(shape):
            if axis in result:
                if result[axis] == shape[axis]:
                    pass
                elif shape[axis] == 1:
                    pass
                elif result[axis] == 1:
                    result[axis] = shape[axis]
                else:
                    raise ValueError(f'shapes {shapes} are not compatible')
            else:
                result[axis] = shape[axis]
    result = {axis: result[axis] for axis in reversed(result)}
    return result


def shape_broadcasted(*arrays: AbstractArray) -> dict[str, int]:
    shapes = [np.shape(array) for array in arrays if isinstance(array, AbstractArray)]
    return broadcast_shapes(*shapes)


def ndindex(
        shape: dict[str, int],
        axis_ignored: None | str | Sequence[str] = None,
) -> Iterator[dict[str, int]]:

    shape = shape.copy()

    if axis_ignored is None:
        axis_ignored = []
    elif isinstance(axis_ignored, str):
        axis_ignored = [axis_ignored]

    for axis in axis_ignored:
        if axis in shape:
            shape.pop(axis)
    shape_tuple = tuple(shape.values())
    for index in np.ndindex(*shape_tuple):
        yield dict(zip(shape.keys(), index))


def indices(shape: dict[str, int]) -> dict[str, named_arrays.scalars.ScalarArrayRange]:
    import named_arrays.scalars
    return {axis: named_arrays.scalars.ScalarArrayRange(0, shape[axis], axis=axis) for axis in shape}


@dataclasses.dataclass(eq=False)
class AbstractArray(
    named_arrays.mixins.CopyableMixin,
    named_arrays.mixins.NDArrayMethodsMixin,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):
    """
    The ultimate parent class for all array types defined in this package.0
    """

    @property
    @abc.abstractmethod
    def __named_array_priority__(self: Self) -> float:
        pass

    @property
    @abc.abstractmethod
    def ndarray(self: Self) -> npt.ArrayLike:
        pass

    @property
    def ndarray_normalized(self: Self) -> np.ndarray:
        ndarray = self.ndarray
        if not isinstance(ndarray, np.ndarray):
            ndarray = np.array(ndarray)
        return ndarray

    @property
    @abc.abstractmethod
    def axes(self: Self):
        pass

    @property
    @abc.abstractmethod
    def shape(self: Self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def ndim(self: Self) -> int:
        pass

    @property
    @abc.abstractmethod
    def dtype(self: Self) -> npt.DTypeLike:
        pass

    @property
    @abc.abstractmethod
    def unit(self: Self) -> float | u.Unit:
        pass

    @property
    @abc.abstractmethod
    def array(self: Self) -> ArrayBase:
        pass

    @property
    @abc.abstractmethod
    def type_array(self: Self) -> Type[ArrayBase]:
        pass

    @property
    @abc.abstractmethod
    def scalar(self: Self) -> named_arrays.scalars.AbstractScalar:
        pass

    @property
    @abc.abstractmethod
    def components(self: Self) -> dict[str, AbstractArray]:
        pass

    @property
    @abc.abstractmethod
    def nominal(self: Self) -> AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def distribution(self: Self) -> None | AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def centers(self: Self) -> AbstractArray:
        pass

    @abc.abstractmethod
    def astype(
            self: Self,
            dtype: npt.DTypeLike,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> Self:
        pass

    @abc.abstractmethod
    def to(self: Self, unit: u.UnitBase) -> Self:
        pass

    @property
    def broadcasted(self: Self) -> Self:
        return self.broadcast_to(self.shape)

    @property
    @abc.abstractmethod
    def length(self: Self) -> named_arrays.scalars.AbstractScalar:
        pass

    @abc.abstractmethod
    def __getitem__(
            self: Self,
            item: dict[str, int | slice | AbstractArray] | AbstractArray,
    ) -> Self:
        pass

    @property
    def indices(self: Self) -> dict[str, named_arrays.scalars.ScalarArrayRange]:
        return indices(self.shape)

    def ndindex(
            self: Self,
            axis_ignored: None | str | Sequence[str] = None,
    ) -> Iterator[dict[str, int]]:
        return ndindex(
            shape=self.shape,
            axis_ignored=axis_ignored,
        )

    @abc.abstractmethod
    def add_axes(self: Self, axes: str | Sequence[str]) -> Self:
        pass

    @abc.abstractmethod
    def combine_axes(
            self: Self,
            axes: Sequence[str],
            axis_new: str,
    ) -> Self:
        pass

    @abc.abstractmethod
    def ndarray_aligned(self: Self, shape: dict[str, int]) -> QuantityLike:
        pass

    def _interp_linear_recursive(
            self: Self,
            item: dict[str, Self],
            item_base: dict[str, Self],
    ):
        item = item.copy()

        if not item:
            raise ValueError('Item must contain at least one key')

        axis = next(iter(item))
        x = item.pop(axis)

        if x.shape:
            where_below = x < 0
            where_above = (self.shape[axis] - 1) <= x

            x0 = np.floor(x).astype(int)
            x0[where_below] = 0
            x0[where_above] = self.shape[axis] - 2

        else:
            if x < 0:
                x0 = 0
            elif x >= self.shape[axis] - 1:
                x0 = self.shape[axis] - 2
            else:
                x0 = int(x)

        x1 = x0 + 1

        item_base_0 = {**item_base, axis: x0}
        item_base_1 = {**item_base, axis: x1}

        if item:
            y0 = self._interp_linear_recursive(item=item, item_base=item_base_0, )
            y1 = self._interp_linear_recursive(item=item, item_base=item_base_1, )
        else:
            y0 = self[item_base_0]
            y1 = self[item_base_1]

        result = y0 + (x - x0) * (y1 - y0)
        return result

    def interp_linear(
            self: Self,
            item: dict[str, Self],
    ) -> Self:
        return self._interp_linear_recursive(
            item=item,
            item_base=self[{ax: 0 for ax in item}].indices,
        )

    def __call__(self: Self, item: dict[str, Self]) -> Self:
        return self.interp_linear(item=item)

    def index_secant(
            self: Self,
            value: Self,
            axis: None | str | Sequence[str] = None,
    ) -> dict[str, Self]:

        import named_arrays.scalars
        import named_arrays.vectors
        import named_arrays.optimization

        if axis is None:
            axis = list(self.shape.keys())
        elif isinstance(axis, str):
            axis = [axis, ]

        shape = self.shape
        shape_nearest = named_arrays.vectors.CartesianND({ax: shape[ax] for ax in axis})

        if isinstance(self, named_arrays.vectors.VectorInterface):
            coordinates = self.coordinates
            coordinates = {comp: None if value.coordinates[comp] is None else coordinates[comp] for comp in coordinates}
            self_subspace = type(self).from_coordinates(coordinates)
        else:
            self_subspace = self

        def indices_factory(index: named_arrays.vectors.CartesianND) -> dict[str, named_arrays.scalars.Scalar]:
            return index.coordinates

        def get_index(index: named_arrays.vectors.CartesianND) -> named_arrays.vectors.CartesianND:
            index = indices_factory(index)
            print(self_subspace)
            value_new = self_subspace(index)
            diff = value_new - value
            if isinstance(diff, named_arrays.vectors.AbstractVector):
                diff = named_arrays.vectors.CartesianND({c: diff.coordinates[c] for c in diff.coordinates if diff.coordinates[c] is not None})
            return diff

        result = named_arrays.optimization.root_finding.secant(
            func=get_index,
            root_guess=shape_nearest // 2,
            step_size=named_arrays.vectors.CartesianND({ax: 1e-6 for ax in axis}),
        )

        return indices_factory(result)

    def index(
            self: Self,
            value: Self,
            axis: None | str | Sequence[str] = None,
    ) -> dict[str, Self]:
        return self.index_secant(value=value, axis=axis)


ArrayLike = Union[QuantityLike, AbstractArray]


@dataclasses.dataclass(eq=False)
class ArrayBase(
    AbstractArray,
):
    @classmethod
    @abc.abstractmethod
    def empty(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        pass

    @classmethod
    @abc.abstractmethod
    def zeros(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        pass

    @classmethod
    @abc.abstractmethod
    def ones(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        pass

    @property
    def ndim(self: Self) -> int:
        return np.ndim(self.ndarray)

    @property
    def shape(self: Self) -> dict[str, int]:
        ndarray = self.ndarray
        axes = self.axes
        result = dict()
        for i in range(self.ndim):
            result[axes[i]] = ndarray.shape[i]
        return result

    @property
    def dtype(self: Self) -> npt.DTypeLike:
        return self.ndarray_normalized.dtype

    @property
    def unit(self: Self) -> None | u.Unit:
        if isinstance(self.ndarray, (u.Quantity, AbstractArray)):
            return self.ndarray.unit
        else:
            return None


@dataclasses.dataclass
class AbstractParameterizedArray(
    AbstractArray,
):
    @property
    def axes(self: Self) -> list[str]:
        return self.array.axes

    @property
    def dtype(self: Self) -> npt.DTypeLike:
        return self.array.dtype

    @property
    def ndarray(self: Self) -> QuantityLike:
        return self.array.ndarray

    @property
    def ndim(self: Self) -> int:
        return self.array.ndim

    @property
    def shape(self: Self) -> dict[str, int]:
        return self.array.shape

    @property
    def unit(self: Self) -> float | u.Unit:
        return self.array.unit

    @property
    @abc.abstractmethod
    def axis(self: Self) -> str | AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def num(self: Self) -> int | AbstractArray:
        pass


@dataclasses.dataclass(eq=False)
class AbstractRandomMixin(
    abc.ABC,
):

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 10 ** 12)

    @property
    @abc.abstractmethod
    def seed(self: Self) -> int:
        pass

    @seed.setter
    @abc.abstractmethod
    def seed(self: Self, value: int) -> None:
        pass

    @property
    def _rng(self: Self) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)


@dataclasses.dataclass(eq=False)
class AbstractRange(
    AbstractParameterizedArray,
):

    @property
    @abc.abstractmethod
    def start(self: Self) -> int | AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def stop(self: Self) -> int | AbstractArray:
        pass

    @property
    def range(self: Self) -> AbstractArray:
        return self.stop - self.start


@dataclasses.dataclass(eq=False)
class AbstractSymmetricRange(
    AbstractRange
):
    @property
    @abc.abstractmethod
    def center(self: Self) -> ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def width(self: Self) -> ArrayLike:
        pass

    @property
    def start(self: Self) -> ArrayLike:
        return self.center - self.width

    @property
    def stop(self: Self) -> ArrayLike:
        return self.center + self.width


@dataclasses.dataclass(eq=False)
class AbstractUniformRandomSample(
    AbstractRandomMixin,
    AbstractRange,
):
    pass


@dataclasses.dataclass
class AbstractNormalRandomSample(
    AbstractRandomMixin,
    AbstractSymmetricRange,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractLinearParameterizedArrayMixin(
    abc.ABC
):
    @property
    @abc.abstractmethod
    def step(self: Self) -> int | AbstractArray:
        pass


@dataclasses.dataclass(eq=False)
class AbstractArrayRange(
    AbstractLinearParameterizedArrayMixin,
    AbstractRange,
):
    pass


@dataclasses.dataclass
class AbstractSpace(
    AbstractRange,
):
    @property
    @abc.abstractmethod
    def endpoint(self: Self) -> bool:
        pass


@dataclasses.dataclass(eq=False)
class AbstractLinearSpace(
    AbstractLinearParameterizedArrayMixin,
    AbstractSpace
):

    @property
    def step(self: Self) -> AbstractArray:
        if self.endpoint:
            return self.range / (self.num - 1)
        else:
            return self.range / self.num


@dataclasses.dataclass(eq=False)
class AbstractStratifiedRandomSpace(
    AbstractRandomMixin,
    AbstractLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractLogarithmicSpace(
    AbstractSpace
):

    @property
    @abc.abstractmethod
    def start_exponent(self: Self) -> ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def stop_exponent(self: Self) -> ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def base(self: Self) -> ArrayLike:
        pass

    @property
    def start(self: Self) -> ArrayLike:
        return self.base ** self.start_exponent

    @property
    def stop(self: Self) -> ArrayLike:
        return self.base ** self.stop_exponent


@dataclasses.dataclass(eq=False)
class AbstractGeometricSpace(
    AbstractSpace
):
    pass
