from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic, Sequence, Iterator, Union
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
    'broadcast_shapes',
    'shape_broadcasted',
    'ndindex',
    'indices',
    'AbstractArray',
    'ArrayBase',
    'AbstractParameterizedArray',
    'AbstractRandomMixin',
    'RandomMixin',
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

NDArrayMethodsMixinT = TypeVar('NDArrayMethodsMixinT', bound='NDArrayMethodsMixin')
DType = TypeVar('DType', bound=npt.DTypeLike)
# NDArrayT = TypeVar('NDArrayT', bound=npt.ArrayLike)
AbstractArrayT = TypeVar('AbstractArrayT', bound='AbstractArray')
ArrayBaseT = TypeVar('ArrayBaseT', bound='ArrayBase')
AbstractParameterizedArrayT = TypeVar('AbstractParameterizedArrayT', bound='AbstractParameterizedArray')
AbstractRandomMixinT = TypeVar('AbstractRandomMixinT', bound='AbstractRandomMixin')
RandomMixinT = TypeVar('RandomMixinT', bound='RandomMixin')
AbstractRangeT = TypeVar('AbstractRangeT', bound='AbstractRange')
AbstractSymmetricRangeT = TypeVar('AbstractSymmetricRangeT', bound='AbstractSymmetricRange')
AbstractLinearParameterizedArrayMixinT = TypeVar('AbstractLinearParameterizedArrayMixinT', bound='AbstractLinearParameterizedArrayMixin')
AbstractArrayRangeT = TypeVar('AbstractArrayRangeT', bound='AbstractArrayRange')
AbstractSpaceT = TypeVar('AbstractSpaceT', bound='AbstractSpace')
AbstractLinearSpaceT = TypeVar('AbstractLinearSpaceT', bound='AbstractLinearSpace')
AbstractLogarithmicSpaceT = TypeVar('AbstractLogarithmicSpaceT', bound='AbstractLogarithmicSpace')
AbstractGeometricSpaceT = TypeVar('AbstractGeometricSpaceT', bound='AbstractGeometricSpace')
AbstractWorldCoordinateSpaceT = TypeVar('AbstractWorldCoordinateSpaceT', bound='AbstractWorldCoordinateSpace')

QuantityLike = Union[int, float, complex, np.ndarray, u.Quantity]


def broadcast_shapes(*shapes: dict[str, int]) -> dict[str, int]:
    result = dict()
    for shape in shapes:
        for axis in shape:
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
    return result


def shape_broadcasted(*arrays: AbstractArray) -> dict[str, int]:
    shapes = [array.shape for array in arrays if isinstance(array, AbstractArray)]
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
    def ndarray(self: AbstractArrayT) -> npt.ArrayLike:
        pass

    @property
    def ndarray_normalized(self: AbstractArrayT) -> np.ndarray:
        ndarray = self.ndarray
        if not isinstance(ndarray, np.ndarray):
            ndarray = np.array(ndarray)
        return ndarray

    @property
    @abc.abstractmethod
    def axes(self: AbstractArrayT):
        pass

    @property
    @abc.abstractmethod
    def shape(self: AbstractArrayT) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def ndim(self: AbstractArrayT) -> int:
        pass

    @property
    @abc.abstractmethod
    def dtype(self: AbstractArrayT) -> npt.DTypeLike:
        pass

    @property
    @abc.abstractmethod
    def unit(self) -> float | u.Unit:
        pass

    @property
    @abc.abstractmethod
    def array(self: AbstractArrayT) -> ArrayBase:
        pass

    @property
    @abc.abstractmethod
    def scalar(self: AbstractArrayT) -> named_arrays.scalars.AbstractScalar:
        pass

    @property
    @abc.abstractmethod
    def components(self: AbstractArrayT) -> dict[str, AbstractArray]:
        pass

    @property
    @abc.abstractmethod
    def nominal(self: AbstractArrayT) -> AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def distribution(self: AbstractArrayT) -> None | AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def centers(self: AbstractArrayT) -> AbstractArray:
        pass

    @abc.abstractmethod
    def astype(
            self: AbstractArrayT,
            dtype: npt.DTypeLike,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> AbstractArrayT:
        pass

    @abc.abstractmethod
    def to(self: AbstractArrayT, unit: u.UnitBase) -> AbstractArrayT:
        pass

    @property
    def broadcasted(self: AbstractArrayT) -> AbstractArrayT:
        return self.broadcast_to(self.shape)

    @property
    @abc.abstractmethod
    def length(self: AbstractArrayT) -> named_arrays.scalars.AbstractScalar:
        pass

    @abc.abstractmethod
    def __getitem__(
            self: AbstractArrayT,
            item: dict[str, int | slice | AbstractArrayT] | AbstractArrayT,
    ) -> AbstractArray:
        pass

    @property
    def indices(self: AbstractArrayT) -> dict[str, AbstractArrayT]:
        return indices(self.shape)

    def ndindex(
            self: AbstractArrayT,
            axis_ignored: None | str | Sequence[str] = None,
    ) -> Iterator[dict[str, int]]:
        return ndindex(
            shape=self.shape,
            axis_ignored=axis_ignored,
        )

    @abc.abstractmethod
    def add_axes(self: AbstractArrayT, axes: str | Sequence[str]) -> AbstractArrayT:
        pass

    @abc.abstractmethod
    def combine_axes(
            self: AbstractArrayT,
            axes: Sequence[str],
            axis_new: str,
    ) -> AbstractArrayT:
        pass

    @abc.abstractmethod
    def ndarray_aligned(self: AbstractArrayT, shape: dict[str, int]) -> QuantityLike:
        pass

    def _interp_linear_recursive(
            self: AbstractArrayT,
            item: dict[str, AbstractArrayT],
            item_base: dict[str, AbstractArrayT],
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
            self: AbstractArrayT,
            item: dict[str, AbstractArrayT],
    ) -> AbstractArrayT:
        return self._interp_linear_recursive(
            item=item,
            item_base=self[{ax: 0 for ax in item}].indices,
        )

    def __call__(self: AbstractArrayT, item: dict[str, AbstractArrayT]) -> AbstractArrayT:
        return self.interp_linear(item=item)

    def index_secant(
            self: AbstractArrayT,
            value: AbstractArrayT,
            axis: None | str | Sequence[str] = None,
    ) -> dict[str, AbstractArrayT]:

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
            self: AbstractArrayT,
            value: AbstractArrayT,
            axis: None | str | Sequence[str] = None,
    ) -> dict[str, AbstractArrayT]:
        return self.index_secant(value=value, axis=axis)


ArrayLike = Union[QuantityLike, AbstractArray]


@dataclasses.dataclass(eq=False)
class ArrayBase(
    AbstractArray,
):
    @property
    @abc.abstractmethod
    def ndarray(self: ArrayBaseT) -> npt.ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def axes(self: ArrayBaseT) -> list[str]:
        pass

    @property
    def ndim(self: ArrayBaseT) -> int:
        return np.ndim(self.ndarray)

    @property
    def shape(self: ArrayBaseT) -> dict[str, int]:
        ndarray = self.ndarray
        axes = self.axes
        result = dict()
        for i in range(self.ndim):
            result[axes[i]] = ndarray.shape[i]
        return result

    @property
    def dtype(self: ArrayBaseT) -> npt.DTypeLike:
        return self.ndarray_normalized.dtype

    @property
    def unit(self: ArrayBaseT) -> float | u.Unit:
        if isinstance(self.ndarray, (u.Quantity, AbstractArray)):
            return self.ndarray.unit
        else:
            return 1


@dataclasses.dataclass
class AbstractParameterizedArray(
    AbstractArray,
):
    @property
    def axes(self: AbstractParameterizedArrayT) -> list[str]:
        return self.array.axes

    @property
    def dtype(self: AbstractParameterizedArrayT) -> npt.DTypeLike:
        return self.array.dtype

    @property
    def ndarray(self: AbstractParameterizedArrayT) -> QuantityLike:
        return self.array.ndarray

    @property
    def ndim(self: AbstractParameterizedArrayT) -> int:
        return self.array.ndim

    @property
    def shape(self: AbstractParameterizedArrayT) -> dict[str, int]:
        return self.array.shape

    @property
    def unit(self: AbstractParameterizedArrayT) -> float | u.Unit:
        return self.array.unit

    @property
    @abc.abstractmethod
    def axis(self: AbstractParameterizedArrayT) -> str | AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def num(self: AbstractParameterizedArrayT) -> int | AbstractArray:
        pass


@dataclasses.dataclass(eq=False)
class AbstractRandomMixin(
    abc.ABC,
):

    @property
    @abc.abstractmethod
    def seed(self: AbstractRandomMixinT) -> int:
        pass

    @property
    def _seed_normalized(self):
        return

    @property
    def _rng(self: AbstractRandomMixinT) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)


@dataclasses.dataclass(eq=False)
class RandomMixin(
    AbstractRandomMixin
):

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 10 ** 12)


@dataclasses.dataclass(eq=False)
class AbstractRange(
    AbstractParameterizedArray,
):

    @property
    @abc.abstractmethod
    def start(self: AbstractRangeT) -> int | AbstractArray:
        pass

    @property
    @abc.abstractmethod
    def stop(self: AbstractRangeT) -> int | AbstractArray:
        pass

    @property
    def range(self: AbstractRangeT) -> AbstractArray:
        return self.stop - self.start


@dataclasses.dataclass(eq=False)
class AbstractSymmetricRange(
    AbstractRange
):
    @property
    @abc.abstractmethod
    def center(self: AbstractSymmetricRangeT) -> ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def width(self: AbstractSymmetricRangeT) -> ArrayLike:
        pass

    @property
    def start(self: AbstractSymmetricRangeT) -> ArrayLike:
        return self.center - self.width

    @property
    def stop(self: AbstractSymmetricRangeT) -> ArrayLike:
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
    def step(self: AbstractLinearParameterizedArrayMixinT) -> int | AbstractArray:
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
    def endpoint(self: AbstractSpaceT) -> bool:
        pass


@dataclasses.dataclass(eq=False)
class AbstractLinearSpace(
    AbstractLinearParameterizedArrayMixin,
    AbstractSpace
):

    @property
    def step(self: AbstractLinearSpaceT) -> AbstractArray:
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
    def start_exponent(self: AbstractLogarithmicSpaceT) -> ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def stop_exponent(self: AbstractLogarithmicSpaceT) -> ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def base(self: AbstractLogarithmicSpaceT) -> ArrayLike:
        pass

    @property
    def start(self: AbstractLogarithmicSpaceT) -> ArrayLike:
        return self.base ** self.start_exponent

    @property
    def stop(self: AbstractLogarithmicSpaceT) -> ArrayLike:
        return self.base ** self.stop_exponent


@dataclasses.dataclass(eq=False)
class AbstractGeometricSpace(
    AbstractSpace
):
    pass
