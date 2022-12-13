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
    'type_array',
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
    """
    Get the equivalent :attr:`numpy.ndarray.dtype` of the argument.

    If the argument is an instance of :class:`numpy.ndarray`, this function simply returns :attr:`numpy.ndarray.dtype`.
    Otherwise, this function wraps the argument in an :func:`numpy.array()` call and then evaluates the ``dtype``.

    Parameters
    ----------
    value
        Object to find the ``dtype`` of

    Returns
    -------
    ``dtype`` of the argument

    """
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


def type_array(
        *values: bool | int | float | complex | str | np.ndarray | u.Quantity | AbstractArray,
) -> Type[ArrayBase]:
    cls = None
    priority_max = 0
    for value in values:
        if isinstance(value, AbstractArray):
            cls_tmp = value.type_array
            priority_tmp = cls_tmp.__named_array_priority__
            if priority_tmp > priority_max:
                priority_max = priority_tmp
                cls = cls_tmp
    return cls


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
    The ultimate parent class for all array types defined in this package.
    """

    @property
    @abc.abstractmethod
    def __named_array_priority__(self: Self) -> float:
        """
        Attribute used to decide what type of array to return in instances where there is more than one option.

        Similar to :attr:`numpy.class.__array_priority__`

        :return: :type:`int` describing this class's array priority
        """

    @property
    @abc.abstractmethod
    def ndarray(self: Self) -> bool | int | float | complex | str | np.ndarray | u.Quantity:
        """
        Underlying data that is wrapped by this class.

        This is usually an instance of :class:`numpy.ndarray` or :class:`astropy.units.Quantity`, but it can also be a
        built-in python type such as a :class:`int`, :class:`float`, or :class:`bool`
        """

    @property
    def ndarray_normalized(self: Self) -> np.ndarray:
        """
        Similar to :attr:`ndarray` but guaranteed to be an instance of
        :class:`numpy.ndarray`.
        """
        ndarray = self.ndarray
        if not isinstance(ndarray, np.ndarray):
            ndarray = np.array(ndarray)
        return ndarray

    @property
    @abc.abstractmethod
    def axes(self: Self) -> tuple[str, ...]:
        """
        A :class:`tuple` of :class:`str` representing the names of each dimension of :attr:`ndarray`.

        Must have the same length as the number of dimensions of :attr:`ndarray`.
        """

    @property
    def axes_flattened(self: Self) -> str:
        """
        Combine :attr:`axes` into a single :class:`str`.

        This is useful for functions like :func:`numpy.flatten` which returns an array with only one dimension.
        """
        return '*'.join(self.axes)

    @property
    @abc.abstractmethod
    def shape(self: Self) -> dict[str, int]:
        """
        Shape of the array. Analogous to :attr:`numpy.ndarray.shape` but represented as a :class:`dict` where the keys
        are the axis names and the values are the axis sizes.
        """

    @property
    @abc.abstractmethod
    def ndim(self: Self) -> int:
        """
        Number of dimensions of the array. Equivalent to :attr:`numpy.ndarray.ndim`.
        """

    @property
    @abc.abstractmethod
    def size(self: Self) -> int:
        """
        Total number of elements in the array. Equivalent to :attr:`numpy.ndarray.size`
        """

    @property
    @abc.abstractmethod
    def dtype(self: Self) -> Type:
        """
        Data type of the array. Equivalent to :attr:`numpy.ndarray.dtype`
        """

    @property
    @abc.abstractmethod
    def unit(self: Self) -> None | u.Unit:
        """
        Unit associated with the array.

        If :attr:`ndarray` is an instance of :class:`astropy.units.Quantity`, return :attr:`astropy.units.Quantity.unit`,
        otherwise return :class:`None`.
        """

    @property
    def unit_normalized(self: Self) -> u.Unit:
        """
        Similar to :attr:`unit` but returns :attr:`astropy.units.dimensionless_unscaled` if :attr:`ndarray` is not an
        instance of :class:`astropy.units.Quantity`.
        """
        result = self.unit
        if result is None:
            result = u.dimensionless_unscaled
        return result

    @property
    @abc.abstractmethod
    def array(self: Self) -> ArrayBase:
        """
        Converts this array to an instance of :class:`named_arrays.ArrayBase`
        """

    @property
    @abc.abstractmethod
    def type_array(self: Self) -> Type[ArrayBase]:
        """
        The :class:`ArrayBase` type corresponding to this array
        """

    @property
    @abc.abstractmethod
    def scalar(self: Self) -> named_arrays.scalars.AbstractScalar:
        """
        Converts this array to an instance of :class:`named_arrays.AbstractScalar`
        """

    @property
    @abc.abstractmethod
    def components(self: Self) -> dict[str, AbstractArray]:
        """
        The vector components of this array expressed as a :class:`dict` where the keys are the names of the component.
        """

    @property
    @abc.abstractmethod
    def nominal(self: Self) -> AbstractArray:
        """
        The nominal value of this array.
        """

    @property
    @abc.abstractmethod
    def distribution(self: Self) -> None | AbstractArray:
        """
        The distribution of values of this array.
        """

    @property
    @abc.abstractmethod
    def centers(self: Self) -> AbstractArray:
        """
        The central value for this array. Usually returns this array unless an instance of
        :class:`named_arrays.AbstractStratifiedRandomSpace`
        """

    @abc.abstractmethod
    def astype(
            self: Self,
            dtype: Type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> Self:
        """
        Copy of the array cast to a specific data type.

        Equivalent to :meth:`numpy.ndarray.astype`.
        """

    @abc.abstractmethod
    def to(self: Self, unit: u.UnitBase) -> Self:
        """
        Convert this array to a new unit.

        Equivalent to :meth:`astropy.units.Quantity.to`.

        Parameters
        ----------
        unit
            New unit of the returned array

        Returns
        -------
            Array with :attr:`unit` set to the new value
        """

    @property
    def broadcasted(self: Self) -> Self:
        return self.broadcast_to(self.shape)

    @property
    @abc.abstractmethod
    def length(self: Self) -> named_arrays.scalars.AbstractScalar:
        """
        L2-norm of this array.
        """

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
        """
        Add new singleton axes to this array

        Parameters
        ----------
        axes
            New axes to add to the array

        Returns
        -------
        Array with new axes added
        """

    @abc.abstractmethod
    def combine_axes(
            self: Self,
            axes: Sequence[str],
            axis_new: str,
    ) -> Self:
        """
        Combine some of the axes of the array into a single new axis.

        Parameters
        ----------
        axes
            The axes to combine into a new axis
        axis_new
            The name of the new axis

        Returns
        -------
        Array with the specified axes combined
        """

    @abc.abstractmethod
    def ndarray_aligned(self: Self, shape: dict[str, int]) -> QuantityLike:
        """
        Align :attr:`ndarray` to a particular shape.

        Parameters
        ----------
        shape
            New shape to align :attr:`ndarray` to.

        Returns
        -------
        An instance of :class:`numpy.ndarray` with the axes aligned.
        """

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
        """
        Create a new empty array

        Parameters
        ----------
        shape
            shape of the new array
        dtype
            data type of the new array

        Returns
        -------
            A new empty array with the specified shape and data type
        """

    @classmethod
    @abc.abstractmethod
    def zeros(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        """
        Create a new array of zeros

        Parameters
        ----------
        shape
            shape of the new array
        dtype
            data type of the new array

        Returns
        -------
            A new array of zeros with the specified shape and data type
        """

    @classmethod
    @abc.abstractmethod
    def ones(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        """
        Create a new array of ones

        Parameters
        ----------
        shape
            shape of the new array
        dtype
            data type of the new array

        Returns
        -------
            A new array of ones with the specified shape and data type
        """

    @property
    def ndim(self: Self) -> int:
        return np.ndim(self.ndarray)

    @property
    def size(self: Self) -> int:
        return np.size(self.ndarray)

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
    def axes(self: Self) -> tuple[str, ...]:
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
    def size(self: Self) -> int:
        return self.array.size

    @property
    def shape(self: Self) -> dict[str, int]:
        return self.array.shape

    @property
    def unit(self: Self) -> float | u.Unit:
        return self.array.unit

    @property
    @abc.abstractmethod
    def axis(self: Self) -> str | AbstractArray:
        """
        The axis along which the array is parameterized
        """

    @property
    @abc.abstractmethod
    def num(self: Self) -> int | AbstractArray:
        """
        Number of elements in the parameterization
        """


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
