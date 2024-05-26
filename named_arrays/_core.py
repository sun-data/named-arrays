from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic, Sequence, Iterator, Union, Type, Callable, Collection, Any
from typing_extensions import Self
import abc
import dataclasses
import copy
import secrets
import numpy as np
import numpy.typing as npt
import astropy.units as u
import named_arrays as na

__all__ = [
    'QuantityLike',
    'StartT',
    'StopT',
    'named_array_like',
    'get_dtype',
    'value',
    'type_array',
    'broadcast_shapes',
    'shape_broadcasted',
    'ndindex',
    'indices',
    'flatten_axes',
    'axis_normalized',
    'explicit',
    'AbstractArray',
    'ArrayLike',
    'AbstractExplicitArray',
    'AbstractImplicitArray',
    'AbstractRandomMixin',
    'AbstractRangeMixin',
    'AbstractSymmetricRangeMixin',
    'AbstractRandomSample',
    'AbstractParameterizedArray',
    'AbstractLinearParameterizedArrayMixin',
    'AbstractArrayRange',
    'AbstractSpace',
    'AbstractLinearSpace',
    'strata',
    'AbstractStratifiedRandomSpace',
    'AbstractLogarithmicSpace',
    'AbstractGeometricSpace',
    'AbstractUniformRandomSample',
    'AbstractNormalRandomSample',
    'AbstractPoissonRandomSample',
]

QuantityLike = Union[int, float, complex, np.ndarray, u.Quantity]

AxisT = TypeVar("AxisT", bound="str | AbstractArray")
NumT = TypeVar("NumT", bound="int | AbstractArray")
StartT = TypeVar("StartT", bound="QuantityLike | AbstractArray")
StopT = TypeVar("StopT", bound="QuantityLike | AbstractArray")
CenterT = TypeVar("CenterT", bound="QuantityLike | AbstractArray")
WidthT = TypeVar("WidthT", bound="QuantityLike | AbstractArray")
StartExponentT = TypeVar("StartExponentT", bound="QuantityLike | AbstractArray")
StopExponentT = TypeVar("StopExponentT", bound="QuantityLike | AbstractArray")
BaseT = TypeVar("BaseT", bound="QuantityLike | AbstractArray")


def named_array_like(a: Any) -> bool:
    """
    Check if an object is compatible with the :mod:`named_arrays` API.

    If the object has a ``__named_array_function__`` method it is considered compatible.

    Parameters
    ----------
    a
        Object to be checked for compatibility with the :mod:`named_arrays` API.

    Examples
    --------

    Instances of :class:`named_arrays.ScalarArray` are compatible with the :mod:`named_arrays` API

    .. jupyter-execute::

        import named_arrays as na

        na.named_array_like(na.ScalarArray(2))

    But instances of :class:`numpy.ndarray` are not compatible

    .. jupyter-execute::

        import numpy as np

        na.named_array_like(np.empty(3))
    """
    if hasattr(a, "__named_array_function__"):
        return True
    else:
        return False


def get_dtype(
        value: bool | int | float | complex | str | np.ndarray | AbstractArray,
) -> np.dtype:
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
    if isinstance(value, np.ndarray):
        return value.dtype
    if isinstance(value, AbstractArray):
        if isinstance(value, na.AbstractScalar):
            return value.dtype
        else:
            raise ValueError("non-scalar instances of `na.AbstractArray` may not be represented by a single dtype")
    else:
        return np.array(value).dtype


def value(a: float | u.Quantity | AbstractArray):
    """
    Remove the units (if they exist) from the input.

    Parameters
    ----------
    a
        A numeric value that may or may not have associated units

    Returns
    -------
        The same numeric value, but guaranteed to be dimensionless.
    """

    if isinstance(a, AbstractArray):
        return a.value
    elif isinstance(a, u.Quantity):
        return a.value
    else:
        return a


def type_array(
        *values: bool | int | float | complex | str | np.ndarray | u.Quantity | AbstractArray,
) -> Type[AbstractExplicitArray]:
    cls = None
    priority_max = 0
    for value in values:
        if isinstance(value, AbstractArray):
            cls_tmp = value.type_explicit
            priority_tmp = cls_tmp.__named_array_priority__
            if priority_tmp > priority_max:
                priority_max = priority_tmp
                cls = cls_tmp
    return cls


def broadcast_shapes(*shapes: dict[str, int]) -> dict[str, int]:
    if not shapes:
        return dict()
    result = shapes[0].copy()
    for shape in shapes[1:]:
        if shape == result:
            continue
        for axis in shape:
            shape_axis = shape[axis]
            if axis in result:
                result_axis = result[axis]
                if result_axis == shape_axis:
                    pass
                elif shape_axis == 1:
                    pass
                elif result_axis == 1:
                    result[axis] = shape_axis
                else:
                    raise ValueError(f'shapes {shapes} are not compatible')
            else:
                result[axis] = shape_axis
    return result


def shape_broadcasted(*arrays: Any) -> dict[str, int]:
    shapes = [a.shape for a in arrays if hasattr(a, "__named_array_function__")]
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


def indices(shape: dict[str, int]) -> dict[str, na.ScalarArrayRange]:
    return {axis: na.ScalarArrayRange(0, shape[axis], axis=axis) for axis in shape}


def flatten_axes(axes: Sequence[str]):
    if not axes:
        raise ValueError(f"`axes` must be a non-empty sequence, got {axes}")
    return '*'.join(axes)


def axis_normalized(
        a: AbstractArray,
        axis: None | str | Sequence[str],
) -> tuple[str]:
    """
    Convert all the possible values of the ``axis`` argument to a :class:`tuple` of :class:`str`.

    :param a: If ``axis`` is :class:`None` the result is ``a.axes``.
    :param axis: The ``axis`` value to normalize.
    :return: Normalized ``axis`` parameter.
    """

    if axis is None:
        result = a.axes
    elif isinstance(axis, str):
        result = axis,
    else:
        result = tuple(axis)
    return result


def explicit(value: Any | AbstractArray):
    """
    Converts an array to its explicit version if possible.

    Parameters
    ----------
    value
        any value or an instance of :class:`AbstractArray`

    Returns
    -------
        If an instance of :class:`AbstractArray`, returns :attr:`AbstractArray.explicit` otherwise just returns the
        input value.
    """
    if isinstance(value, AbstractArray):
        return value.explicit
    else:
        return value


@dataclasses.dataclass(eq=False, repr=False)
class AbstractArray(
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
    def type_abstract(self: Self) -> Type[AbstractArray]:
        """
        The :class:`AbstractArray` type corresponding to this array
        """

    @property
    @abc.abstractmethod
    def type_explicit(self: Self) -> Type[AbstractExplicitArray]:
        """
        The :class:`AbstractExplicitArray` type corresponding to this array
        """

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
        return flatten_axes(self.axes)

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
    def value(self: Self) -> Self:
        """
        Returns a new array with its units removed, if they exist
        """

    @property
    @abc.abstractmethod
    def explicit(self: Self) -> AbstractExplicitArray:
        """
        Converts this array to an instance of :class:`named_arrays.AbstractExplicitArray`
        """

    @property
    def broadcasted(self) -> na.AbstractExplicitArray:
        """
        if this array has multiple components, broadcast them against each other.

        Equivalent to ``a.broadcast_to(a.shape)``
        """
        a = self.explicit
        return a.broadcast_to(a.shape)

    @abc.abstractmethod
    def astype(
            self: Self,
            dtype: str | np.dtype | Type,
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
    def to(
        self: Self,
        unit: u.UnitBase,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = [],
        copy: bool = True,
    ) -> Self:
        """
        Convert this array to a new unit.

        Equivalent to :meth:`astropy.units.Quantity.to`.

        Parameters
        ----------
        unit
            New unit of the returned array
        equivalencies
            A list of equivalence pairs to try if the units are not directly
            convertible.
        copy
            Boolean flag controlling whether to copy the array.
        """

    @property
    @abc.abstractmethod
    def length(self: Self) -> na.AbstractScalar:
        """
        L2-norm of this array.
        """

    @property
    def indices(self: Self) -> dict[str, na.ScalarArrayRange]:
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
    def add_axes(self: Self, axes: str | Sequence[str]) -> AbstractExplicitArray:
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
            axes: None | Sequence[str] = None,
            axis_new: None | str = None,
    ) -> AbstractExplicitArray:
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

    def volume_cell(self, axis: None | str | Sequence[str]) -> na.AbstractScalar:
        """
        Computes the n-dimensional volume of each cell formed by interpreting
        this array as a logically-rectangular grid of vertices.

        Note that this method is usually only used for sorted arrays.

        If `self` is a scalar, this method computes the length of each edge,
        and is equivalent to :func:`numpy.diff`.
        If `self` is a 2d vector, this method computes the area of each
        quadrilateral, and if `self` is a 3d vector, this method computes
        the volume of each cuboid.

        Parameters
        ----------
        axis
            The axis or axes defining the logically-rectangular grid.
            If `self` is a physical scalar, there should only be one axis.
            If `self` is a physical vector, there should be one axis for each
            component of the vector.
        """
        raise NotImplementedError

    def to_string(
            self,
            prefix: None | str = None,
            multiline: None | bool = None,
    ):
        """
        Convert this array instance to a string representation.

        Parameters
        ----------
        prefix
            the length of this string is used to align the output
        multiline
            flag which controls if the output should be spread over multiple
            lines.

        Returns
        -------
        array represented as a :class:`str`
        """
        fields = dataclasses.fields(self)

        if multiline is None:
            multiline_normalized = any(isinstance(getattr(self, f.name), (np.ndarray, na.AbstractArray)) for f in fields)
        else:
            multiline_normalized = multiline

        if multiline_normalized:
            delim_field = "\n"
            pre = " " * len(prefix) if prefix is not None else ""
            tab = " " * 4
        else:
            delim_field = " "
            pre = tab = ""

        result = f"{self.__class__.__qualname__}("
        if multiline_normalized:
            result += "\n"

        for i, f in enumerate(fields):
            field_str = f"{pre}{tab}{f.name}="
            val = getattr(self, f.name)
            if isinstance(val, AbstractArray):
                val_str = val.to_string(prefix=f"{pre}{tab}", multiline=multiline)
            elif isinstance(val, np.ndarray):
                val_str = np.array2string(
                    a=val,
                    separator=", ",
                    prefix=field_str,
                )
                if isinstance(val, u.Quantity):
                    val_str = f"{val_str} {val.unit}"
            else:
                val_str = repr(val)
            field_str += val_str
            if multiline_normalized or i < (len(fields) - 1):
                field_str += f",{delim_field}"
            result += field_str
        result += f"{pre})"
        return result

    def __repr__(self):
        return self.to_string()

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

    @abc.abstractmethod
    def _getitem(
            self: Self,
            item: dict[str, int | slice | AbstractArray] | AbstractArray,
    ):
        pass

    @abc.abstractmethod
    def _getitem_reversed(
            self: Self,
            array: AbstractArray,
            item: dict[str, int | slice | AbstractArray] | AbstractArray
    ):
        pass

    def __getitem__(
            self: Self,
            item: dict[str, int | slice | AbstractArray] | AbstractArray,
    ) -> AbstractExplicitArray:
        result = self._getitem(item)
        if result is not NotImplemented:
            return result

        else:
            if isinstance(item, dict):
                for ax in item:
                    if isinstance(item[ax], AbstractArray):
                        result = item[ax]._getitem_reversed(self, item)
                        if result is not NotImplemented:
                            return result

            elif isinstance(item, AbstractArray):
                result = item._getitem_reversed(self, item)
                if result is not NotImplemented:
                    return result

        raise ValueError(f"item not supported by array with type {type(self)}")

    @abc.abstractmethod
    def __bool__(self: Self) -> bool:
        return True

    @abc.abstractmethod
    def __mul__(self: Self, other: ArrayLike | u.Unit) -> AbstractExplicitArray:
        return super().__mul__(other)

    @abc.abstractmethod
    def __lshift__(self: Self, other: ArrayLike | u.UnitBase) -> AbstractExplicitArray:
        return super().__lshift__(other)

    @abc.abstractmethod
    def __truediv__(self: Self, other: ArrayLike | u.UnitBase) -> AbstractExplicitArray:
        return super().__truediv__(other)

    @abc.abstractmethod
    def __array_matmul__(
            self: Self,
            x1: ArrayLike,
            x2: ArrayLike,
            out: None | AbstractExplicitArray = None,
            **kwargs,
    ) -> AbstractExplicitArray:
        """
        Method to handle the behavior of :func:`numpy.matmul` which has different behavior than the other ufuncs.
        """
        return NotImplemented

    @abc.abstractmethod
    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | AbstractArray | tuple[AbstractArray, ...]:
        """
        Method to override the behavior of numpy's ufuncs.
        """
        if function is np.matmul:
            return self.__array_matmul__(*inputs, **kwargs)
        else:
            return NotImplemented

    @abc.abstractmethod
    def __array_function__(
            self: Self,
            func: Callable,
            types: Collection,
            args: tuple,
            kwargs: dict[str, Any],
    ):
        """
        Method to override the behavior of numpy's array functions.
        """
        from . import _core_array_functions

        if func in _core_array_functions.HANDLED_FUNCTIONS:
            return _core_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def __named_array_function__(self, func, *args, **kwargs):
        """
        Method used to dispatch custom named array functions
        """
        return NotImplemented

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
            initial: npt.ArrayLike = np._NoValue,
            where: Self = np._NoValue,
    ) -> Self:
        return np.min(self, axis=axis, initial=initial, where=where)

    def max(
            self: Self,
            axis: None | str | Sequence[str] = None,
            initial: npt.ArrayLike = np._NoValue,
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

    def percentile(
            self: Self,
            q: int | float | u.Quantity | Self,
            axis: None | str | Sequence[str] = None,
            out: None | Self = None,
            overwrite_input: bool = False,
            method: str = 'linear',
            keepdims: bool = False,
    ):
        return np.percentile(
            a=self,
            q=q,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=method,
            keepdims=keepdims,
        )

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

    def transpose(
            self: Self,
            axes: None | Sequence[str] = None,
    ) -> Self:
        return np.transpose(self, axes=axes)

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
            y0 = self[item_base_0].astype(float)
            y1 = self[item_base_1].astype(float)

        result = y0 + (x - x0) * (y1 - y0)
        return result

    def interp_linear(
            self: Self,
            item: dict[str, Self],
    ) -> Self:
        if item:
            return self._interp_linear_recursive(
                item=item,
                item_base=self[{ax: 0 for ax in item}].indices,
            )
        else:
            return self

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


@dataclasses.dataclass(eq=False, repr=False)
class AbstractExplicitArray(
    AbstractArray,
):
    @classmethod
    @abc.abstractmethod
    def from_scalar_array(
            cls: type[Self],
            a: float | u.Quantity | na.AbstractScalarArray,
            like: None | Self = None,
    ) -> Self:
        """
        Constructs a new version of this array using ``a`` as the underlying data.

        Parameters
        ----------
        a
            Anything that can be coerced into an instance of :class:`named_arrays.AbstractScalarArray`.
        like
            Optional reference object.
            If provided, the result will be defined by this object.
        """
        if like is None:
            return cls()
        else:
            if isinstance(like, cls):
                return type(like)()
            else:
                raise TypeError(
                    f"If `like` is not `None`, it must be an instance of `{cls.__name__}`, "
                    f"got `{type(like).__name__}`"
                )

    @abc.abstractmethod
    def __setitem__(
            self,
            item,
            value,
    ) -> None:
        pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitArray(
    AbstractArray,
):
    @property
    def axes(self: Self) -> tuple[str, ...]:
        return self.explicit.axes

    @property
    def ndim(self: Self) -> int:
        return self.explicit.ndim

    @property
    def size(self: Self) -> int:
        return self.explicit.size

    @property
    def value(self: Self) -> Self:
        return self.explicit.value

    @property
    def shape(self: Self) -> dict[str, int]:
        return self.explicit.shape

    @abc.abstractmethod
    def _attr_normalized(self, name: str) -> AbstractExplicitArray:
        """
        Similar to :func:`getattr`, but normalizes it to an instance of :class:`AbstractExplicitArray`
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRandomMixin(
    abc.ABC,
):

    def __post_init__(self):
        if self.seed is None:
            self.seed = secrets.randbits(128)

    @property
    @abc.abstractmethod
    def seed(self: Self) -> int:
        """
        Seed for the random number generator instance
        """

    @seed.setter
    @abc.abstractmethod
    def seed(self: Self, value: int) -> None:
        pass

    @property
    def _rng(self: Self) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRangeMixin(
    abc.ABC,
):

    @property
    @abc.abstractmethod
    def start(self: Self) -> int | AbstractArray:
        """
        Starting value of the range.
        """

    @property
    @abc.abstractmethod
    def stop(self: Self) -> int | AbstractArray:
        """
        Ending value of the range.
        """

    @property
    def range(self: Self) -> AbstractArray:
        return self.stop - self.start


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSymmetricRangeMixin(
    AbstractRangeMixin,
):
    @property
    @abc.abstractmethod
    def center(self: Self) -> ArrayLike:
        """
        Center value of the range.
        """

    @property
    @abc.abstractmethod
    def width(self: Self) -> ArrayLike:
        """
        Width of the range.
        """

    @property
    def start(self: Self) -> ArrayLike:
        return self.center - self.width

    @property
    def stop(self: Self) -> ArrayLike:
        return self.center + self.width


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRandomSample(
    AbstractRandomMixin,
    AbstractImplicitArray,
):

    @property
    @abc.abstractmethod
    def shape_random(self: Self) -> None | dict[str, int]:
        """
        Dimensions along which the resulting random sample is completely uncorrelated.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractUniformRandomSample(
    AbstractRangeMixin,
    AbstractRandomSample,
    Generic[StartT, StopT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def explicit(self) -> AbstractExplicitArray:

        start = self._attr_normalized("start")
        stop = self._attr_normalized("stop")

        return na.random.uniform(
            low=start,
            high=stop,
            shape_random=self.shape_random,
            seed=self.seed,
        )


@dataclasses.dataclass
class AbstractNormalRandomSample(
    AbstractSymmetricRangeMixin,
    AbstractRandomSample,
    Generic[CenterT, WidthT],
):
    center: CenterT = dataclasses.MISSING
    width: WidthT = dataclasses.MISSING
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def explicit(self) -> AbstractExplicitArray:
        center = self._attr_normalized("center")
        width = self._attr_normalized("width")

        return na.random.normal(
            loc=center,
            scale=width,
            shape_random=self.shape_random,
            seed=self.seed,
        )


@dataclasses.dataclass
class AbstractPoissonRandomSample(
    AbstractRandomSample,
    Generic[CenterT],
):
    center: CenterT = dataclasses.MISSING
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def explicit(self) -> AbstractExplicitArray:
        center = self._attr_normalized("center")

        return na.random.poisson(
            lam=center,
            shape_random=self.shape_random,
            seed=self.seed,
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedArray(
    AbstractImplicitArray,
):

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


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLinearParameterizedArrayMixin(
    abc.ABC
):
    @property
    @abc.abstractmethod
    def step(self: Self) -> int | AbstractArray:
        """
        Spacing between the values.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractArrayRange(
    AbstractLinearParameterizedArrayMixin,
    AbstractRangeMixin,
    AbstractParameterizedArray,
    Generic[StartT, StopT]
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    axis: str | na.AbstractArray= dataclasses.MISSING
    step: int | float | na.AbstractArray = 1

    @property
    def explicit(self: Self) -> AbstractExplicitArray:
        start = self._attr_normalized("start")
        stop = self._attr_normalized("stop")

        return na.arange(
            start=start,
            stop=stop,
            axis=self.axis,
            step=self.step,
        )

    @property
    def num(self: Self) -> int:
        return np.ceil((self.stop - self.start) / self.step).astype(int)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpace(
    AbstractParameterizedArray,
):
    @property
    @abc.abstractmethod
    def endpoint(self: Self) -> bool:
        """
        If ``True``, :attr:`stop` is the last sample, otherwise it is not included.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLinearSpace(
    AbstractLinearParameterizedArrayMixin,
    AbstractRangeMixin,
    AbstractSpace,
    Generic[StartT, StopT, AxisT, NumT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    axis: AxisT = dataclasses.MISSING
    num: NumT = 11
    endpoint: bool = True
    centers: bool = False

    @property
    def explicit(self: Self) -> AbstractExplicitArray:
        return na.linspace(
            start=self._attr_normalized("start"),
            stop=self._attr_normalized("stop"),
            axis=self.axis,
            num=self.num,
            endpoint=self.endpoint,
            centers=self.centers,
        )

    @property
    def step(self: Self) -> AbstractArray:
        return na.step(
            start=self.start,
            stop=self.stop,
            num=self.num,
            endpoint=self.endpoint,
            centers=self.centers,
        )


def strata(a: AbstractArray) -> AbstractArray:
    """
    If ``a`` is an instance of :class:`AbstractStratifiedRandomSpace`,
    return ``a.strata``, otherwise return ``a``

    Parameters
    ----------
    a
        An array to isolate the strata of.

    Examples
    --------

    Make a scatterplot of a 2D stratified random array and the centers of
    the strata.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define a 2D stratified random array.
        a = na.Cartesian2dVectorStratifiedRandomSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=11,
        )

        # Isolate the strata of the array
        strata = na.strata(a)

        # Plot the array and the strata
        fig, ax = plt.subplots()
        na.plt.scatter(
            a.x,
            a.y,
            ax=ax,
        )
        na.plt.scatter(
            strata.x,
            strata.y,
            ax=ax,
        );
    """
    if isinstance(a, AbstractStratifiedRandomSpace):
        return a.strata
    else:
        return a


@dataclasses.dataclass(eq=False, repr=False)
class AbstractStratifiedRandomSpace(
    AbstractRandomMixin,
    AbstractLinearSpace[StartT, StopT, AxisT, NumT],
):
    centers: bool = True
    seed: None | int = None

    @property
    def explicit(self: Self) -> AbstractExplicitArray:
        result = self.strata

        step_size = self.step

        delta = na.random.uniform(
            low=-step_size / 2,
            high=step_size / 2,
            shape_random=result.shape,
            seed=self.seed,
        )

        return result + delta

    @property
    def strata(self: Self) -> AbstractExplicitArray:
        return na.linspace(
            start=self._attr_normalized("start"),
            stop=self._attr_normalized("stop"),
            num=self.num,
            endpoint=self.endpoint,
            axis=self.axis,
            centers=self.centers,
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLogarithmicSpace(
    AbstractRangeMixin,
    AbstractSpace,
    Generic[StartExponentT, StopExponentT, BaseT, AxisT, NumT]
):
    start_exponent: StartExponentT = dataclasses.MISSING
    stop_exponent: StopExponentT = dataclasses.MISSING
    base: BaseT = dataclasses.MISSING
    axis: AxisT = dataclasses.MISSING
    num: NumT = 11
    endpoint: bool = True

    @property
    def explicit(self: Self) -> AbstractExplicitArray:
        return na.logspace(
            start=self._attr_normalized("start_exponent"),
            stop=self._attr_normalized("stop_exponent"),
            axis=self.axis,
            num=self.num,
            endpoint=self.endpoint,
            base=self.base,
        )

    @property
    def start(self: Self) -> ArrayLike:
        return self.base ** self.start_exponent

    @property
    def stop(self: Self) -> ArrayLike:
        return self.base ** self.stop_exponent


@dataclasses.dataclass(eq=False, repr=False)
class AbstractGeometricSpace(
    AbstractRangeMixin,
    AbstractSpace,
    Generic[StartT, StopT, AxisT, NumT]
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    axis: AxisT = dataclasses.MISSING
    num: NumT = 11
    endpoint: bool = True

    @property
    def explicit(self: Self) -> AbstractExplicitArray:
        return na.geomspace(
            start=self._attr_normalized("start"),
            stop=self._attr_normalized("stop"),
            axis=self.axis,
            num=self.num,
            endpoint=self.endpoint,
        )
