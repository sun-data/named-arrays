from __future__ import annotations
from typing import TypeVar, Generic, ClassVar, Type, Sequence, Callable, Collection, Any, Union, cast, Dict
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import astropy.units as u
import named_arrays as na

__all__ = [
    'ScalarStartT',
    'ScalarStopT',
    'ScalarTypeError',
    'as_named_array',
    'AbstractScalar',
    'AbstractScalarArray',
    'ScalarLike',
    'ScalarArray',
    'AbstractImplicitScalarArray',
    'ScalarUniformRandomSample',
    'ScalarNormalRandomSample',
    'ScalarPoissonRandomSample',
    'AbstractParameterizedScalarArray',
    'ScalarArrayRange',
    'AbstractScalarSpace',
    'ScalarLinearSpace',
    'ScalarStratifiedRandomSpace',
    'ScalarLogarithmicSpace',
    'ScalarGeometricSpace',
]

NDArrayT = TypeVar('NDArrayT', bound=npt.ArrayLike)
StartT = TypeVar('StartT', bound='ScalarLike')
StopT = TypeVar('StopT', bound='ScalarLike')
ScalarStartT = TypeVar('ScalarStartT', bound='ScalarLike')
ScalarStopT = TypeVar('ScalarStopT', bound='ScalarLike')
CenterT = TypeVar('CenterT', bound='ScalarLike')
WidthT = TypeVar('WidthT', bound='ScalarLike')
StartExponentT = TypeVar('StartExponentT', bound='ScalarLike')
StopExponentT = TypeVar('StopExponentT', bound='ScalarLike')
BaseT = TypeVar('BaseT', bound='ScalarLike')


class ScalarTypeError(TypeError):
    pass


def _normalize(a: float | u.Quantity | na.AbstractScalarArray) -> na.AbstractScalarArray:

    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalarArray):
            result = a
        else:
            raise ScalarTypeError
    else:
        result = na.ScalarArray(a)

    return result


def as_named_array(value: bool | int | float | complex | str | u.Quantity | na.AbstractArray):
    if not hasattr(value, "__named_array_function__"):
        return ScalarArray(value)
    else:
        return value


@dataclasses.dataclass(eq=False, repr=False)
class AbstractScalar(
    na.AbstractArray,
):

    @property
    @abc.abstractmethod
    def dtype(self: Self) -> np.dtype:
        """
        Data type of the array. Equivalent to :attr:`numpy.ndarray.dtype`
        """

    @property
    def unit(self: Self) -> None | u.UnitBase:
        """
        Unit associated with the array.

        If :attr:`ndarray` is an instance of :class:`astropy.units.Quantity`, return :attr:`astropy.units.Quantity.unit`,
        otherwise return :class:`None`.
        """
        return na.unit(self)

    @property
    def unit_normalized(self: Self) -> u.UnitBase:
        """
        Similar to :attr:`unit` but returns :attr:`astropy.units.dimensionless_unscaled` if :attr:`ndarray` is not an
        instance of :class:`astropy.units.Quantity`.
        """
        return na.unit_normalized(self)

    @property
    def length(self) -> AbstractScalar:
        if np.issubdtype(self.dtype, np.number):
            return np.abs(self)
        else:
            raise ValueError('Can only compute length of numeric arrays')

    def volume_cell(self, axis: None | str | tuple[str]) -> na.AbstractScalar:
        if axis is None:
            if self.ndim != 1:
                raise ValueError(
                    f"If {axis=}, then {self.ndim=} must be one dimensional"
                )
            axis = self.axes[0]
        elif not isinstance(axis, str):
            axis, = axis

        return np.diff(self, axis=axis)

    def __array_matmul__(
            self: Self,
            x1: na.ArrayLike,
            x2: na.ArrayLike,
            out: None | AbstractScalar = None,
            **kwargs,
    ) -> na.AbstractExplicitArray:
        result = super().__array_matmul__(x1=x1, x2=x2, out=out, **kwargs)
        if result is not NotImplemented:
            return result

        return np.multiply(
            x1,
            x2,
            out=out,
            **kwargs,
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractScalarArray(
    AbstractScalar,
    Generic[NDArrayT],
):

    __named_array_priority__: ClassVar[int] = 1

    @property
    def type_abstract(self: Self) -> Type[AbstractScalarArray]:
        return AbstractScalarArray

    @property
    def type_explicit(self: Self) -> Type[ScalarArray]:
        return ScalarArray

    @property
    @abc.abstractmethod
    def ndarray(self: Self) -> bool | int | float | complex | str | np.ndarray | u.Quantity:
        """
        Underlying data that is wrapped by this class.

        This is usually an instance of :class:`numpy.ndarray` or :class:`astropy.units.Quantity`, but it can also be a
        built-in python type such as a :class:`int`, :class:`float`, or :class:`bool`
        """

    @property
    def value(self: Self) -> ScalarArray:
        return self.type_explicit(
            ndarray=na.value(self.ndarray),
            axes=self.axes,
        )

    def astype(
            self: Self,
            dtype: str | np.dtype | Type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> ScalarArray:
        return ScalarArray(
            ndarray=np.asanyarray(self.ndarray).astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            ),
            axes=self.axes,
        )

    def to(
        self: Self,
        unit: u.Unit,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = [],
        copy: bool = True,
    ) -> ScalarArray:
        ndarray = self.ndarray
        if not isinstance(ndarray, u.Quantity):
            ndarray = ndarray << u.dimensionless_unscaled
        return ScalarArray(
            ndarray=ndarray.to(
                unit=unit,
                equivalencies=equivalencies,
                copy=copy,
            ),
            axes=self.axes,
        )

    def ndarray_aligned(self: Self, axes: Sequence[str]) -> np.ndarray:
        """
        Align :attr:`ndarray` to a particular sequence of axes.

        Parameters
        ----------
        axes
            New sequence of axes to align :attr:`ndarray` to.

        Returns
        -------
        An instance of :class:`numpy.ndarray` with the axes aligned.
        """
        axes = tuple(axes)
        axes_self = self.axes

        ndarray = np.asanyarray(self.ndarray)

        if axes == axes_self:
            return ndarray

        ndim_missing = len(axes) - ndarray.ndim
        value = ndarray[(...,) + ndim_missing * (np.newaxis,)]
        source = []
        destination = []
        for axis_index, axis_name in enumerate(axes_self):
            source.append(axis_index)
            if axis_name not in axes:
                raise ValueError(
                    f"`axes` is missing axes present in the input array. "
                    f"`axes` is {axes} but `self.axes` is {self.axes}"
                )
            destination.append(axes.index(axis_name))
        value = np.moveaxis(value, source=source, destination=destination)
        return value

    def add_axes(self: Self, axes: str | Sequence[str]) -> ScalarArray:
        if isinstance(axes, str):
            axes = [axes]
        shape_new = {axis: 1 for axis in axes}
        shape = shape_new | self.shape
        return ScalarArray(
            ndarray=self.ndarray_aligned(shape),
            axes=tuple(shape.keys()),
        )

    def change_axis_index(self: Self, axis: str, index: int) -> ScalarArray:
        shape = self.shape
        size_axis = shape.pop(axis)
        keys = list(shape.keys())
        values = list(shape.values())
        index = index % len(self.shape) + 1
        keys.insert(index, axis)
        values.insert(index, size_axis)
        shape_new = {k: v for k, v in zip(keys, values)}
        return ScalarArray(
            ndarray=self.ndarray_aligned(shape_new),
            axes=tuple(keys),
        )

    def combine_axes(
            self: Self,
            axes: None | Sequence[str] = None,
            axis_new: None | str = None,
    ) -> ScalarArray:

        if axes is None:
            axes = self.axes

        if axis_new is None:
            axis_new = na.flatten_axes(axes)

        axes_new = list(self.axes)
        shape_new = self.shape
        for axis in axes:
            axes_new.append(axes_new.pop(axes_new.index(axis)))
            shape_new[axis] = shape_new.pop(axis)

        source = []
        destination = []
        for axis in axes:
            source.append(self.axes.index(axis))
            destination.append(axes_new.index(axis))

        for axis in axes:
            axes_new.remove(axis)
            shape_new.pop(axis)
        axes_new.append(axis_new)
        shape_new[axis_new] = -1

        return ScalarArray(
            ndarray=np.moveaxis(self.ndarray, source=source, destination=destination).reshape(tuple(shape_new.values())),
            axes=tuple(axes_new),
        )

    def _getitem(
            self: Self,
            item: dict[str, int | slice | AbstractScalarArray] | AbstractScalarArray,
    ):

        if isinstance(item, AbstractScalarArray):

            if not set(item.shape).issubset(self.axes):
                raise ValueError(
                    f"the axes in item, {item.axes}, must be a subset of the axes in the array, {self.axes}")

            value = np.moveaxis(
                a=self.ndarray,
                source=[self.axes.index(axis) for axis in item.axes],
                destination=np.arange(len(item.axes)),
            )

            if item.shape:
                axis_new = item.axes_flattened
            else:
                axis_new = "boolean"

            return ScalarArray(
                ndarray=np.moveaxis(value[item.ndarray], 0, ~0),
                axes=tuple(axis for axis in self.axes if axis not in item.axes) + (axis_new, )
            )

        elif isinstance(item, dict):
            axes = self.axes

            item_advanced = dict()      # type: typ.Dict[str, AbstractScalarArray]
            for axis in item:
                item_axis = item[axis]
                if isinstance(item_axis, na.AbstractArray):
                    if isinstance(item_axis, AbstractScalarArray):
                        item_advanced[axis] = item_axis
                    else:
                        return NotImplemented

            if not set(ax for ax in item if item[ax] is not None).issubset(axes):
                raise ValueError(f"the axes in item, {tuple(item)}, must be a subset of the axes in the array, {axes}")

            shape_advanced = na.shape_broadcasted(*item_advanced.values())

            ndarray_organized = np.moveaxis(
                self.ndarray,
                source=tuple(axes.index(ax) for ax in item_advanced),
                destination=tuple(range(len(item_advanced))),
            )

            axes_organized = list(item_advanced.keys()) + list(ax for ax in axes if ax not in item_advanced)

            axes_new = axes_organized.copy()
            index = [slice(None)] * self.ndim   # type: list[int | slice | AbstractScalar]
            for ax in item:
                item_axis = item[ax]
                if item_axis is None:
                    continue
                if ax in item_advanced:
                    item_axis = item_axis.ndarray_aligned(shape_advanced)
                index[axes_organized.index(ax)] = item_axis
                if not isinstance(item_axis, slice):
                    axes_new.remove(ax)

            if any(ax in shape_advanced for ax in axes_new):
                raise ValueError(
                    f"axis in advanced axes, {tuple(shape_advanced)}, "
                    f"is already in basic axes, {tuple(axes_new)}"
                )

            return ScalarArray(
                ndarray=ndarray_organized[tuple(index)],
                axes=tuple(shape_advanced.keys()) + tuple(axes_new),
            )

        else:
            return NotImplemented

    def _getitem_reversed(
            self: Self,
            array: AbstractScalarArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        return NotImplemented

    def __bool__(self: Self) -> bool:
        return super().__bool__() and self.ndarray.__bool__()

    def __mul__(self: Self, other: na.ArrayLike | u.Unit) -> ScalarArray:
        if isinstance(other, u.UnitBase):
            return ScalarArray(
                ndarray=self.ndarray * other,
                axes=self.axes,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self: Self, other: na.ArrayLike | u.UnitBase) -> ScalarArray:
        if isinstance(other, u.UnitBase):
            return ScalarArray(
                ndarray=self.ndarray << other,
                axes=self.axes
            )
        else:
            return super().__lshift__(other)

    def __truediv__(self: Self, other: na.ArrayLike | u.UnitBase) -> ScalarArray:
        if isinstance(other, u.UnitBase):
            return ScalarArray(
                ndarray=self.ndarray / other,
                axes=self.axes
            )
        else:
            return super().__truediv__(other)

    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | ScalarArray | tuple[ScalarArray, ...]:

        result = super().__array_ufunc__(
            function,
            method,
            *inputs,
            **kwargs,
        )
        if result is not NotImplemented:
            return result

        if function is np.matmul:
            return NotImplemented

        nout = function.nout

        kwargs_ndarray = kwargs.copy()

        if "out" in kwargs_ndarray:
            out = kwargs_ndarray["out"]
            out_normalized = list()
            for o in out:
                if o is not None:
                    if isinstance(o, ScalarArray):
                        if isinstance(o.ndarray, np.ndarray):
                            o = o.ndarray
                        else:
                            o = None
                    else:
                        return NotImplemented
                out_normalized.append(o)
            kwargs_ndarray["out"] = tuple(out_normalized)
        else:
            out = (None, ) * nout

        out_arrays = tuple(o for o in out if o is not None)
        if out_arrays:
            shape = out_arrays[0].shape
            if any(o.shape != shape for o in out_arrays[1:]):
                raise ValueError(
                    f"all the `out` arrays should have the same shape, got {tuple(o.shape for o in out_arrays)}"
                )
        else:
            shape = na.shape_broadcasted(*inputs)

        if 'where' in kwargs_ndarray:
            if isinstance(kwargs_ndarray['where'], na.AbstractArray):
                if isinstance(kwargs_ndarray['where'], na.AbstractScalarArray):
                    kwargs_ndarray['where'] = kwargs_ndarray['where'].ndarray_aligned(shape)
                else:
                    return NotImplemented

        inputs_ndarray = []
        for inp in inputs:
            if isinstance(inp, na.AbstractArray):
                if isinstance(inp, AbstractScalarArray):
                    inp = inp.ndarray_aligned(shape)
                else:
                    return NotImplemented
            elif inp is None:
                return None
            inputs_ndarray.append(inp)

        result_ndarray = getattr(function, method)(*inputs_ndarray, **kwargs_ndarray)
        if nout == 1:
            result_ndarray = (result_ndarray, )
        result = list(ScalarArray(result_ndarray[i], axes=tuple(shape.keys())) for i in range(nout))

        for i in range(nout):
            if out[i] is not None:
                out[i].ndarray = result[i].ndarray
                result[i] = out[i]

        if nout == 1:
            result = result[0]
        else:
            result = tuple(result)
        return result


    def __array_function__(
            self: Self,
            func: Callable,
            types: Collection,
            args: tuple,
            kwargs: dict[str, Any],
    ):
        result = super().__array_function__(func=func, types=types, args=args, kwargs=kwargs)
        if result is not NotImplemented:
            return result

        from . import scalar_array_functions

        if func in scalar_array_functions.SINGLE_ARG_FUNCTIONS:
            return scalar_array_functions.array_function_single_arg(func, *args, **kwargs)

        if func in scalar_array_functions.ARRAY_CREATION_LIKE_FUNCTIONS:
            return scalar_array_functions.array_function_array_creation_like(func, *args, **kwargs)

        if func in scalar_array_functions.SEQUENCE_FUNCTIONS:
            return scalar_array_functions.array_function_sequence(func, *args, **kwargs)

        if func in scalar_array_functions.DEFAULT_FUNCTIONS:
            return scalar_array_functions.array_function_default(func, *args, **kwargs)

        if func in scalar_array_functions.PERCENTILE_LIKE_FUNCTIONS:
            return scalar_array_functions.array_function_percentile_like(func, *args, **kwargs)

        if func in scalar_array_functions.ARG_REDUCE_FUNCTIONS:
            return scalar_array_functions.array_function_arg_reduce(func, *args, **kwargs)

        if func in scalar_array_functions.FFT_LIKE_FUNCTIONS:
            return scalar_array_functions.array_function_fft_like(func, *args, **kwargs)

        if func in scalar_array_functions.FFTN_LIKE_FUNCTIONS:
            return scalar_array_functions.array_function_fftn_like(func, *args, **kwargs)

        if func in scalar_array_functions.EMATH_FUNCTIONS:
            return scalar_array_functions.array_function_emath(func, *args, **kwargs)

        if func in scalar_array_functions.HANDLED_FUNCTIONS:
            return scalar_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def __named_array_function__(self, func, *args, **kwargs):
        result = super().__named_array_function__(func, *args, **kwargs)
        if result is not NotImplemented:
            return result

        from . import scalar_named_array_functions

        if func in scalar_named_array_functions.ASARRAY_LIKE_FUNCTIONS:
            return scalar_named_array_functions.asarray_like(func=func, *args, **kwargs)

        if func in scalar_named_array_functions.RANDOM_FUNCTIONS:
            return scalar_named_array_functions.random(func=func, *args, **kwargs)

        if func in scalar_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS:
            return scalar_named_array_functions.plt_plot_like(func, *args, **kwargs)

        if func in scalar_named_array_functions.HANDLED_FUNCTIONS:
            return scalar_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def matrix_multiply(
            self: Self,
            other: AbstractScalar,
            axis_rows: str,
            axis_columns: str,
    ) -> ScalarArray:

        shape = self.shape_broadcasted(other)
        shape_rows = shape.pop(axis_rows)
        shape_columns = shape.pop(axis_columns)
        shape = {**shape, axis_rows: shape_rows, axis_columns: shape_columns}

        data_self = self.ndarray_aligned(shape)
        data_other = other.ndarray_aligned(shape)

        return ScalarArray(
            ndarray=np.matmul(data_self, data_other),
            axes=list(shape.keys()),
        )

    def matrix_determinant(
            self: Self,
            axis_rows: str,
            axis_columns: str
    ) -> ScalarArray:
        shape = self.shape
        if shape[axis_rows] != shape[axis_columns]:
            raise ValueError('Matrix must be square')

        if shape[axis_rows] == 2:
            a = self[{axis_rows: 0, axis_columns: 0}]
            b = self[{axis_rows: 0, axis_columns: 1}]
            c = self[{axis_rows: 1, axis_columns: 0}]
            d = self[{axis_rows: 1, axis_columns: 1}]
            return a * d - b * c

        elif shape[axis_rows] == 3:
            a = self[{axis_rows: 0, axis_columns: 0}]
            b = self[{axis_rows: 0, axis_columns: 1}]
            c = self[{axis_rows: 0, axis_columns: 2}]
            d = self[{axis_rows: 1, axis_columns: 0}]
            e = self[{axis_rows: 1, axis_columns: 1}]
            f = self[{axis_rows: 1, axis_columns: 2}]
            g = self[{axis_rows: 2, axis_columns: 0}]
            h = self[{axis_rows: 2, axis_columns: 1}]
            i = self[{axis_rows: 2, axis_columns: 2}]
            return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h)

        else:
            value = np.moveaxis(
                a=self.ndarray,
                source=[self.axis_names.index(axis_rows), self.axis_names.index(axis_columns)],
                destination=[~1, ~0],
            )

            axes_new = list(self.axes)
            axes_new.remove(axis_rows)
            axes_new.remove(axis_columns)

            return ScalarArray(
                ndarray=np.linalg.det(value),
                axes=tuple(axes_new),
            )

    def matrix_inverse(
            self: Self,
            axis_rows: str,
            axis_columns: str,
    ) -> ScalarArray:

        axis_rows_inverse = axis_columns
        axis_columns_inverse = axis_rows

        index_axis_rows = self.axes.index(axis_rows)
        index_axis_columns = self.axes.index(axis_columns)
        value = np.moveaxis(
            a=self.ndarray,
            source=[index_axis_rows, index_axis_columns],
            destination=[~1, ~0],
        )

        axes_new = list(self.axes)
        axes_new.remove(axis_rows)
        axes_new.remove(axis_columns)
        axes_new.append(axis_rows_inverse)
        axes_new.append(axis_columns_inverse)

        return ScalarArray(
            ndarray=np.linalg.inv(value),
            axes=tuple(axes_new),
        )

    def filter_median(
            self: Self,
            shape_kernel: dict[str, int],
            mode: str = 'reflect',
    ):

        shape = self.shape
        shape_kernel_final = {axis: shape_kernel[axis] if axis in shape_kernel else 1 for axis in shape}

        inp = self.ndarray
        if isinstance(inp, u.Quantity):
            inp = inp.value
        result = ScalarArray(
            ndarray=scipy.ndimage.median_filter(
                input=inp,
                size=tuple(shape_kernel_final.values()),
                mode=mode,
            ),
            axes=self.axes,
        )
        if isinstance(self.ndarray, u.Quantity):
            result = result << self.unit

        return result


ScalarLike = Union[na.QuantityLike, AbstractScalar]


@dataclasses.dataclass(eq=False, repr=False)
class ScalarArray(
    AbstractScalarArray,
    na.AbstractExplicitArray,
    Generic[NDArrayT],
):
    """
    An array representing a scalar quantity (like pressure or temperature) with names for each of its `N` axes.

    A :class:`ScalarArray` is defined by a :class:`numpy.ndarray` and a :class:`tuple` of axis names.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na

        x = np.array([1, 2, 3])
        x = na.ScalarArray(x, axes=('position_x', ))
        print(x)

    They can also be defined using a :class:`astropy.units.Quantity`, or ascribed units after creation.

    .. jupyter-execute::

        import astropy.units as u

        x = x * u.cm

        print(x)

    What happens when we do math with instances of :class:`ScalarArray`?

    .. jupyter-execute::

        y = np.array([4, 5, 6]) * u.mm

        y = na.ScalarArray(y, axes=('position_y', ))
        print(y**2)

        radius = np.sqrt(x**2 + y**2)
        print(radius)
        print(radius.shape)

    Note how when performing mathematical operations on two instances of :class:`ScalarArray` with different axes, the
    arrays are automatically broadcast over every axis. There is no need to insert extra dimensions for alignment like
    you would normally do with instances of :class:`numpy.ndarray`.

    We can also use reduction operators (mean, sum, etc) on instances of :class:`ScalarArray` by using the axis name
    instead of the axis index.

    .. jupyter-execute::

        print(radius.mean())
        print(radius.mean(axis='position_x'))
    """

    ndarray: None | NDArrayT = 0
    axes: None | str | tuple[str, ...] = None

    def __post_init__(self: Self):
        if self.axes is None:
            self.axes = tuple()
        if isinstance(self.axes, str):
            self.axes = (self.axes, )
        if getattr(self.ndarray, 'ndim', 0) != len(self.axes):
            raise ValueError('The number of axis names must match the number of dimensions.')
        if len(self.axes) != len(set(self.axes)):
            raise ValueError(f'Each axis name must be unique, got {self.axes}.')

    @classmethod
    def from_scalar_array(
            cls: type[Self],
            a: float | u.Quantity | na.AbstractScalarArray,
            like: None | Self = None,
    ) -> Self:

        self = super().from_scalar_array(a=a, like=like)

        if isinstance(a, na.AbstractArray):
            if isinstance(a, na.AbstractScalarArray):
                self.ndarray = a.ndarray
                self.axes = a.axes
            else:
                raise TypeError(
                    f"If `a` is an instance of `{na.AbstractArray.__name__}`, it must be an instance of "
                    f"`{na.AbstractScalarArray.__name__}`, got `{type(a).__name__}`."
                )
        else:
            self.ndarray = a
            self.axes = tuple()

        return self

    @classmethod
    def empty(cls: Type[Self], shape: dict[str, int], dtype: Type | np.dtype = float) -> Self:
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
        return cls(
            ndarray=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axes=tuple(shape.keys()),
        )

    @classmethod
    def zeros(cls: Type[Self], shape: dict[str, int], dtype: Type | np.dtype = float) -> Self:
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
        return cls(
            ndarray=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axes=tuple(shape.keys()),
        )

    @classmethod
    def ones(cls: Type[Self], shape: dict[str, int], dtype: Type | np.dtype = float) -> Self:
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
        return cls(
            ndarray=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axes=tuple(shape.keys()),
        )

    @property
    def shape(self: Self) -> dict[str, int]:
        try:
            return dict(zip(self.axes, self.ndarray.shape, strict=True))
        except AttributeError:
            return dict()

    @property
    def ndim(self: Self) -> int:
        return np.ndim(self.ndarray)

    @property
    def size(self: Self) -> int:
        return np.size(self.ndarray)

    @property
    def dtype(self: Self) -> np.dtype:
        return na.get_dtype(self.ndarray)

    @property
    def explicit(self: Self) -> Self:
        return self

    def __setitem__(
            self: Self,
            item: dict[str, int | slice | AbstractScalarArray] | AbstractScalarArray,
            value: int | float | u.Quantity | AbstractScalarArray,
    ) -> None:

        shape_self = self.shape

        if isinstance(value, na.AbstractArray):
            if isinstance(value, na.AbstractScalarArray):
                value = value.explicit
            else:
                raise TypeError(
                    f"if `value` is an instance of `AbstractArray`, it must be an instance of `AbstractScalarArray`, "
                    f"got {type(value)}"
                )
        else:
            value = ScalarArray(value)

        if isinstance(item, AbstractScalarArray):

            item = item.explicit
            shape_item = item.shape

            if not set(item.shape).issubset(shape_self):
                raise ValueError(
                    f"if `item` is an instance of `{na.AbstractArray.__name__}`, "
                    f"`item.axes`, {item.axes}, should be a subset of `self.axes`, {self.axes}"
                )

            if shape_item:
                axis_new = item.axes_flattened
            else:
                axis_new = "boolean"

            axes_untouched = tuple(ax for ax in shape_self if ax not in shape_item)
            axes_value = (axis_new, ) + axes_untouched
            axes_self = tuple(shape_item) + axes_untouched

            self.ndarray_aligned(axes_self)[item.ndarray] = value.ndarray_aligned(axes_value)

        elif isinstance(item, dict):

            if not set(item).issubset(shape_self):
                raise ValueError(
                    f"if `item` is a `{dict.__name__}`, the keys in `item`, {tuple(item)}, "
                    f"must be a subset of `self.axes`, {self.axes}"
                )

            item_advanced = {ax: item[ax] for ax in item if na.shape(item[ax])}

            shape_advanced = na.shape_broadcasted(*item_advanced.values())

            axes_self = tuple(shape_advanced) + tuple(ax for ax in shape_self if ax not in shape_advanced)
            axes_value = list(axes_self)

            index = [slice(None)] * self.ndim   # type: list[Union[int, slice, AbstractScalar]]
            for axis in item:
                item_axis = item[axis]
                if isinstance(item_axis, na.AbstractScalarArray):
                    item_axis = item_axis.ndarray_aligned(shape_advanced)
                elif isinstance(item_axis, slice):
                    pass
                elif isinstance(item_axis, int):
                    if axis in value.shape:
                        raise ValueError(f"`value` has an axis, '{axis}', that is set to an `int` in `item`")
                    axes_value.remove(axis)
                else:
                    raise TypeError(
                        f"if `item` is a `{dict}`, all its values must be an instance of an `{int}`, a `{slice}`,"
                        f"or an {na.AbstractScalarArray.__name__}, got {type(item_axis).__name__} for key '{axis}'"
                    )
                index[axes_self.index(axis)] = item_axis

            if value.shape:
                value = value.ndarray_aligned(axes_value)
            else:
                value = value.ndarray

            self.ndarray_aligned(axes_self)[tuple(index)] = value


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitScalarArray(
    AbstractScalarArray,
    na.AbstractImplicitArray,
):

    @property
    def ndarray(self: Self) -> na.QuantityLike:
        return self.explicit.ndarray

    @property
    def dtype(self: Self) -> np.dtype:
        return self.explicit.dtype

    def _attr_normalized(self, name: str) -> ScalarArray:

        attr = getattr(self, name)

        if isinstance(attr, na.AbstractArray):
            if isinstance(attr, na.AbstractScalarArray):
                result = attr
            else:
                raise TypeError(
                    f"if `{name}` is an instance of `AbstractArray`, it must be an instance of `AbstractScalarArray`, "
                    f"got {type(attr)}"
                )
        else:
            result = ScalarArray(attr)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractScalarRandomSample(
    AbstractImplicitScalarArray,
    na.AbstractRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ScalarUniformRandomSample(
    AbstractScalarRandomSample,
    na.AbstractUniformRandomSample[ScalarStartT, ScalarStopT],
):
    def volume_cell(self, axis: None | str | tuple[str]) -> na.AbstractScalar:
        axis = na.axis_normalized(self, axis)
        if len(axis) != 1:
            raise ValueError(
                f"{axis=} must have exactly one element for scalars."
            )
        axis, = axis

        shape_random = self.shape_random
        if axis in shape_random:
            result = (self.stop - self.start) / shape_random[axis]
        else:
            result = super().volume_cell(axis)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class ScalarNormalRandomSample(
    AbstractScalarRandomSample,
    na.AbstractNormalRandomSample[CenterT, WidthT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ScalarPoissonRandomSample(
    AbstractScalarRandomSample,
    na.AbstractPoissonRandomSample[CenterT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedScalarArray(
    AbstractImplicitScalarArray,
    na.AbstractParameterizedArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ScalarArrayRange(
    AbstractParameterizedScalarArray,
    na.AbstractArrayRange[ScalarStartT, ScalarStopT],
):
    """
    An :class:`AbstractScalarArray` over the range [:attr:`start`, :attr:`stop`) incremented by :attr:`step`.
    An analog to :func:`numpy.arange`.

    :class:`ScalarArrayRange` can be used to create a :class:`ScalarArray` of integers.

    .. jupyter-execute::

        import named_arrays as na
        x = na.ScalarArrayRange(1, 8, axis = "x")
        print(x.explicit)
        print(x.shape)

    Note above that ``x`` does not include :attr:`stop`, and won't in almost all cases.  :class:`ScalarArrayRange` can be used to
    create an increasing :class:`ScalarArray` of floats, even with non integer steps.

    .. jupyter-execute::

        x = na.ScalarArrayRange(-0.5, 3, "x", 0.25)
        print(x.explicit)

    For the above, and more complicated uses, it is recommended to use :class:`ScalarLinearSpace` instead.  See numpy
    documentation of :func:`numpy.arange` for more info.
    """
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractScalarSpace(
    AbstractParameterizedScalarArray,
    na.AbstractSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ScalarLinearSpace(
    AbstractScalarSpace,
    na.AbstractLinearSpace[StartT, StopT, str, int],
):
    """
    An evenly spaced :class:`ScalarArray` ranging from start to stop with num elements.

    Most often, instances of :class:`ScalarArray` won't be formed directly from a :class:`numpy.ndarray`, but through
    more useful routines like :class:`ScalarLinearSpace`, a named_arrays equivalent to :func:`numpy.linspace`.
    For example, one can quickly create an evenly spaced coordinate (or axis) array with units.

    .. jupyter-execute::

        import named_arrays as na
        import astropy.units as u

        photon_energy = na.ScalarLinearSpace(1, 10, axis="energy", num=10) * u.keV
        print(photon_energy.shape)
        print(photon_energy)

    Then easily convert that into a wavelength.

    .. jupyter-execute::

        wavelength = (1240 * u.eV * u.nm / photon_energy).to(u.nm)
        print(wavelength)
    """

    def volume_cell(self, axis: None | str | tuple[str]) -> na.AbstractScalar:
        axis = na.axis_normalized(self, axis)
        if len(axis) != 1:
            raise ValueError(
                f"{axis=} must have exactly one element for scalars."
            )
        axis, = axis

        if axis == self.axis:
            result = self.step

        else:
            result = super().volume_cell(axis)

        return result

    # def index(
    #         self: ScalarLinearSpaceT,
    #         value: ScalarLinearSpaceT,
    #         axis: None | str | Sequence[str] = None,
    # ) -> dict[str, ScalarArrayT]:
    #     result = super().index(
    #         value=value,
    #         axis=[a for a in axis if a != self.axis],
    #     )
    #     result[self.axis] = (value - self.start) / self.step
    #
    #     return result
    #
    # def index_nearest(self, value: AbstractScalarT) -> typ.Dict[str, AbstractScalarT]:
    #     return {self.axis: np.rint((value - self.start) / self.step).astype(int)}
    #
    # def index_below(self, value: AbstractScalarT) -> typ.Dict[str, AbstractScalarT]:
    #     return {self.axis: (value - self.start) // self.step}


@dataclasses.dataclass(eq=False, repr=False)
class ScalarStratifiedRandomSpace(
    ScalarLinearSpace[StartT, StopT],
    na.AbstractStratifiedRandomSpace[StartT, StopT, str, int],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ScalarLogarithmicSpace(
    AbstractScalarSpace,
    na.AbstractLogarithmicSpace[StartExponentT, StopExponentT, BaseT, str, int],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class ScalarGeometricSpace(
    AbstractScalarSpace,
    na.AbstractGeometricSpace[StartT, StopT, str, int],
):
    pass
