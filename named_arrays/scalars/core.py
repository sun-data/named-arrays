from __future__ import annotations
from typing import TypeVar, Generic, ClassVar, Type, Sequence, Callable, Collection, Any, Union, cast, Dict
from typing_extensions import Self

import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import astropy.units as u

import named_arrays.core as na

__all__ = [
    'as_named_array',
    'AbstractScalar',
    'AbstractScalarArray',
    'ScalarLike',
    'ScalarArray',
    'AbstractImplicitScalarArray',
    'ScalarUniformRandomSample',
    'ScalarNormalRandomSample',
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
CenterT = TypeVar('CenterT', bound='ScalarLike')
WidthT = TypeVar('WidthT', bound='ScalarLike')
StartExponentT = TypeVar('StartExponentT', bound='ScalarLike')
StopExponentT = TypeVar('StopExponentT', bound='ScalarLike')
BaseT = TypeVar('BaseT', bound='ScalarLike')


def as_named_array(value: bool | int | float | complex | str | u.Quantity | na.AbstractArray):
    if not isinstance(value, na.AbstractArray):
        return ScalarArray(value)
    else:
        return value


@dataclasses.dataclass(eq=False)
class AbstractScalar(
    na.AbstractArray,
):
    @property
    def scalar(self: Self) -> Self:
        return self

    @property
    def length(self) -> AbstractScalar:
        if np.issubdtype(self.dtype, np.number):
            return np.abs(self)
        else:
            raise ValueError('Can only compute length of numeric arrays')


@dataclasses.dataclass(eq=False)
class AbstractScalarArray(
    AbstractScalar,
    Generic[NDArrayT],
):

    type_ndarray_primary: ClassVar[Type] = np.ndarray
    type_ndarray_auxiliary: ClassVar[tuple[Type, ...]] = (str, bool, int, float, complex, np.generic)
    type_ndarray: ClassVar[tuple[Type, ...]] = type_ndarray_auxiliary + (type_ndarray_primary, )

    __named_array_priority__: ClassVar[int] = 1

    @property
    def type_array(self: Self) -> Type[ScalarArray]:
        return ScalarArray

    @property
    def type_array_abstract(self: Self) -> Type[AbstractScalarArray]:
        return AbstractScalarArray

    @property
    def nominal(self: Self) -> Self:
        return self

    @property
    def distribution(self: Self) -> None:
        return None

    def astype(
            self: Self,
            dtype: str | np.dtype | Type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> ScalarArray:
        return ScalarArray(
            ndarray=self.ndarray_normalized.astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            ),
            axes=self.axes,
        )

    def to(self: Self, unit: u.Unit) -> ScalarArray:
        ndarray = self.ndarray
        if not isinstance(ndarray, u.Quantity):
            ndarray = ndarray << u.dimensionless_unscaled
        return ScalarArray(
            ndarray=ndarray.to(unit),
            axes=self.axes,
        )

    def ndarray_aligned(self: Self, shape: dict[str, int]) -> np.ndarray:
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
        ndarray = self.ndarray
        ndim_missing = len(shape) - np.ndim(ndarray)
        value = np.expand_dims(ndarray, tuple(~np.arange(ndim_missing)))
        source = []
        destination = []
        for axis_index, axis_name in enumerate(self.axes):
            source.append(axis_index)
            if axis_name not in shape:
                raise ValueError(
                    f"'shape' is missing dimensions. 'shape' is {shape} but 'self.shape' is {self.shape}"
                )
            destination.append(list(shape.keys()).index(axis_name))
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
            axes: Sequence[str],
            axis_new: None | str = None,
    ) -> ScalarArray:

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

        if function is np.matmul:
            function = np.multiply

        inputs_normalized = []
        for inp in inputs:
            if isinstance(inp, self.type_ndarray):
                inp = ScalarArray(inp)
            elif isinstance(inp, AbstractScalar):
                pass
            elif inp is None:
                return None
            else:
                return NotImplemented
            inputs_normalized.append(inp)
        inputs = inputs_normalized

        shape = na.shape_broadcasted(*inputs)
        inputs = tuple(inp.ndarray_aligned(shape) for inp in inputs)

        if 'out' in kwargs:
            kwargs['out'] = tuple(o.ndarray_aligned(shape) for o in kwargs['out'])

        if 'where' in kwargs:
            if isinstance(kwargs['where'], na.AbstractArray):
                kwargs['where'] = kwargs['where'].ndarray_aligned(shape)

        for inp in inputs:
            result = inp.__array_ufunc__(function, method, *inputs, **kwargs)
            if result is not NotImplemented:
                axes=tuple(shape.keys())
                if function.nout > 1:
                    return tuple(self.type_array(r, axes=axes) for r in result)
                else:
                    return ScalarArray(result, axes=axes)

        return NotImplemented

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

        from . import array_functions

        if func in array_functions.DEFAULT_FUNCTIONS:
            return array_functions.array_function_default(func, *args, **kwargs)

        if func in array_functions.PERCENTILE_LIKE_FUNCTIONS:
            return array_functions.array_function_percentile_like(func, *args, **kwargs)

        if func in array_functions.ARG_REDUCE_FUNCTIONS:
            return array_functions.array_function_arg_reduce(func, *args, **kwargs)

        if func in array_functions.FFT_LIKE_FUNCTIONS:
            return array_functions.array_function_fft_like(func, *args, **kwargs)

        if func in array_functions.FFTN_LIKE_FUNCTIONS:
            return array_functions.array_function_fftn_like(func, *args, **kwargs)

        if func in array_functions.HANDLED_FUNCTIONS:
            return array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def _getitem(
            self: Self,
            item: dict[str, int | slice | AbstractScalarArray] | AbstractScalarArray,
    ):

        if isinstance(item, AbstractScalarArray):
            value = np.moveaxis(
                a=self.ndarray,
                source=[self.axes.index(axis) for axis in item.axes],
                destination=np.arange(len(item.axes)),
            )

            return ScalarArray(
                ndarray=np.moveaxis(value[item.ndarray], 0, ~0),
                axes=tuple(axis for axis in self.axes if axis not in item.axes) + ('boolean', )
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
                if ax in item_advanced:
                    item_axis = item_axis.ndarray_aligned(shape_advanced)
                index[axes_organized.index(ax)] = item_axis
                if not isinstance(item_axis, slice):
                    axes_new.remove(ax)

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
        shape = self.shape
        if shape[axis_rows] != shape[axis_columns]:
            raise ValueError('Matrix must be square')

        axis_rows_inverse = axis_columns
        axis_columns_inverse = axis_rows

        if shape[axis_rows] == 1:
            return 1 / self

        elif shape[axis_rows] == 2:
            result = ScalarArray(ndarray=self.ndarray.copy(), axes=self.axes.copy())
            result[{axis_rows_inverse: 0, axis_columns_inverse: 0}] = self[{axis_rows: 1, axis_columns: 1}]
            result[{axis_rows_inverse: 1, axis_columns_inverse: 1}] = self[{axis_rows: 0, axis_columns: 0}]
            result[{axis_rows_inverse: 0, axis_columns_inverse: 1}] = -self[{axis_rows: 0, axis_columns: 1}]
            result[{axis_rows_inverse: 1, axis_columns_inverse: 0}] = -self[{axis_rows: 1, axis_columns: 0}]
            return result / self.matrix_determinant(axis_rows=axis_rows, axis_columns=axis_columns)

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

            result = ScalarArray(ndarray=self.array.copy(), axes=self.axes.copy())
            result[{axis_rows_inverse: 0, axis_columns_inverse: 0}] = (e * i - f * h)
            result[{axis_rows_inverse: 0, axis_columns_inverse: 1}] = -(b * i - c * h)
            result[{axis_rows_inverse: 0, axis_columns_inverse: 2}] = (b * f - c * e)
            result[{axis_rows_inverse: 1, axis_columns_inverse: 0}] = -(d * i - f * g)
            result[{axis_rows_inverse: 1, axis_columns_inverse: 1}] = (a * i - c * g)
            result[{axis_rows_inverse: 1, axis_columns_inverse: 2}] = -(a * f - c * d)
            result[{axis_rows_inverse: 2, axis_columns_inverse: 0}] = (d * h - e * g)
            result[{axis_rows_inverse: 2, axis_columns_inverse: 1}] = -(a * h - b * g)
            result[{axis_rows_inverse: 2, axis_columns_inverse: 2}] = (a * e - b * d)
            return result / self.matrix_determinant(axis_rows=axis_rows, axis_columns=axis_columns)

        else:
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


@dataclasses.dataclass(eq=False)
class ScalarArray(
    AbstractScalarArray,
    na.AbstractExplicitArray,
    Generic[NDArrayT],
):
    ndarray: NDArrayT = dataclasses.MISSING
    axes: None | tuple[str, ...] = None

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
    def empty(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        return cls(
            ndarray=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axes=tuple(shape.keys()),
        )

    @classmethod
    def zeros(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        return cls(
            ndarray=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axes=tuple(shape.keys()),
        )

    @classmethod
    def ones(cls: Type[Self], shape: dict[str, int], dtype: Type = float) -> Self:
        return cls(
            ndarray=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axes=tuple(shape.keys()),
        )

    @property
    def array(self: Self) -> Self:
        return self

    @property
    def centers(self: Self) -> Self:
        return self

    def __setitem__(
            self: Self,
            key: dict[str, int | slice | AbstractScalar] | AbstractScalar,
            value: int | float | u.Quantity | AbstractScalar,
    ) -> None:

        if not isinstance(value, AbstractScalar):
            value = ScalarArray(value)

        if isinstance(key, ScalarArray):
            shape = self.shape_broadcasted(key)
            self.ndarray_aligned(shape)[key.ndarray_aligned(shape)] = value.ndarray

        else:
            key_casted = cast(dict[str, Union[int, slice, AbstractScalar]], key)
            index = [slice(None)] * self.ndim   # type: list[Union[int, slice, AbstractScalar]]
            axes = list(self.axes)
            for axis in key_casted:
                item_axis = key_casted[axis]
                if isinstance(item_axis, int):
                    axes.remove(axis)
                if isinstance(item_axis, ScalarArray):
                    item_axis = item_axis.ndarray_aligned(self.shape_broadcasted(item_axis))
                index[self.axes.index(axis)] = item_axis

            if value.shape:
                value = value.ndarray_aligned({axis: 1 for axis in axes})
            else:
                value = value.ndarray

            self.ndarray[tuple(index)] = value


@dataclasses.dataclass(eq=False)
class AbstractImplicitScalarArray(
    AbstractScalarArray,
    na.AbstractImplicitArray,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractScalarRandomSample(
    AbstractImplicitScalarArray,
    na.AbstractRandomSample,
):
    pass


@dataclasses.dataclass(eq=False)
class ScalarUniformRandomSample(
    AbstractScalarRandomSample,
    na.AbstractUniformRandomSample,
    Generic[StartT, StopT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    shape_random: dict[str, int] = None
    seed: None | int = None

    @property
    def array(self: Self) -> ScalarArray:
        start = self.start
        if not isinstance(start, na.AbstractArray):
            start = ScalarArray(start)

        stop = self.stop
        if not isinstance(stop, na.AbstractArray):
            stop = ScalarArray(stop)

        shape_random = self.shape_random if self.shape_random is not None else dict()
        shape = na.shape_broadcasted(start, stop) | shape_random

        start = start.ndarray_aligned(shape)
        stop = stop.ndarray_aligned(shape)

        unit = None
        if isinstance(start, u.Quantity):
            unit = start.unit
            start = start.value
            stop = stop.to(unit).value

        value = self._rng.uniform(
            low=start,
            high=stop,
            size=tuple(shape.values()),
        )

        if unit is not None:
            value = value << unit

        return ScalarArray(
            ndarray=value,
            axes=tuple(shape.keys())
        )

    @property
    def centers(self: Self) -> Self:
        return self


@dataclasses.dataclass(eq=False)
class ScalarNormalRandomSample(
    AbstractScalarRandomSample,
    na.AbstractNormalRandomSample,
    Generic[CenterT, WidthT],
):
    center: CenterT = dataclasses.MISSING
    width: WidthT = dataclasses.MISSING
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def array(self: Self) -> ScalarArray:
        center = self.center
        if not isinstance(center, na.AbstractArray):
            center = ScalarArray(center)

        width = self.width
        if not isinstance(width, na.AbstractArray):
            width = ScalarArray(width)

        shape_random = self.shape_random if self.shape_random is not None else dict()
        shape = na.shape_broadcasted(center, width) | shape_random

        center = center.ndarray_aligned(shape)
        width = width.ndarray_aligned(shape)

        unit = None
        if isinstance(center, u.Quantity):
            unit = center.unit
            center = center.value
            width = width.to(unit).value

        value = self._rng.normal(
            loc=center,
            scale=width,
            size=tuple(shape.values()),
        )

        if unit is not None:
            value = value << unit

        return ScalarArray(
            ndarray=value,
            axes=tuple(shape.keys())
        )

    @property
    def centers(self: Self) -> Self:
        return self


@dataclasses.dataclass(eq=False)
class AbstractParameterizedScalarArray(
    AbstractImplicitScalarArray,
    na.AbstractParameterizedArray,
):
    pass


@dataclasses.dataclass(eq=False)
class ScalarArrayRange(
    AbstractParameterizedScalarArray,
    na.AbstractArrayRange,
    Generic[StartT, StopT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    axis: str = dataclasses.MISSING
    step: int = 1

    @property
    def array(self: Self) -> ScalarArray:
        return ScalarArray(
            ndarray=np.arange(
                start=self.start,
                stop=self.stop,
                step=self.step,
            ),
            axes=(self.axis, ),
        )

    @property
    def centers(self: Self) -> Self:
        return self

    @property
    def num(self: Self) -> int:
        return int(np.ceil((self.stop - self.start) / self.step))


@dataclasses.dataclass(eq=False)
class AbstractScalarSpace(
    AbstractParameterizedScalarArray,
    na.AbstractSpace,
):
    pass


@dataclasses.dataclass(eq=False)
class ScalarLinearSpace(
    AbstractScalarSpace,
    na.AbstractLinearSpace,
    Generic[StartT, StopT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    axis: str = dataclasses.MISSING
    num: int = 11
    endpoint: bool = False

    @property
    def array(self: Self) -> ScalarArray:
        start = self.start
        if not isinstance(start, AbstractScalar):
            start = ScalarArray(start)
        stop = self.stop
        if not isinstance(stop, AbstractScalar):
            stop = ScalarArray(stop)
        shape = na.shape_broadcasted(start, stop)
        return ScalarArray(
            ndarray=np.linspace(
                start=start.ndarray_aligned(shape),
                stop=stop.ndarray_aligned(shape),
                num=self.num,
                endpoint=self.endpoint,
                axis=~0,
            ),
            axes=tuple(shape.keys()) + (self.axis, )
        )

    @property
    def centers(self: Self) -> Self:
        return self

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

    def interp_linear(
            self: Self,
            item: dict[str, Self],
    ) -> AbstractScalar:

        item = item.copy()

        if self.axis in item:

            x = item.pop(self.axis)
            x0 = 0

            y0 = self.start.interp_linear(item)
            y1 = self.stop.interp_linear(item)

            result = y0 + (x - x0) * (y1 - y0)
            return result

        else:
            return type(self)(
                start=self.start.interp_linear(item),
                stop=self.stop.interp_linear(item),
                num=self.num,
                endpoint=self.endpoint,
                axis=self.axis,
            )

#
# @dataclasses.dataclass(eq=False)
# class _RandomSpaceMixin(_SpaceMixin):
#
#     seed: typ.Optional[int] = None
#
#     def __post_init__(self):
#         if self.seed is None:
#             self.seed = random.randint(0, 10 ** 12)
#
#     @property
#     def _rng(self: _RandomSpaceMixinT) -> np.random.Generator:
#         return np.random.default_rng(seed=self.seed)


@dataclasses.dataclass(eq=False)
class ScalarStratifiedRandomSpace(
    ScalarLinearSpace[StartT, StopT],
    na.AbstractStratifiedRandomSpace,
):
    seed: None | int = None

    @property
    def array(self: Self) -> ScalarArray:
        result = self.centers

        step_size = self.step

        delta = ScalarUniformRandomSample(
            start=-step_size / 2,
            stop=step_size / 2,
            axis=self.axis,
            num=self.num,
            seed=self.seed,
        )

        return result + delta

    @property
    def centers(self: Self) -> ScalarLinearSpace:
        return ScalarLinearSpace(
            start=self.start,
            stop=self.stop,
            num=self.num,
            endpoint=self.endpoint,
            axis=self.axis,
        )


@dataclasses.dataclass(eq=False)
class ScalarLogarithmicSpace(
    AbstractScalarSpace,
    na.AbstractLogarithmicSpace,
    Generic[StartExponentT, StopExponentT, BaseT]
):
    start_exponent: StartExponentT = dataclasses.MISSING
    stop_exponent: StopExponentT = dataclasses.MISSING
    base: BaseT = dataclasses.MISSING
    axis: str = dataclasses.MISSING
    num: int = 11
    endpoint: bool = False

    @property
    def array(self: Self) -> ScalarArray:
        start_exponent = self.start_exponent
        if not isinstance(start_exponent, AbstractScalar):
            start_exponent = ScalarArray(start_exponent)
        stop_exponent = self.stop_exponent
        if not isinstance(stop_exponent, AbstractScalar):
            stop_exponent = ScalarArray(stop_exponent)
        base = self.base
        if not isinstance(base, AbstractScalar):
            base = ScalarArray(base)
        shape = na.shape_broadcasted(start_exponent, stop_exponent, base)
        return ScalarArray(
            ndarray=np.logspace(
                start=start_exponent.ndarray_aligned(shape),
                stop=stop_exponent.ndarray_aligned(shape),
                num=self.num,
                endpoint=self.endpoint,
                base=base.ndarray_aligned(shape),
                axis=~0,
            ),
            axes=tuple(shape.keys()) + (self.axis, )
        )

    @property
    def centers(self: Self) -> Self:
        return self


@dataclasses.dataclass(eq=False)
class ScalarGeometricSpace(
    AbstractScalarSpace,
    na.AbstractGeometricSpace,
    Generic[StartT, StopT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    axis: str = dataclasses.MISSING
    num: int = 11
    endpoint: bool = False

    @property
    def array(self: Self) -> ScalarArray:
        start = self.start
        if not isinstance(start, AbstractScalar):
            start = ScalarArray(start)
        stop = self.stop
        if not isinstance(stop, AbstractScalar):
            stop = ScalarArray(stop)
        shape = na.shape_broadcasted(start, stop)
        return ScalarArray(
            ndarray=np.geomspace(
                start=start.ndarray_aligned(shape),
                stop=stop.ndarray_aligned(shape),
                num=self.num,
                endpoint=self.endpoint,
                axis=~0,
            ),
            axes=tuple(shape.keys()) + (self.axis, )
        )

    @property
    def centers(self: Self) -> Self:
        return self
