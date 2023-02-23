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
    'AbstractScalarImplicitArray',
    'ScalarUniformRandomSample',
    'ScalarNormalRandomSample',
    'ScalarPoissonRandomSample',
    'AbstractScalarParameterizedArray',
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
    def components(self: Self) -> dict[str, Self]:
        return {'': self}

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
    def nominal(self: Self) -> Self:
        return self

    @property
    def distribution(self: Self) -> None:
        return None

    def astype(
            self: Self,
            dtype: npt.DTypeLike,
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

    # @property
    # def ndim(self: AbstractArrayT) -> int:
    #     return len(self.shape)
    #
    # @classmethod
    # def broadcast_shapes(cls: Type[AbstractArrayT], *arrays: AbstractArrayT) -> dict[str, int]:
    #     shape = dict()      # type: typ.Dict[str, int]
    #     for a in arrs:
    #         if hasattr(a, 'shape'):
    #             a_shape = a.shape
    #             for k in a_shape:
    #                 if k in shape:
    #                     shape[k] = max(shape[k], a_shape[k])
    #                 else:
    #                     shape[k] = a_shape[k]
    #     return shape

    def ndarray_aligned(self: Self, shape: dict[str, int]) -> np.ndarray:
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
            axis_new = ''.join(axes)

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

    def __bool__(self: Self) -> bool:
        return self.ndarray.__bool__()

    def __mul__(self: Self, other: na.ArrayLike | u.Unit) -> ScalarArray:
        if isinstance(other, u.UnitBase):
            return ScalarArray(
                ndarray=self.ndarray * other,
                axes=self.axes,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self: Self, other: u.UnitBase) -> ScalarArray:
        return ScalarArray(
            ndarray=self.ndarray << other,
            axes=self.axes
        )

    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | ScalarArray | tuple[ScalarArray, ...]:

        if function is np.matmul:
            raise ValueError('np.matmul not supported, please use named_arrays.AbstractScalarArray.matrix_inverse()')

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
        from . import array_functions

        if func in array_functions.DEFAULT_FUNCTIONS:
            return array_functions.array_function_default(func, *args, **kwargs)

        if func in array_functions.PERCENTILE_LIKE_FUNCTIONS:
            return array_functions.array_function_percentile_like(func, *args, **kwargs)

        if func in array_functions.ARG_REDUCE_FUNCTIONS:
            return array_functions.array_function_arg_reduce(func, *args, **kwargs)

        if func in array_functions.HANDLED_FUNCTIONS:
            return array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented


    def __array_function__old(
            self: Self,
            func: Callable,
            types: Collection,
            args: tuple,
            kwargs: dict[str, Any],
    ):
        if func is np.broadcast_to:
            args = list(args)
            if 'array' in kwargs:
                array = kwargs.pop('array')
            else:
                array = args.pop(0)

            if 'shape' in kwargs:
                shape = kwargs.pop('shape')
            else:
                shape = args.pop(0)

            return ScalarArray(
                ndarray=np.broadcast_to(array.ndarray_aligned(shape), tuple(shape.values()), subok=True),
                axes=tuple(shape.keys()),
            )

        elif func is np.shape:
            return self.shape

        elif func in (np.transpose, ):
            args = list(args)
            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')

            if args:
                axes = args.pop(0)
            else:
                axes = kwargs.pop('axes', None)

            axes_ndarray = tuple(self.axes.index(axis) for axis in axes) if axes is not None else axes
            ndarray_new = np.transpose(self.ndarray, axes=axes_ndarray)

            return self.type_array(ndarray_new, axes=axes if axes is not None else tuple(reversed(self.axes)))

        elif func is np.moveaxis:

            args = list(args)

            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')
            axes = list(a.axes)

            if args:
                source = args.pop(0)
            else:
                source = kwargs.pop('source')

            if args:
                destination = args.pop(0)
            else:
                destination = kwargs.pop('destination')

            types_sequence = (list, tuple,)
            if not isinstance(source, types_sequence):
                source = (source, )
            if not isinstance(destination, types_sequence):
                destination = (destination, )

            for src, dest in zip(source, destination):
                if src in axes:
                    axes[axes.index(src)] = dest

            return ScalarArray(
                ndarray=a.ndarray,
                axes=tuple(axes)
            )

        elif func is np.reshape:
            args = list(args)
            if 'a' in kwargs:
                array = kwargs.pop('a')
            else:
                array = args.pop(0)

            if 'newshape' in kwargs:
                shape = kwargs.pop('newshape')
            else:
                shape = args.pop(0)

            return ScalarArray(
                ndarray=np.reshape(array.ndarray, tuple(shape.values())),
                axes=tuple(shape.keys()),
            )

        elif func is np.result_type:
            return type(self)
        elif func is np.unravel_index:
            args = list(args)

            if args:
                indices = args.pop(0)
            else:
                indices = kwargs.pop('indices')

            if args:
                shape = args.pop(0)
            else:
                shape = kwargs.pop('shape')

            result_value = np.unravel_index(indices=indices.array, shape=tuple(shape.values()))
            result = dict()     # type: dict[str, ScalarArray]
            for axis, array in zip(shape, result_value):
                result[axis] = ScalarArray(
                    ndarray=array,
                    axes=self.axes,
                )
            return result

        elif func is np.linalg.inv:
            raise ValueError(f'{func} is unsupported, use kgpy.LabeledArray.matrix_inverse() instead.')

        elif func is np.stack:
            args = list(args)
            kwargs = kwargs.copy()

            if args:
                if 'arrays' in kwargs:
                    raise TypeError(f"{func} got multiple values for 'arrays'")
                arrays = args.pop(0)
            else:
                arrays = kwargs.pop('arrays')

            if args:
                if 'axis' in kwargs:
                    raise TypeError(f"{func} got multiple values for 'axis'")
                axis = args.pop(0)
            else:
                axis = kwargs.pop('axis')

            shape = na.shape_broadcasted(*arrays)
            arrays = [ScalarArray(arr) if not isinstance(arr, na.AbstractArray) else arr for arr in arrays]
            for array in arrays:
                if not isinstance(array, AbstractScalar):
                    return NotImplemented
            arrays = [arr.broadcast_to(shape).ndarray for arr in arrays]

            return ScalarArray(
                ndarray=np.stack(arrays, axis=0, **kwargs),
                axes=(axis, ) + tuple(shape.keys()),
            )

        elif func is np.concatenate:
            args = list(args)
            if args:
                arrays = args.pop(0)
            else:
                arrays = kwargs['arrays']

            if args:
                axis = args.pop(0)
            else:
                axis = kwargs['axis']

            arrays = [ScalarArray(arr) if not isinstance(arr, na.AbstractArray) else arr for arr in arrays]
            for arr in arrays:
                if not isinstance(arr, AbstractScalar):
                    return NotImplemented

            shape = self.broadcast_shapes(*arrays)
            arrays = [arr.broadcast_to(shape).ndarray for arr in arrays]

            axes = tuple(shape.keys())
            return ScalarArray(
                ndarray=func(arrays, axis=axes.index(axis)),
                axes=axes,
            )

        elif func is np.argsort:
            args = list(args)
            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')

            if args:
                axis = args.pop(0)
            else:
                axis = kwargs.pop('axis')

            result = func(a.ndarray, *args, axis=a.axes.index(axis), **kwargs)
            result = {axis: ScalarArray(result, axes=a.axes)}
            return result

        elif func is np.nonzero:
            args = list(args)
            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')
            result = func(a.ndarray, *args, **kwargs)

            return {a.axes[r]: ScalarArray(result[r], axes=('nonzero', )) for r, _ in enumerate(result)}

        elif func is np.histogram2d:
            args = list(args)
            if args:
                x = args.pop(0)
            else:
                x = kwargs.pop('x')

            if args:
                y = args.pop(0)
            else:
                y = kwargs.pop('y')

            shape = self.broadcast_shapes(x, y)
            x = x.broadcast_to(shape)
            y = y.broadcast_to(shape)

            bins = kwargs.pop('bins')           # type: dict[str, int]
            if not isinstance(bins[next(iter(bins))], int):
                raise NotImplementedError
            range = kwargs.pop('range')
            weights = kwargs.pop('weights')

            key_x, key_y = bins.keys()

            shape_hist = shape.copy()
            shape_hist[key_x] = bins[key_x]
            shape_hist[key_y] = bins[key_y]

            shape_edges_x = shape_hist.copy()
            shape_edges_x[key_x] = shape_edges_x[key_x] + 1
            shape_edges_x.pop(key_y)

            shape_edges_y = shape_hist.copy()
            shape_edges_y[key_y] = shape_edges_y[key_y] + 1
            shape_edges_y.pop(key_x)

            hist = ScalarArray.empty(shape_hist)
            edges_x = ScalarArray.empty(shape_edges_x) * x.unit
            edges_y = ScalarArray.empty(shape_edges_y) * y.unit

            for index in x.ndindex(axis_ignored=(key_x, key_y)):
                if range is not None:
                    range_index = [[elem.ndarray.value for elem in range[component]] for component in range]
                else:
                    range_index = None

                if weights is not None:
                    weights_index = weights[index].ndarray.reshape(-1)
                else:
                    weights_index = None

                hist[index].ndarray[:], edges_x[index].ndarray[:], edges_y[index].ndarray[:] = np.histogram2d(
                    x=x[index].array.reshape(-1),
                    y=y[index].array.reshape(-1),
                    bins=tuple(bins.values()),
                    range=range_index,
                    weights=weights_index,
                    **kwargs,
                )

            return hist, edges_x, edges_y

        elif func in [np.nan_to_num]:
            args = list(args)
            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')

            return ScalarArray(
                ndarray=func(a.ndarray, *args, **kwargs),
                axes=self.axes,
            )

        elif func in [
            np.ndim,
            np.argmin,
            np.nanargmin,
            np.min,
            np.nanmin,
            np.argmax,
            np.nanargmax,
            np.max,
            np.nanmax,
            np.sum,
            np.nansum,
            np.mean,
            np.nanmean,
            np.std,
            np.median,
            np.nanmedian,
            np.percentile,
            np.nanpercentile,
            np.all,
            np.any,
            np.array_equal,
            np.isclose,
            np.roll,
            np.clip,
            np.ptp,
            np.trapz,
        ]:

            arrays = [arg for arg in args if isinstance(arg, AbstractScalar)]
            arrays += [kwargs[k] for k in kwargs if isinstance(kwargs[k], AbstractScalar)]
            shape = na.shape_broadcasted(*arrays)
            axes = list(shape.keys())

            args = tuple(arg.ndarray_aligned(shape) if isinstance(arg, AbstractScalar) else arg for arg in args)
            kwargs = {k: kwargs[k].ndarray_aligned(shape) if isinstance(kwargs[k], AbstractScalar) else kwargs[k] for k in kwargs}
            types = tuple(type(arg) for arg in args if getattr(arg, '__array_function__', None) is not None)

            axes_new = axes.copy()
            if func not in [np.isclose, np.roll, np.clip]:
                if 'keepdims' not in kwargs:
                    if 'axis' in kwargs:
                        if kwargs['axis'] is None:
                            axes_new = []
                        elif np.isscalar(kwargs['axis']):
                            if kwargs['axis'] in axes_new:
                                axes_new.remove(kwargs['axis'])
                        else:
                            for axis in kwargs['axis']:
                                if axis in axes_new:
                                    axes_new.remove(axis)
                    else:
                        axes_new = []

            if 'axis' in kwargs:
                if kwargs['axis'] is None:
                    pass
                elif np.isscalar(kwargs['axis']):
                    if kwargs['axis'] in axes:
                        kwargs['axis'] = axes.index(kwargs['axis'])
                    else:
                        return self
                else:
                    kwargs['axis'] = tuple(axes.index(ax) for ax in kwargs['axis'] if ax in axes)

            ndarray = self.ndarray
            if not hasattr(ndarray, '__array_function__'):
                ndarray = np.array(ndarray)
            ndarray = ndarray.__array_function__(func, types, args, kwargs)

            if func in [
                np.array_equal,
            ]:
                return ndarray
            else:
                return ScalarArray(
                    ndarray=ndarray,
                    axes=tuple(axes_new),
                )
        else:
            raise ValueError(f'{func} not supported')

    def __getitem__(
            self: Self,
            item: dict[str, int | slice | AbstractScalar] | AbstractScalar,
    ) -> ScalarArray:

        if isinstance(item, AbstractScalar):
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
            item_casted = cast(Dict[str, Union[int, slice, AbstractScalar]], item)
            axes_advanced = []
            axes_indices_advanced = []
            item_advanced = dict()      # type: typ.Dict[str, AbstractScalar]
            for axis in item_casted:
                item_axis = item_casted[axis]
                if isinstance(item_axis, AbstractScalar):
                    axes_advanced.append(axis)
                    axes_indices_advanced.append(self.axes.index(axis))
                    item_advanced[axis] = item_axis

            shape_advanced = na.shape_broadcasted(*item_advanced.values())

            value = np.moveaxis(
                self.ndarray,
                source=axes_indices_advanced,
                destination=list(range(len(axes_indices_advanced))),
            )

            axes = list(self.axes)
            for a, axis in enumerate(axes_advanced):
                axes.remove(axis)
                axes.insert(a, axis)

            axes_new = axes.copy()
            index = [slice(None)] * self.ndim   # type: list[int | slice | AbstractScalar]
            for axis_name in item_casted:
                item_axis = item_casted[axis_name]
                if isinstance(item_axis, AbstractScalar):
                    item_axis = item_axis.ndarray_aligned(shape_advanced)
                index[axes.index(axis_name)] = item_axis
                if not isinstance(item_axis, slice):
                    axes_new.remove(axis_name)

            return ScalarArray(
                ndarray=value[tuple(index)],
                axes=tuple(shape_advanced.keys()) + tuple(axes_new),
            )

        else:
            raise ValueError('Invalid index type')

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


@dataclasses.dataclass(eq=False, slots=True)
class ScalarArray(
    AbstractScalarArray,
    na.AbstractExplicitArray,
    Generic[NDArrayT],
):
    """
    An array representing a scalar quantity with names for each of it's N axes.

    A :class:`ScalarArray` is defined by a :class:`numpy.ndarray` and an axis.

    .. jupyter-execute::

        import numpy as np
        import named_arrays.core as na

        x = np.array([1, 4, 9, 11, 17])
        x = na.ScalarArray(x,axes='position_x')
        print(x)

    They can also be defined using a :class:`astropy.units.Quantity`, or ascribed units after creation.

    .. jupyter-execute::

        import astropy.units as u

        x = x * u.cm

        print(x)

    What happens when we do math with :class:`ScalarArray`s?

    .. jupyter-execute::

        y = np.array([0, 2, 4, 5]) * u.mm

        y = na.ScalarArray(y, axes='position_y')
        print(y**2)

        radius = np.sqrt(y**2 + x**2)
        print(radius)
        print(radius.shape)


    Note how when performing mathematical operations on two :class:`ScalarArray`s with different axes, the arrays are
    automatically broadcast over every axis. There is no need to insert extra dimensions for alignment like you would
    normally do with instances of :class:`numpy.ndarray`.

    We can also use reduction operators (mean, sum, etc) on :class:`ScalarArray`s, and if desired specify the axis by name without
    knowledge of the corresponding index axes of the original array.

    .. jupyter-execute::

        print(radius.mean())
        print(radius.mean(axis='position_x'))
    """


    ndarray: NDArrayT
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

    # @property
    # def normalized(self: ArrayT) -> ArrayT:
    #     other = super().normalized
    #     if isinstance(other.ndarray, other.type_array_auxiliary):
    #         other.ndarray = np.array(other.ndarray)
    #     if other.axes is None:
    #         other.axes = []
    #     return other

    # @property
    # def unit(self) -> float | u.Unit:
    #     unit = super().unit
    #     if hasattr(self.ndarray, 'unit'):
    #         unit = self.ndarray.unit
    #     return unit

    # @property
    # def shape(self: ArrayT) -> dict[str, int]:
    #     shape = super().shape
    #     for i in range(np.ndim(self.ndarray)):
    #         shape[self.axes[i]] = self.array.shape[i]
    #     return shape

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
class AbstractScalarImplicitArray(
    AbstractScalarArray,
    na.AbstractImplicitArray,
):
    pass


@dataclasses.dataclass(eq=False, slots=True)
class AbstractScalarRandomSample(
    AbstractScalarImplicitArray,
    na.AbstractRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, slots=True)
class ScalarUniformRandomSample(
    AbstractScalarRandomSample,
    na.AbstractUniformRandomSample,
    Generic[StartT, StopT],
):
    start: StartT
    stop: StopT
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


@dataclasses.dataclass(eq=False, slots=True)
class ScalarNormalRandomSample(
    AbstractScalarRandomSample,
    na.AbstractNormalRandomSample,
    Generic[CenterT, WidthT],
):
    center: CenterT
    width: WidthT
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


@dataclasses.dataclass(eq=False, slots=True)
class ScalarPoissonRandomSample(
    AbstractScalarRandomSample,
    na.AbstractPoissonRandomSample,
    Generic[CenterT],
):
    center: CenterT
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def array(self: Self) -> ScalarArray:
        center = self.center
        if not isinstance(center, na.AbstractArray):
            center = ScalarArray(center)


        shape_random = self.shape_random if self.shape_random is not None else dict()
        shape = na.shape_broadcasted(center) | shape_random

        center = center.ndarray_aligned(shape)

        unit = None
        if isinstance(center, u.Quantity):
            unit = center.unit
            center = center.value

        value = self._rng.poisson(
            lam=center,
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


@dataclasses.dataclass(eq=False, slots=True)
class AbstractScalarParameterizedArray(
    AbstractScalarImplicitArray,
    na.AbstractParameterizedArray,
):
    pass


@dataclasses.dataclass(eq=False, slots=True)
class ScalarArrayRange(
    AbstractScalarParameterizedArray,
    na.AbstractArrayRange,
    Generic[StartT, StopT],
):
    """
    A :class:`ScalarArray` over the range [:attr:`start`, :attr:`stop`) incremented by :attr:`step`. An analog to :class:`numpy.arange`.

    :class:`ScalarArrayRange` can be used to create a :class:`ScalarArray` of integers.

    .. jupyter-execute::

        import named_arrays as na
        x = na.ScalarArrayRange(1, 8, axis = "x")
        print(x.array)
        print(x.shape)

    Note above that ``x`` does not include :attr:`stop`, and won't in almost all cases.  :class:`ScalarArrayRange` can be used to
    create an increasing ScalarArray of floats, even with non integer steps.

    .. jupyter-execute::

        x = na.ScalarArrayRange(-0.5, 3, "x", 0.25)
        print(x.array)

    For the above, and more complicated uses, it is recommended to use :class:`ScalarLinearSpace` instead.  See numpy
    documentation of :class:`numpy.arange` for more info.
    """
    start: StartT
    stop: StopT
    axis: str
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
    AbstractScalarParameterizedArray,
    na.AbstractSpace,
):
    pass


@dataclasses.dataclass(eq=False, slots=True)
class ScalarLinearSpace(
    AbstractScalarSpace,
    na.AbstractLinearSpace,
    Generic[StartT, StopT],
):
    """
    An evenly spaced :class:`ScalarArray` ranging from start to stop with num elements.

    Most often :class:`ScalarArray`s won't be formed directly from a :class:`numpy.ndarray`, but through more useful routines
    like :class:`ScalarLinearSpace`, a named_arrays equivalent to :class:`numpy.linspace`.  For example,
    one can quickly create an evenly spaced coordinate (or axis) array with units.

    .. jupyter-execute::

        import named_arrays as na
        import astropy.units as u

        photon_energy = na.ScalarLinearSpace(1, 25, axis="energy", num=25) * u.keV
        print(photon_energy.shape)
        print(photon_energy)

    Then easily convert that into a wavelength.

    .. jupyter-execute::

        wavelength = 1240 * u.eV * u.nm / photon_energy
        wavelength.axes = 'lambda'
        print(wavelength)

    """
    start: StartT
    stop: StopT
    axis: str
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
    #     return {self.axis: np.print((value - self.start) / self.step).astype(int)}
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


@dataclasses.dataclass(eq=False, slots=True)
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


@dataclasses.dataclass(eq=False, slots=True)
class ScalarLogarithmicSpace(
    AbstractScalarSpace,
    na.AbstractLogarithmicSpace,
    Generic[StartExponentT, StopExponentT, BaseT]
):
    start_exponent: StartExponentT
    stop_exponent: StopExponentT
    base: BaseT
    axis: str
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


@dataclasses.dataclass(eq=False, slots=True)
class ScalarGeometricSpace(
    AbstractScalarSpace,
    na.AbstractGeometricSpace,
    Generic[StartT, StopT],
):
    start: StartT
    stop: StopT
    axis: str
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
