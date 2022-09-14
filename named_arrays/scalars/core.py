from __future__ import annotations
from typing import TypeVar, Generic, ClassVar, Type, Sequence, Callable, Collection, Any, Union, cast, Dict

import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import astropy.units as u

import named_arrays.core as na

__all__ = [
    'AbstractScalar',
    'AbstractScalarArray',
    'ScalarLike',
    'ScalarArray',
    'ScalarRange',
    'ScalarLinearSpace',
    'ScalarUniformRandomSpace',
]

DType = TypeVar('DType', bound=npt.DTypeLike)
NDArrayT = TypeVar('NDArrayT', bound=npt.ArrayLike)
AbstractScalarT = TypeVar('AbstractScalarT', bound='AbstractScalar')
AbstractScalarArrayT = TypeVar('AbstractScalarArrayT', bound='AbstractScalarArray')
ScalarArrayT = TypeVar('ScalarArrayT', bound='ScalarArray')
StartT = TypeVar('StartT', bound='ScalarLike')
StopT = TypeVar('StopT', bound='ScalarLike')
ScalarRangeT = TypeVar('ScalarRangeT', bound='ScalarRange')
ScalarLinearSpaceT = TypeVar('ScalarLinearSpaceT', bound='ScalarLinearSpace')
ScalarUniformRandomSpaceT = TypeVar('ScalarUniformRandomSpaceT', bound='ScalarUniformRandomSpace')
ScalarStratifiedRandomSpaceT = TypeVar('ScalarStratifiedRandomSpaceT', bound='ScalarStratifiedRandomSpace')
CenterT = TypeVar('CenterT', bound='ScalarLike')
WidthT = TypeVar('WidthT', bound='ScalarLike')
ScalarNormalRandomSpaceT = TypeVar('ScalarNormalRandomSpaceT', bound='ScalarNormalRandomSpace')


@dataclasses.dataclass(eq=False)
class AbstractScalar(
    na.AbstractArray,
):
    @property
    def scalar(self: AbstractScalarT) -> AbstractScalarT:
        return self

    @property
    def components(self: AbstractScalarT) -> dict[str, AbstractScalarT]:
        return {'': self}

    def shape_broadcasted(self: AbstractScalarArrayT, *arrays: na.AbstractArray) -> dict[str, int]:
        return na.shape_broadcasted(self, *arrays)


@dataclasses.dataclass(eq=False)
class AbstractScalarArray(
    AbstractScalar,
    Generic[NDArrayT],
):

    type_array_primary: ClassVar[Type] = np.ndarray
    type_array_auxiliary: ClassVar[tuple[Type, ...]] = (str, bool, int, float, complex, np.generic)
    type_array: ClassVar[tuple[Type, ...]] = type_array_auxiliary + (type_array_primary, )

    def astype(
            self: AbstractScalarArrayT,
            dtype: npt.DTypeLike,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> ScalarArray:
        return ScalarArray(
            ndarray=self.ndarray.astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            ),
            axes=self.axes,
        )

    def to(self: AbstractScalarArrayT, unit: u.Unit) -> ScalarArray:
        ndarray = self.ndarray
        if not isinstance(ndarray, u.Quantity):
            ndarray = ndarray << u.dimensionless_unscaled
        return ScalarArray(
            ndarray=ndarray.to(unit),
            axes=self.axes.copy(),
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

    def ndarray_aligned(self: AbstractScalarArrayT, shape: dict[str, int]) -> NDArrayT:
        ndarray = self.ndarray
        ndim_missing = len(shape) - np.ndim(ndarray)
        value = np.expand_dims(ndarray, tuple(~np.arange(ndim_missing)))
        source = []
        destination = []
        for axis_index, axis_name in enumerate(self.axes):
            source.append(axis_index)
            destination.append(list(shape.keys()).index(axis_name))
        value = np.moveaxis(value, source=source, destination=destination)
        return value

    def aligned(self: AbstractScalarArrayT, shape: dict[str, int]) -> ScalarArray:
        return ScalarArray(ndarray=self.ndarray_aligned(shape), axes=list(shape.keys()))

    def add_axes(self: AbstractScalarArrayT, axes: list[str]) -> ScalarArray:
        shape_new = {axis: 1 for axis in axes}
        shape = {**self.shape, **shape_new}
        return ScalarArray(
            ndarray=self.ndarray_aligned(shape),
            axes=list(shape.keys()),
        )

    def change_axis_index(self: AbstractScalarArrayT, axis: str, index: int) -> ScalarArray:
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
            axes=keys,
        )

    def combine_axes(
            self: AbstractScalarArrayT,
            axes: Sequence[str],
            axis_new: None | str = None,
    ) -> ScalarArray:

        if axis_new is None:
            axis_new = ''.join(axes)

        axes_new = self.axes.copy()
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
            axes=axes_new,
        )

    def matrix_multiply(
            self: AbstractScalarArrayT,
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
            self: AbstractScalarArrayT,
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

            axes_new = self.axes.copy()
            axes_new.remove(axis_rows)
            axes_new.remove(axis_columns)

            return ScalarArray(
                ndarray=np.linalg.det(value),
                axes=axes_new,
            )

    def matrix_inverse(
            self: AbstractScalarArrayT,
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

            axes_new = self.axes.copy()
            axes_new.remove(axis_rows)
            axes_new.remove(axis_columns)
            axes_new.append(axis_rows_inverse)
            axes_new.append(axis_columns_inverse)

            return ScalarArray(
                ndarray=np.linalg.inv(value),
                axes=axes_new,
            )

    def __mul__(self: AbstractScalarArrayT, other: na.ArrayLike | u.Unit) -> ScalarArray:
        if isinstance(other, u.UnitBase):
            return ScalarArray(
                ndarray=self.ndarray * other,
                axes=self.axes.copy(),
            )
        else:
            return super().__mul__(other)

    def __lshift__(self: AbstractScalarArrayT, other: u.UnitBase) -> ScalarArray:
        axes = self.axes
        if axes is not None:
            axes = axes.copy()
        return ScalarArray(
            ndarray=self.ndarray << other,
            axes=axes
        )

    def __array_ufunc__(
            self,
            function,
            method,
            *inputs,
            **kwargs,
    ) -> None | ScalarArray:

        inputs_normalized = []

        for inp in inputs:
            if isinstance(inp, self.type_array):
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

        for inp in inputs:
            result = inp.__array_ufunc__(function, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return ScalarArray(
                    ndarray=result,
                    axes=list(shape.keys()),
                )

        return NotImplemented

    def __array_function__(
            self: AbstractScalarArrayT,
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
                axes=list(shape.keys()),
            )

        elif func is np.moveaxis:

            args = list(args)

            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')
            a = ScalarArray(ndarray=a.ndarray, axes=a.axes.copy())

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
                if src in a.axes:
                    a.axes[a.axes.index(src)] = dest

            return a

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
                axes=list(shape.keys()),
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
                    axes=self.axes.copy(),
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
                axes=[axis] + list(shape.keys()),
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

            axes = list(shape.keys())
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

            return {a.axes[r]: ScalarArray(result[r], axes=['nonzero']) for r, _ in enumerate(result)}

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
                axes=self.axes.copy(),
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
                    axes=axes_new,
                )
        else:
            raise ValueError(f'{func} not supported')

    def __bool__(self: AbstractScalarArrayT) -> bool:
        return self.ndarray.__bool__()

    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: typ.Dict[str, int]) -> 'Array': ...
    #
    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: typ.Dict[str, slice]) -> 'Array': ...
    #
    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: typ.Dict[str, AbstractArrayT]) -> 'Array': ...
    #
    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: 'AbstractArray') -> 'Array': ...

    def __getitem__(
            self: AbstractScalarArrayT,
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
                axes=[axis for axis in self.axes if axis not in item.axes] + ['boolean']
            )

        else:
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

            axes = self.axes.copy()
            for a, axis in enumerate(axes_advanced):
                axes.remove(axis)
                axes.insert(a, axis)

            axes_new = axes.copy()
            index = [slice(None)] * self.ndim   # type: list[int | slice | AbstractScalar]
            for axis_name in item_casted:
                item_axis = item_casted[axis_name]
                if isinstance(item_axis, AbstractScalar):
                    item_axis = item_axis.ndarray_aligned(shape_advanced)
                if axis_name not in axes:
                    continue
                index[axes.index(axis_name)] = item_axis
                if not isinstance(item_axis, slice):
                    axes_new.remove(axis_name)

            return ScalarArray(
                ndarray=value[tuple(index)],
                axes=list(shape_advanced.keys()) + axes_new,
            )
        # else:
        #     raise ValueError('Invalid index type')

    def filter_median(
            self: AbstractScalarArrayT,
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
            axes=self.axes.copy(),
        )
        if isinstance(self.ndarray, u.Quantity):
            result = result << self.unit

        return result


ScalarLike = Union[na.QuantityLike, AbstractScalar]


@dataclasses.dataclass(eq=False)
class ScalarArray(
    AbstractScalarArray,
    na.ArrayBase,
    Generic[NDArrayT],
):
    ndarray: NDArrayT = 0 * u.dimensionless_unscaled
    axes: None | list[str] = None

    def __post_init__(self: ScalarArrayT):
        if self.axes is None:
            self.axes = []
        if getattr(self.ndarray, 'ndim', 0) != len(self.axes):
            raise ValueError('The number of axis names must match the number of dimensions.')
        if len(self.axes) != len(set(self.axes)):
            raise ValueError(f'Each axis name must be unique, got {self.axes}.')

    @classmethod
    def empty(cls: Type[ScalarArrayT], shape: dict[str, int], dtype: npt.DTypeLike = float) -> ScalarArrayT:
        return cls(
            ndarray=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @classmethod
    def zeros(cls: Type[ScalarArrayT], shape: dict[str, int], dtype: npt.DTypeLike = float) -> ScalarArrayT:
        return cls(
            ndarray=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @classmethod
    def ones(cls: Type[ScalarArrayT], shape: dict[str, int], dtype: npt.DTypeLike = float) -> ScalarArrayT:
        return cls(
            ndarray=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @property
    def array(self: ScalarArrayT) -> ScalarArrayT:
        return self

    @property
    def nominal(self: ScalarArrayT) -> ScalarArrayT:
        return self

    @property
    def distribution(self: ScalarArrayT) -> None:
        return None

    @property
    def centers(self: ScalarArrayT) -> ScalarArrayT:
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
            self: ScalarArrayT,
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
            axes = self.axes.copy()
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
class ScalarRangeMixin(
    na.AbstractRangeMixin,
    Generic[StartT, StopT],
):
    start: StartT = 0
    stop: StopT = 1
    axis: None | str = None


@dataclasses.dataclass(eq=False)
class ScalarRange(
    ScalarRangeMixin[StartT, StopT],
    AbstractScalarArray,
    na.AbstractRange,
):
    step: int = 1

    @property
    def array(self: ScalarRangeT) -> ScalarArray:
        return ScalarArray(
            ndarray=np.arange(
                start=self.start,
                stop=self.stop,
                step=self.step,
            ),
            axes=[self.axis],
        )

    @property
    def nominal(self: ScalarRangeT) -> ScalarRangeT:
        return self

    @property
    def distribution(self: ScalarRangeT) -> None:
        return None

    @property
    def centers(self: ScalarRangeT) -> ScalarRangeT:
        return self


# @dataclasses.dataclass(eq=False)
# class _SpaceMixin(
#     AbstractScalar[kgpy.units.QuantityLike],
# ):
#     num: int = None
#     endpoint: bool = True
#     axis: str = None
#
#     @property
#     def shape(self: _SpaceMixinT) -> typ.Dict[str, int]:
#         shape = super().shape
#         shape[self.axis] = self.num
#         return shape
#
#     @property
#     def axes(self: _SpaceMixinT) -> typ.List[str]:
#         return list(self.shape.keys())
#
#
# @dataclasses.dataclass(eq=False)
# class _RangeMixin(
#     AbstractScalar[kgpy.units.QuantityLike],
#     typ.Generic[StartArrayT, StopArrayT],
# ):
#     start: StartArrayT = None
#     stop: StopArrayT = None
#
#     @property
#     def normalized(self: _RangeMixinT) -> _RangeMixinT:
#         other = super().normalized
#         if not isinstance(other.start, ArrayInterface):
#             other.start = ScalarArray(other.start)
#         if not isinstance(other.stop, ArrayInterface):
#             other.stop = ScalarArray(other.stop)
#         return other
#
#     @property
#     def unit(self) -> typ.Union[float, u.Unit]:
#         unit = super().unit
#         if hasattr(self.start, 'unit'):
#             unit = self.start.unit
#         return unit
#
#     @property
#     def range(self: _RangeMixinT) -> ScalarArray:
#         return self.stop - self.start
#
#     @property
#     def shape(self: _RangeMixinT) -> typ.Dict[str, int]:
#         norm = self.normalized
#         return dict(**super().shape, **self.broadcast_shapes(norm.start, norm.stop))


@dataclasses.dataclass(eq=False)
class ScalarSpaceMixin(
    na.AbstractSpaceMixin,
):
    num: int = 11
    endpoint: bool = True


@dataclasses.dataclass(eq=False)
class ScalarLinearSpace(
    ScalarSpaceMixin,
    ScalarRangeMixin[StartT, StopT],
    AbstractScalarArray,
    na.AbstractLinearSpace,
):

    # @property
    # def step(self: LinearSpaceT) -> typ.Union[StartArrayT, StopArrayT]:
    #     if self.endpoint:
    #         return self.range / (self.num - 1)
    #     else:
    #         return self.range / self.num

    @property
    def array(self: ScalarLinearSpaceT) -> ScalarArray:
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
            axes=list(shape.keys()) + [self.axis]
        )

    @property
    def nominal(self: ScalarLinearSpaceT) -> ScalarLinearSpaceT:
        return self

    @property
    def distribution(self: ScalarLinearSpaceT) -> None:
        return None

    @property
    def centers(self: ScalarLinearSpaceT) -> ScalarLinearSpaceT:
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
            self: ScalarLinearSpaceT,
            item: dict[str, AbstractScalarArrayT],
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
class ScalarUniformRandomSpace(
    na.RandomMixin,
    ScalarLinearSpace[StartT, StopT],
    na.AbstractUniformRandomSpace,
):

    @property
    def array(self: ScalarUniformRandomSpaceT) -> ScalarArray:
        start = self.start
        if not isinstance(start, na.AbstractArray):
            start = ScalarArray(start)

        stop = self.stop
        if not isinstance(stop, na.AbstractArray):
            stop = ScalarArray(stop)

        shape = na.shape_broadcasted(start, stop)
        shape[self.axis] = self.num

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
        )

        if unit is not None:
            value = value << unit

        return ScalarArray(
            ndarray=value,
            axes=list(shape.keys())
        )


@dataclasses.dataclass(eq=False)
class ScalarStratifiedRandomSpace(
    na.RandomMixin,
    ScalarLinearSpace[StartT, StopT],
    na.AbstractStratifiedRandomSpace,
):
    # shape_extra: typ.Dict[str, int] = dataclasses.field(default_factory=dict)
    #
    # @property
    # def shape(self: StratifiedRandomSpaceT) -> typ.Dict[str, int]:
    #     return {**self.shape_extra, **super().shape}

    @property
    def array(self: ScalarStratifiedRandomSpaceT) -> ScalarArray:
        result = super().array

        step_size = self.step

        delta = ScalarUniformRandomSpace(
            start=-step_size / 2,
            stop=step_size / 2,
            axis=self.axis,
            num=self.num,
            seed=self.seed,
        )

        # norm = self.normalized
        # shape = norm.shape
        # shape[norm.axis] = norm.num
        # shape = {**shape, **self.shape_extra}
        # step_size = norm.step
        # step_size = step_size.broadcast_to(shape).array
        #
        # if isinstance(step_size, u.Quantity):
        #     unit = step_size.unit
        #     step_size = step_size.value
        # else:
        #     unit = None
        #
        # delta = self._rng.uniform(
        #     low=-step_size / 2,
        #     high=step_size / 2,
        # )
        #
        # if unit is not None:
        #     delta = delta << unit

        return result + delta

    @property
    def centers(self: ScalarStratifiedRandomSpaceT) -> ScalarLinearSpace:
        return ScalarLinearSpace(
            start=self.start,
            stop=self.stop,
            num=self.num,
            endpoint=self.endpoint,
            axis=self.axis,
        )


@dataclasses.dataclass(eq=False)
class SymmetricMixin(
    # AbstractScalar[kgpy.units.QuantityLike],
    # typ.Generic[CenterT, WidthT]
    na.AbstractSymmetricMixin,
    Generic[CenterT, WidthT],
):

    center: CenterT = 0
    width: WidthT = 0

    # @property
    # def normalized(self: _SymmetricMixinT) -> _SymmetricMixinT:
    #     other = super().normalized
    #     if not isinstance(other.center, ArrayInterface):
    #         other.center = ScalarArray(other.center)
    #     if not isinstance(other.width, ArrayInterface):
    #         other.width = ScalarArray(other.width)
    #     return other
    #
    # @property
    # def unit(self) -> typ.Optional[u.Unit]:
    #     unit = super().unit
    #     if hasattr(self.center, 'unit'):
    #         unit = self.center.unit
    #     return unit
    #
    # @property
    # def shape(self: _SymmetricMixinT) -> typ.Dict[str, int]:
    #     norm = self.normalized
    #     return dict(**super().shape, **self.broadcast_shapes(norm.width, norm.center))


@dataclasses.dataclass(eq=False)
class ScalarNormalRandomSpace(
    na.RandomMixin,
    ScalarSpaceMixin,
    SymmetricMixin[CenterT, WidthT],
    na.AbstractNormalRandomSpace,
):

    @property
    def array(self: ScalarUniformRandomSpaceT) -> ScalarArray:
        center = self.center
        if not isinstance(center, na.AbstractArray):
            center = ScalarArray(center)

        width = self.width
        if not isinstance(width, na.AbstractArray):
            width = ScalarArray(width)

        shape = na.shape_broadcasted(center, width)
        shape[self.axis] = self.num

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
        )

        if unit is not None:
            value = value << unit

        return ScalarArray(
            ndarray=value,
            axes=list(shape.keys())
        )



    # @property
    # def array(self: ScalarNormalRandomSpaceT) -> ScalarArray:
    #
    #     shape = self.shape
    #
    #     norm = self.normalized
    #     center = norm.center.broadcast_to(shape).array
    #     width = norm.width.broadcast_to(shape).array
    #
    #     unit = None
    #     if isinstance(center, u.Quantity):
    #         unit = center.unit
    #         center = center.value
    #         width = width.to(unit).value
    #
    #     value = self._rng.normal(
    #         loc=center,
    #         scale=width,
    #     )
    #
    #     if unit is not None:
    #         value = value << unit
    #
    #     return value


# ReferencePixelT = typ.TypeVar('ReferencePixelT', bound='kgpy.vectors.Cartesian')
#
#
# @dataclasses.dataclass(eq=False)
# class WorldCoordinateSpace(
#     AbstractScalar,
# ):
#
#     crval: ScalarArray
#     crpix: kgpy.vectors.CartesianND
#     cdelt: ScalarArray
#     pc_row: kgpy.vectors.CartesianND
#     shape_wcs: typ.Dict[str, int]
#
#     @property
#     def unit(self: WorldCoordinateSpaceT) -> u.UnitBase:
#         return self.cdelt.unit
#
#     @property
#     def normalized(self: WorldCoordinateSpaceT) -> WorldCoordinateSpaceT:
#         return self
#
#     def __call__(self: WorldCoordinateSpaceT, item: typ.Dict[str, AbstractScalarT]) -> ScalarArrayT:
#         import kgpy.vectors
#         coordinates_pix = kgpy.vectors.CartesianND(item) * u.pix
#         coordinates_pix = coordinates_pix - self.crpix
#         coordinates_world = 0 * u.pix
#         for component in coordinates_pix.coordinates:
#             pc_component = self.pc_row.coordinates[component]
#             if np.any(pc_component != 0):
#                 coordinates_world = coordinates_world + pc_component * coordinates_pix.coordinates[component]
#         coordinates_world = self.cdelt * coordinates_world + self.crval
#         return coordinates_world
#
#     def interp_linear(
#             self: WorldCoordinateSpaceT,
#             item: typ.Dict[str, AbstractScalarT],
#     ) -> ScalarArrayT:
#         return self(item)
#
#     @property
#     def array_labeled(self: WorldCoordinateSpaceT) -> ScalarArrayT:
#         return self(indices(self.shape_wcs))
#
#     @property
#     def array(self: WorldCoordinateSpaceT) -> ArrT:
#         return self.array_labeled.array
#
#     @property
#     def axes(self: WorldCoordinateSpaceT) -> typ.Optional[typ.List[str]]:
#         # return self.array_labeled.axes
#         return list(self.shape.keys())
#
#     @property
#     def shape(self: WorldCoordinateSpaceT) -> typ.Dict[str, int]:
#         shape = self(indices({axis: 1 for axis in self.shape_wcs})).shape
#         # shape_base = self.broadcast_shapes(self.crpix, self.crval, self.cdelt, self.pc_row)
#         shape = {axis: self.shape_wcs[axis] if axis in self.shape_wcs else shape[axis] for axis in shape}
#         # shape = {**shape_base, **self.shape_wcs}
#         return shape
#
#     def broadcast_to(
#             self: WorldCoordinateSpaceT,
#             shape: typ.Dict[str, int],
#     ) -> typ.Union[WorldCoordinateSpaceT, AbstractScalarT]:
#
#         if self.shape == shape:
#             if all(self.shape[axis_self] == shape[axis] for axis_self, axis in zip(self.shape, shape)):
#                 return self
#
#         return super().broadcast_to(shape)