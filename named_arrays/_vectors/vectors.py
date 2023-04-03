from __future__ import annotations
from typing import ClassVar, Type, Sequence, Callable, Collection, Any, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na


__all__ = [
    'AbstractVectorArray',
    'AbstractScalarOrVectorArray',
    'AbstractExplicitVectorArray',
    'AbstractImplicitVectorArray',
    'AbstractVectorRandomSample',
    'AbstractVectorUniformRandomSample',
    'AbstractVectorNormalRandomSample',
    'AbstractParameterizedVectorArray',
    'AbstractVectorArrayRange',
    'AbstractVectorSpace',
    'AbstractVectorLinearSpace',
    'AbstractVectorStratifiedRandomSpace',
    'AbstractVectorLogarithmicSpace',
    'AbstractVectorGeometricSpace',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorArray(
    na.AbstractArray
):

    __named_array_priority__: ClassVar[float] = 100 * na.AbstractScalarArray.__named_array_priority__

    @property
    @abc.abstractmethod
    def type_explicit(self: Self) -> Type[AbstractExplicitVectorArray]:
        pass

    @property
    @abc.abstractmethod
    def type_matrix(self) -> Type[na.AbstractExplicitMatrixArray]:
        """
        The corresponding :class:`named_arrays.AbstractMatrixArray` class
        """

    @property
    @abc.abstractmethod
    def components(self: Self) -> dict[str, na.ArrayLike]:
        """
        The vector components of this array expressed as a :class:`dict` where the keys are the names of the component.
        """
        return dict()

    @property
    def entries(self) -> dict[str, na.ArrayLike]:
        """
        The scalar entries that compose this object.
        """
        return self.components

    @property
    def value(self) -> na.AbstractExplicitVectorArray:
        components = self.components
        components = {c: na.value(components[c]) for c in components}
        return self.type_explicit.from_components(components)

    @property
    @abc.abstractmethod
    def explicit(self: Self) -> AbstractExplicitVectorArray:
        pass

    @property
    def centers(self: Self) -> AbstractExplicitVectorArray:
        components = self.components
        components_result = dict()
        for c in components:
            if isinstance(components[c], na.AbstractArray):
                components_result[c] = components[c].centers
            else:
                components_result[c] = components[c]
        return self.type_explicit.from_components(components_result)

    def astype(
            self: Self,
            dtype: str | np.dtype | Type | dict[str, str | np.dtype | Type],
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> Self:
        components = self.components

        components_normalized = dict()
        for c in components:
            if not isinstance(components[c], (np.ndarray, na.AbstractArray)):
                components_normalized[c] = np.array(components[c])
            else:
                components_normalized[c] = components[c]
        components = components_normalized

        if not isinstance(dtype, dict):
            dtype = {c: dtype for c in components}
        kwargs = dict(
            order=order,
            casting=casting,
            subok=subok,
            copy=copy,
        )
        return self.type_explicit.from_components({c: components[c].astype(dtype=dtype[c], **kwargs) for c in components})

    def to(self: Self, unit: u.UnitBase | dict[str, None | u.UnitBase]) -> AbstractExplicitVectorArray:
        components = self.components
        if not isinstance(unit, dict):
            unit = {c: unit for c in components}
        components_result = dict()
        for c in components:
            if unit[c] is not None:
                if not isinstance(components[c], (u.Quantity, na.AbstractArray)):
                    components_c = components[c] << u.dimensionless_unscaled
                else:
                    components_c = components[c]
                components_result[c] = components_c.to(unit[c])
            else:
                components_result[c] = components[c]
        return self.type_explicit.from_components(components_result)

    def add_axes(self: Self, axes: str | Sequence[str]) -> AbstractExplicitVectorArray:
        components = self.components
        return self.type_explicit.from_components({c: na.add_axes(components[c], axes) for c in components})

    def combine_axes(
            self: Self,
            axes: Sequence[str] = None,
            axis_new: str = None,
    ) -> AbstractExplicitVectorArray:

        shape = self.shape

        if axes is None:
            axes = tuple(shape)

        shape_base = {ax: shape[ax] for ax in shape if ax in axes}

        components = self.components
        components_result = dict()
        for c in components:
            shape_c = na.shape(components[c]) | shape_base
            components_result[c] = na.broadcast_to(components[c], shape=shape_c).combine_axes(
                axes=axes,
                axis_new=axis_new
            )

        return self.type_explicit.from_components(components_result)

    def _getitem(
            self: Self,
            item: dict[str, int | slice | AbstractScalarOrVectorArray] | AbstractScalarOrVectorArray,
    ) -> Self:

        array = self.explicit
        shape_array = array.shape
        components = array.components

        if isinstance(item, na.AbstractArray):
            item = item.explicit
            shape_item = item.shape

            shape_base = {ax: shape_array[ax] for ax in shape_item if ax in shape_array}
            for c in components:
                component = na.as_named_array(components[c])
                components[c] = component.broadcast_to(na.broadcast_shapes(component.shape, shape_base))

            if item.type_abstract == self.type_abstract:
                item_accumulated = True
                components_item = item.components
                for c in components_item:
                    item_accumulated = item_accumulated & components_item[c]
                item = self.type_explicit.from_scalar(item_accumulated)
            elif isinstance(item, na.AbstractScalar):
                item = self.type_explicit.from_scalar(item)
            else:
                return NotImplemented

        elif isinstance(item, dict):
            shape_base = {ax: shape_array[ax] for ax in item if ax in shape_array}
            for c in components:
                component = na.as_named_array(components[c])
                components[c] = component.broadcast_to(na.broadcast_shapes(component.shape, shape_base))

            item = item.copy()
            for ax in item:
                if isinstance(item[ax], na.AbstractArray):
                    if item[ax].type_abstract == self.type_abstract:
                        item[ax] = item[ax].explicit
                    elif isinstance(item[ax], na.AbstractScalar):
                        item[ax] = self.type_explicit.from_scalar(item[ax])
                    else:
                        return NotImplemented
                elif isinstance(item[ax], (int, slice)):
                    item[ax] = self.type_explicit.from_scalar(item[ax])
                elif item[ax] is None:
                    item[ax] = self.type_explicit.from_scalar(item[ax])
                else:
                    return NotImplemented

        else:
            return NotImplemented

        components_result = dict()
        for c in components:
            if isinstance(item, dict):
                components_result[c] = na.as_named_array(components[c])[{ax: item[ax].components[c] for ax in item}]
            else:
                components_result[c] = components[c][na.as_named_array(item.components[c])]

        return self.type_explicit.from_components(components_result)

    def _getitem_reversed(
            self: Self,
            array: na.ScalarArray,
            item: dict[str, int | slice | AbstractVectorArray] | AbstractVectorArray,
    ):
        if array.type_abstract == self.type_abstract:
            pass
        elif isinstance(array, na.ScalarArray):
            array = self.type_explicit.from_scalar(array)
        else:
            return NotImplemented
        return array._getitem(item)

    def __bool__(self: Self) -> bool:
        result = super().__bool__()
        components = self.components
        for c in components:
            result = result and bool(components[c])
        return result

    def __array_matmul__(
            self: Self,
            x1: na.ArrayLike,
            x2: na.ArrayLike,
            out: tuple[None | na.AbstractExplicitArray] = (None, ),
            **kwargs,
    ) -> na.AbstractExplicitArray:

        result = super().__array_matmul__(
            x1=x1,
            x2=x2,
            out=out,
            **kwargs,
        )
        if result is not NotImplemented:
            return result

        out = out[0]

        if isinstance(x1, AbstractVectorArray) and isinstance(x2, AbstractVectorArray):
            if x1.type_abstract == x2.type_abstract:
                components_x1 = x1.broadcasted.components
                components_x2 = x2.broadcasted.components
                result = 0
                for c in components_x1:
                    component_x1 = na.as_named_array(components_x1[c])
                    component_x2 = na.as_named_array(components_x2[c])
                    result = np.add(result, np.matmul(component_x1, component_x2), out=out)
                return result
            else:
                return NotImplemented

        else:
            return np.multiply(x1, x2, out=out, **kwargs)

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

        from . import vector_array_functions

        if func in vector_array_functions.DEFAULT_FUNCTIONS:
            return vector_array_functions.array_function_default(func, *args, **kwargs)

        if func in vector_array_functions.PERCENTILE_LIKE_FUNCTIONS:
            return vector_array_functions.array_function_percentile_like(func, *args, **kwargs)

        if func in vector_array_functions.ARG_REDUCE_FUNCTIONS:
            return vector_array_functions.array_function_arg_reduce(func, *args, **kwargs)

        if func in vector_array_functions.FFT_LIKE_FUNCTIONS:
            return vector_array_functions.array_function_fft_like(func, *args, **kwargs)

        if func in vector_array_functions.FFTN_LIKE_FUNCTIONS:
            return vector_array_functions.array_function_fftn_like(func, *args, **kwargs)

        if func in vector_array_functions.STACK_LIKE_FUNCTIONS:
            return vector_array_functions.array_function_stack_like(func, *args, **kwargs)

        if func in vector_array_functions.HANDLED_FUNCTIONS:
            return vector_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    @property
    def broadcasted(self: Self) -> Self:
        a = self.explicit
        return a.broadcast_to(a.shape)


AbstractScalarOrVectorArray = na.AbstractScalar | AbstractVectorArray


@dataclasses.dataclass(eq=False, repr=False)
class AbstractExplicitVectorArray(
    AbstractVectorArray,
    na.AbstractExplicitArray,
):

    @classmethod
    def from_components(
            cls: Type[Self],
            components: dict[str, na.AbstractArray],
    ) -> AbstractExplicitVectorArray:
        return cls(**components)

    @classmethod
    @abc.abstractmethod
    def from_scalar(
            cls: Type[Self],
            scalar: na.ScalarLike,
    ) -> AbstractExplicitVectorArray:
        """
        Convert a scalar (an instance of :class:`named_arrays.AbstractScalar`) into a vector.
        """

    @property
    def components(self: Self) -> dict[str, na.ArrayLike]:
        return self.__dict__

    @components.setter
    def components(self, value: dict[str, na.ArrayLike]) -> None:
        self.__dict__ = value

    @property
    def axes(self: Self) -> tuple[str, ...]:
        return tuple(self.shape.keys())

    @property
    def shape(self: Self) -> dict[str, int]:
        return na.shape_broadcasted(*self.components.values())

    @property
    def ndim(self: Self) -> int:
        return len(self.shape)

    @property
    def size(self: Self) -> int:
        return int(np.array(tuple(self.shape.values())).prod())

    @property
    def explicit(self: Self) -> AbstractExplicitVectorArray:
        components = self.components
        components_result = dict()
        for c in components:
            if isinstance(components[c], na.AbstractArray):
                components_result[c] = components[c].explicit
            else:
                components_result[c] = components[c]
        return self.type_explicit.from_components(components_result)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitVectorArray(
    AbstractVectorArray,
    na.AbstractImplicitArray,
):
    @property
    def components(self) -> dict[str, na.ArrayLike]:
        return self.explicit.components


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorRandomSample(
    AbstractImplicitVectorArray,
    na.AbstractRandomSample,
):
    pass


StartT = TypeVar('StartT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray)
StopT = TypeVar('StopT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorUniformRandomSample(
    AbstractVectorRandomSample,
    na.AbstractUniformRandomSample,
    Generic[StartT, StopT]
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def explicit(self) -> AbstractExplicitVectorArray:

        start = self.start
        if isinstance(start, na.AbstractArray):
            if start.type_abstract == self.type_abstract:
                pass
            elif isinstance(start, na.AbstractScalar):
                start = self.type_explicit.from_scalar(start)
            else:
                raise ValueError(
                    f"`start` must either be an instance of {float}, {u.Quantity}, {na.AbstractScalar},"
                    f" or {self.type_abstract}, got {type(start)}"
                )
        else:
            start = self.type_explicit.from_scalar(start)

        stop = self.stop
        if isinstance(stop, na.AbstractArray):
            if stop.type_abstract == self.type_abstract:
                pass
            elif isinstance(stop, na.AbstractScalar):
                stop = self.type_explicit.from_scalar(stop)
            else:
                raise ValueError(
                    f"`stop` must either be an instance of {float}, {u.Quantity}, {na.AbstractScalar},"
                    f" or {self.type_abstract}, got {type(stop)}"
                )
        else:
            stop = self.type_explicit.from_scalar(stop)

        seed = self.seed

        result = self.type_explicit()
        components_start = start.components
        components_stop = stop.components

        for c in result.components:
            result.components[c] = na.ScalarUniformRandomSample(
                start=components_start[c],
                stop=components_stop[c],
                shape_random=self.shape_random,
                seed=seed,
            ).explicit
            seed += 1

        return result


CenterT = TypeVar('CenterT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray)
WidthT = TypeVar('WidthT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorNormalRandomSample(
    AbstractVectorRandomSample,
    na.AbstractNormalRandomSample,
    Generic[CenterT, WidthT]
):
    center: CenterT = dataclasses.MISSING
    width: WidthT = dataclasses.MISSING
    shape_random: None | dict[str, int] = None
    seed: None | int = None

    @property
    def explicit(self) -> AbstractExplicitVectorArray:

        center = self.center
        if isinstance(center, na.AbstractArray):
            if center.type_abstract == self.type_abstract:
                pass
            elif isinstance(center, na.AbstractScalar):
                center = self.type_explicit.from_scalar(center)
            else:
                raise ValueError(
                    f"`start` must either be an instance of {float}, {u.Quantity}, {na.AbstractScalar},"
                    f" or {self.type_abstract}, got {type(center)}"
                )
        else:
            center = self.type_explicit.from_scalar(center)

        width = self.width
        if isinstance(width, na.AbstractArray):
            if width.type_abstract == self.type_abstract:
                pass
            elif isinstance(width, na.AbstractScalar):
                width = self.type_explicit.from_scalar(width)
            else:
                raise ValueError(
                    f"`start` must either be an instance of {float}, {u.Quantity}, {na.AbstractScalar},"
                    f" or {self.type_abstract}, got {type(width)}"
                )
        else:
            width = self.type_explicit.from_scalar(width)

        seed = self.seed

        result = self.type_explicit()
        components_center = center.components
        components_width = width.components

        for c in result.components:
            result.components[c] = na.ScalarNormalRandomSample(
                center=components_center[c],
                width=components_width[c],
                shape_random=self.shape_random,
                seed=seed,
            ).explicit
            seed += 1

        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedVectorArray(
    AbstractImplicitVectorArray,
    na.AbstractParameterizedArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorArrayRange(
    AbstractParameterizedVectorArray,
    na.AbstractArrayRange,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorSpace(
    AbstractParameterizedVectorArray,
    na.AbstractSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorLinearSpace(
    AbstractVectorSpace,
    na.AbstractLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorStratifiedRandomSpace(
    AbstractVectorLinearSpace,
    na.AbstractStratifiedRandomSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorLogarithmicSpace(
    AbstractVectorSpace,
    na.AbstractLogarithmicSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorGeometricSpace(
    AbstractVectorSpace,
    na.AbstractGeometricSpace,
):
    pass
