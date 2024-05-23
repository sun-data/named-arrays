from __future__ import annotations
from typing import ClassVar, Type, Sequence, Callable, Collection, Any, Generic, TypeVar
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    'VectorPrototypeT',
    'VectorTypeError',
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
    'AbstractWcsVector',
]

VectorPrototypeT = TypeVar("VectorPrototypeT", bound="AbstractVectorArray")
VectorStartT = TypeVar('VectorStartT', bound="float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray")
VectorStopT = TypeVar('VectorStopT', bound="float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray")


class VectorTypeError(TypeError):
    pass


def _prototype(*arrays: float | u.Quantity | na.AbstractArray) -> na.AbstractVectorArray:
    for array in arrays:
        if isinstance(array, na.AbstractVectorArray):
            return array

    raise VectorTypeError


def _normalize(
        a: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        prototype: VectorPrototypeT,
) -> VectorPrototypeT:
    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractVectorArray):
            if a.type_abstract == prototype.type_abstract:
                result = a
            else:
                raise VectorTypeError
        elif isinstance(a, na.AbstractScalar):
            result = prototype.type_explicit.from_scalar(a, like=prototype)
        else:
            raise VectorTypeError
    else:
        result = prototype.type_explicit.from_scalar(a, like=prototype)

    return result


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
    def matrix(self) -> na.AbstractMatrixArray:
        new_dict = {}
        for c in self.components:
            component = self.components[c]
            if isinstance(component, AbstractVectorArray):
                new_dict[c] = component.matrix
            elif isinstance(na.as_named_array(component), na.AbstractScalar):
                new_dict[c] = component
            else:
                raise NotImplementedError


        return self.type_matrix.from_components(new_dict)

    @property
    def cartesian_nd(self) -> na.AbstractCartesianNdVectorArray:
        """
        Convert any instance of :class:`AbstractVectorArray` to an instance of :class:`AbstractCartesianNdVectorArray`
        """
        components_new = dict()
        components = self.components
        for c in components:

            component = components[c]
            if isinstance(component, na.AbstractVectorArray):
                component2 = component.cartesian_nd.components
                for c2 in component2:
                    components_new[f"{c}_{c2}"] = component2[c2]
            else:
                components_new[c] = component

        return na.CartesianNdVectorArray(components_new)

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
        return self.cartesian_nd.components

    @property
    def value(self) -> na.AbstractExplicitVectorArray:
        components = self.components
        components = {c: na.value(components[c]) for c in components}
        return self.type_explicit.from_components(components)

    @property
    def prototype_vector(self) -> na.AbstractExplicitVectorArray:
        """
        Return vector of same type with all components zeroed.
        """
        return self.type_explicit.from_components(dict.fromkeys(self.components, 0))

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
        return self.type_explicit.from_components(
            {c: components[c].astype(dtype=dtype[c], **kwargs) for c in components})

    def to(
        self: Self,
        unit: u.UnitBase | dict[str, None | u.UnitBase],
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = [],
        copy: bool = True,
    ) -> AbstractExplicitVectorArray:
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
                components_result[c] = components_c.to(
                    unit=unit[c],
                    equivalencies=equivalencies,
                    copy=copy,
                )
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
                item = self.type_explicit.from_scalar(item_accumulated, like=self)
            elif isinstance(item, na.AbstractScalar):
                item = self.type_explicit.from_scalar(item, like=self)
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
                        item[ax] = self.type_explicit.from_scalar(item[ax], like=self)
                    else:
                        return NotImplemented
                elif isinstance(item[ax], (int, slice)):
                    item[ax] = self.type_explicit.from_scalar(item[ax], like=self)
                elif item[ax] is None:
                    item[ax] = self.type_explicit.from_scalar(item[ax], like=self)
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
        elif isinstance(array, na.AbstractArray):
            array = self.type_explicit.from_scalar(array, like=self)
        else:
            return NotImplemented
        return array._getitem(item)

    def __bool__(self: Self) -> bool:
        result = super().__bool__()
        components = self.components
        for c in components:
            component = bool(components[c])
            result = result and component
        return result

    def __array_matmul__(
            self: Self,
            x1: na.ArrayLike,
            x2: na.ArrayLike,
            out: tuple[None | na.AbstractExplicitArray] = (None,),
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

        if isinstance(x1, AbstractVectorArray):
            if isinstance(x2, na.AbstractVectorArray):
                components_x2 = x2.cartesian_nd.components
                components_x1 = x1.cartesian_nd.components

                if components_x1.keys() == components_x2.keys():
                    result = 0
                    for c in components_x1:
                        component_x1 = na.as_named_array(components_x1[c])
                        component_x2 = na.as_named_array(components_x2[c])
                        result = np.add(result, np.matmul(component_x1, component_x2), out=out)
                else:
                    result = NotImplemented
            elif isinstance(na.as_named_array(x2), na.ScalarArray):
                result = np.multiply(x1, x2, out=out, **kwargs)
            else:
                result = NotImplemented

        elif isinstance(na.as_named_array(x1), na.ScalarArray):
            if isinstance(x2, na.AbstractVectorArray):
                result = np.multiply(x1, x2, out=out, **kwargs)
            else:
                result = NotImplemented

        else:
            result = NotImplemented

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

        from . import vector_array_functions

        if func in vector_array_functions.SINGLE_ARG_FUNCTIONS:
            return vector_array_functions.array_functions_single_arg(func, *args, **kwargs)

        if func in vector_array_functions.ARRAY_CREATION_LIKE_FUNCTIONS:
            return vector_array_functions.array_function_array_creation_like(func, *args, **kwargs)

        if func in vector_array_functions.SEQUENCE_FUNCTIONS:
            return vector_array_functions.array_function_sequence(func, *args, **kwargs)

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

        if func in vector_array_functions.EMATH_FUNCTIONS:
            return vector_array_functions.array_function_emath(func, *args, **kwargs)

        if func in vector_array_functions.STACK_LIKE_FUNCTIONS:
            return vector_array_functions.array_function_stack_like(func, *args, **kwargs)

        if func in vector_array_functions.HANDLED_FUNCTIONS:
            return vector_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def __named_array_function__(self, func, *args, **kwargs):
        result = super().__named_array_function__(func, *args, **kwargs)
        if result is not NotImplemented:
            return result

        from . import vector_named_array_functions

        if func in vector_named_array_functions.ASARRAY_LIKE_FUNCTIONS:
            return vector_named_array_functions.asarray_like(func=func, *args, **kwargs)

        if func in vector_named_array_functions.RANDOM_FUNCTIONS:
            return vector_named_array_functions.random(func=func, *args, **kwargs)

        if func in vector_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS:
            return vector_named_array_functions.plt_plot_like(func, *args, **kwargs)

        if func in vector_named_array_functions.HANDLED_FUNCTIONS:
            return vector_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented


AbstractScalarOrVectorArray = na.AbstractScalar | AbstractVectorArray


@dataclasses.dataclass(eq=False, repr=False)
class AbstractExplicitVectorArray(
    AbstractVectorArray,
    na.AbstractExplicitArray,
):
    @classmethod
    def from_scalar_array(
            cls: Type[Self],
            a: None | float | u.Quantity | na.AbstractArray,
            like: None | AbstractExplicitVectorArray = None,
    ) -> AbstractExplicitVectorArray:

        self = super().from_scalar_array(a=a, like=like)

        components_self = dict()

        if like is None:
            for c in self.components:
                components_self[c] = a
        else:
            components_like = like.components
            for c in components_like:
                component_like = components_like[c]
                if isinstance(component_like, na.AbstractArray):
                    components_self[c] = component_like.from_scalar_array(a, like=component_like)
                else:
                    components_self[c] = a

        self.components = components_self

        return self

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
            like: None | AbstractExplicitVectorArray = None,
    ) -> AbstractExplicitVectorArray:
        """
        Convert a scalar (an instance of :class:`named_arrays.AbstractScalar`) into a vector.
        """

        if like is not None:
            return like.type_explicit.from_components({c: scalar for c in like.components})
        else:
            return NotImplemented

    @classmethod
    def from_cartesian_nd(
            cls: AbstractExplicitVectorArray,
            array: na.CartesianNdVectorArray,
            like: None | AbstractExplicitVectorArray = None,
    ) -> AbstractExplicitVectorArray:

        if like is None:
            components_new = array.components

        else:
            nd_components = array.components
            components_new = {}
            components = like.components
            for c in components:

                component = components[c]
                if isinstance(component, na.AbstractVectorArray):
                    nd_key_mod = f"{c}_"
                    sub_dict = {k[len(nd_key_mod):]: v for k, v in nd_components.items() if k.startswith(nd_key_mod)}
                    components_new[c] = component.type_explicit.from_cartesian_nd(
                        na.CartesianNdMatrixArray(sub_dict),
                        like=component
                    )
                else:
                    components_new[c] = nd_components[c]

        return cls.from_components(components_new)

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

    def __setitem__(
            self,
            item: dict[str, int | slice | AbstractScalarOrVectorArray] | AbstractScalarOrVectorArray,
            value: AbstractScalarOrVectorArray,
    ):
        components_self = self.components

        if isinstance(item, na.AbstractArray):
            if isinstance(item, na.AbstractVectorArray):
                if item.type_abstract == self.type_abstract:
                    components_item = item.components
                else:
                    raise TypeError(
                        f"if `item` is an instance of `{na.AbstractVectorArray.__name__}`, "
                        f"`item.type_abstract`, `{item.type_abstract}`, "
                        f"should be equal to `self.type_abstract`, `{self.type_abstract}`"
                    )
            elif isinstance(item, na.AbstractScalar):
                components_item = {c: item for c in components_self}
            else:
                raise TypeError(
                    f"if `item` is an instance of `{na.AbstractArray.__name__}`, "
                    f"it must be an instance of `{na.AbstractVectorArray.__name__}` "
                    f"or `{na.AbstractScalar.__name__}`, got `{type(item)}`"
                )

        elif isinstance(item, dict):

            components_item = {c: dict() for c in components_self}

            for axis in item:
                item_axis = item[axis]
                if isinstance(item_axis, na.AbstractArray):
                    if isinstance(item_axis, na.AbstractVectorArray):
                        if item_axis.type_abstract == self.type_abstract:
                            components_item_axis = item_axis.components
                            for c in components_item:
                                components_item[c][axis] = components_item_axis[c]
                        else:
                            raise TypeError(
                                f"if `item['{axis}']` is an instance of `{na.AbstractVectorArray.__name__}`, "
                                f"`item['{axis}'].type_abstract`, `{item_axis.type_abstract}`, "
                                f"should be equal to `self.type_abstract`, `{self.type_abstract}`"
                            )
                    elif isinstance(item_axis, na.AbstractScalar):
                        for c in components_item:
                            components_item[c][axis] = item_axis
                    else:
                        raise TypeError(
                            f"if `item['{axis}']` is an instance of `{na.AbstractArray.__name__}`, "
                            f"it must be an instance of `{na.AbstractVectorArray.__name__}` "
                            f"or `{na.AbstractScalar.__name__}`, got `{type(item_axis)}`"
                        )
                else:
                    for c in components_item:
                        components_item[c][axis] = item_axis

        else:
            raise TypeError(
                f"`item` must be an instance of `{na.AbstractArray.__name__}` or {dict.__name__}, "
                f"got `{type(item)}`"
            )

        if isinstance(value, na.AbstractArray):
            if isinstance(value, na.AbstractVectorArray):
                if value.type_abstract == self.type_abstract:
                    components_value = value.components
                else:
                    raise TypeError(
                        f"if `value` is an instance of `{na.AbstractVectorArray.__name__}`, "
                        f"`value.type_abstract`, `{value.type_abstract}`, "
                        f"must be equal to `self.type_abstract`, `{self.type_abstract}`"
                    )
            elif isinstance(value, na.AbstractScalar):
                components_value = {c: value for c in components_self}
            else:
                raise TypeError(
                    f"if `value` is an instance of `{na.AbstractArray.__name__}`, "
                    f"it must be an instance of `{na.AbstractVectorArray.__name__}` "
                    f"or `{na.AbstractScalar.__name__}`, got `{type(value)}`"
                )
        else:
            components_value = {c: value for c in components_self}

        for c in components_self:
            components_self[c][components_item[c]] = components_value[c]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitVectorArray(
    AbstractVectorArray,
    na.AbstractImplicitArray,
):
    @property
    def components(self) -> dict[str, na.ArrayLike]:
        return self.explicit.components

    def _attr_normalized(self, name: str) -> AbstractExplicitVectorArray:

        attr = getattr(self, name)

        if isinstance(attr, na.AbstractArray):
            if attr.type_abstract == self.type_abstract:
                result = attr
            elif isinstance(attr, na.AbstractScalar):
                result = self.type_explicit.from_scalar(attr)
            else:
                raise ValueError(
                    f"if `{name}` is an instance of `AbstractArray` it must be an instance of `{self.type_abstract}` "
                    f"or `AbstractScalar`, got {type(attr)}"
                )
        else:
            result = self.type_explicit.from_scalar(attr)

        return result


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
    na.AbstractUniformRandomSample[VectorStartT, VectorStopT],
):
    pass


CenterT = TypeVar('CenterT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray)
WidthT = TypeVar('WidthT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorNormalRandomSample(
    AbstractVectorRandomSample,
    na.AbstractNormalRandomSample[CenterT, WidthT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedVectorArray(
    AbstractImplicitVectorArray,
    na.AbstractParameterizedArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVectorArrayRange(
    AbstractParameterizedVectorArray,
    na.AbstractArrayRange[StartT, StopT],
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


@dataclasses.dataclass(eq=False, repr=False)
class AbstractWcsVector(
    AbstractImplicitVectorArray,
):
    @property
    @abc.abstractmethod
    def crval(self) -> AbstractVectorArray:
        """
        The reference point in world coordinates
        """

    @property
    @abc.abstractmethod
    def crpix(self) -> na.CartesianNdVectorArray:
        """
        The reference point in pixel coordinates
        """

    @property
    @abc.abstractmethod
    def cdelt(self) -> AbstractVectorArray:
        """
        The plate scale at the reference point
        """

    @property
    @abc.abstractmethod
    def pc(self) -> na.AbstractMatrixArray:
        """
        The transformation matrix between pixel coordinates and
        world coordinates
        """

    @property
    @abc.abstractmethod
    def shape_wcs(self) -> dict[str, int]:
        """
        The shape of the WCS components of the vector
        """

    @property
    @abc.abstractmethod
    def _components_explicit(self) -> dict[str, na.ArrayLike]:
        """
        The components of this vector that are not specified by the WCS parameters
        """

    @property
    def _components_wcs(self):
        crval = self.crval
        r = self.crpix
        s = self.cdelt
        m = self.pc
        shape_wcs = self.shape_wcs
        p = na.CartesianNdVectorArray(na.indices(shape_wcs)) - 0.5
        q = m @ (p - r)
        x = s * q + crval
        return x.components

    @property
    def explicit(self) -> na.AbstractExplicitArray:
        components = self._components_explicit | self._components_wcs
        return self.type_explicit.from_components(components)

