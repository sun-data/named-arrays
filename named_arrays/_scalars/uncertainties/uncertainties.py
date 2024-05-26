from __future__ import annotations
from typing import TypeVar, Generic, ClassVar, Type, Sequence, Callable, Collection, Any
from typing_extensions import Self

import abc
import dataclasses
import numpy as np
import astropy.units as u

import named_arrays as na

__all__ = [
    'UncertainScalarStartT',
    'UncertainScalarStopT',
    'UncertainScalarTypeError',
    'AbstractUncertainScalarArray',
    'UncertainScalarArray',
    'UniformUncertainScalarArray',
    'NormalUncertainScalarArray',
    'UncertainScalarUniformRandomSample',
    'UncertainScalarNormalRandomSample',
    'UncertainScalarPoissionRandomSample',
    'AbstractParameterizedUncertainScalarArray',
    'AbstractUncertainScalarSpace',
    'UncertainScalarLinearSpace',
    'UncertainScalarStratifiedRandomSpace',
    'UncertainScalarLogarithmicSpace',
    'UncertainScalarGeometricSpace',
]

NominalArrayT = TypeVar(
    'NominalArrayT',
    bound=None | float | complex | np.ndarray | u.Quantity | na.AbstractScalarArray,
)
DistributionArrayT = TypeVar(
    'DistributionArrayT',
    bound=None | float | complex | np.ndarray | u.Quantity | na.AbstractScalarArray,
)
WidthT = TypeVar('WidthT', bound=int | float | np.ndarray | u.Quantity | na.AbstractScalarArray)
UncertainScalarStartT = TypeVar("UncertainScalarStartT", bound=float | u.Quantity | na.AbstractScalar)
UncertainScalarStopT = TypeVar("UncertainScalarStopT", bound=float | u.Quantity | na.AbstractScalar)
UncertainScalarCenterT = TypeVar("UncertainScalarCenterT", bound=float | u.Quantity | na.AbstractScalar)
UncertainScalarWidthT = TypeVar("UncertainScalarWidthT", bound=float | u.Quantity | na.AbstractScalar)

_axis_distribution_default = "_distribution"
_num_distribution_default = 11


class UncertainScalarTypeError(TypeError):
    pass


def _normalize(a: float | u.Quantity | na.AbstractScalar):
    if isinstance(a, na.AbstractArray):
        if isinstance(a, na.AbstractScalar):
            if isinstance(a, na.AbstractUncertainScalarArray):
                result = a
            else:
                result = na.UncertainScalarArray(a, a)
        else:
            raise UncertainScalarTypeError
    else:
        result = na.UncertainScalarArray(a, a)

    return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractUncertainScalarArray(
    na.AbstractScalar
):
    __named_array_priority__: ClassVar[int] = 10 * na.AbstractScalarArray.__named_array_priority__

    axis_distribution: ClassVar[str] = "_distribution"

    @property
    def type_explicit(self) -> Type[UncertainScalarArray]:
        return UncertainScalarArray

    @property
    def type_abstract(self) -> Type[AbstractUncertainScalarArray]:
        return AbstractUncertainScalarArray

    @property
    @abc.abstractmethod
    def nominal(self) -> float | complex | u.Quantity | na.AbstractScalarArray:
        """
        Nominal value of the array.
        """

    @property
    @abc.abstractmethod
    def distribution(self) -> na.AbstractScalarArray:
        """
        Distribution of possible values of the array.
        """

    @property
    @abc.abstractmethod
    def num_distribution(self) -> int:
        """
        Number samples along :attr:`axis_distribution`.
        """

    @property
    def shape_distribution(self) -> dict[str, int]:
        return na.shape_broadcasted(self.nominal, self.distribution)

    @property
    def dtype(self) -> np.dtype:
        return np.promote_types(
            na.get_dtype(self.nominal),
            na.get_dtype(self.distribution),
        )

    @property
    def value(self) -> UncertainScalarArray:
        return self.type_explicit(
            nominal=na.value(self.nominal),
            distribution=na.value(self.distribution),
        )

    def astype(
            self,
            dtype: str | np.dtype | Type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            ),
            distribution=self.distribution.astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            ),
        )

    def to(
        self,
        unit: u.UnitBase,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = None,
        copy: bool = True,
    ) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).to(
                unit=unit,
                equivalencies=equivalencies,
                copy=copy,
            ),
            distribution=self.distribution.to(unit),
        )

    def add_axes(self, axes: str | Sequence[str]) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).add_axes(axes),
            distribution=self.distribution.add_axes(axes),
        )

    def combine_axes(
            self,
            axes: None | Sequence[str] = None,
            axis_new: None | str =None,
    ) -> UncertainScalarArray:

        shape = self.shape

        if axes is None:
            axes = tuple(self.shape)

        shape_base = {ax: shape[ax] for ax in shape if ax in axes}

        nominal = na.broadcast_to(self.nominal, na.shape(self.nominal) | shape_base)
        distribution = na.broadcast_to(self.distribution, na.shape(self.distribution) | shape_base)

        return UncertainScalarArray(
            nominal=nominal.combine_axes(axes=axes, axis_new=axis_new),
            distribution=distribution.combine_axes(axes=axes, axis_new=axis_new),
        )

    def _getitem(
            self,
            item: dict[str, int | slice | na.AbstractScalar] | na.AbstractScalar,
    ):
        array = self.explicit
        shape_array = array.shape
        shape_array_distribution = array.shape_distribution

        nominal = na.as_named_array(array.nominal)
        distribution = na.as_named_array(array.distribution)

        if isinstance(item, na.AbstractArray):
            item = item.explicit
            if isinstance(item, AbstractUncertainScalarArray):
                item_nominal = item_distribution = item.nominal & np.all(item.distribution, axis=self.axis_distribution)
            elif isinstance(item, na.AbstractScalarArray):
                item_nominal = item_distribution = item
            else:
                return NotImplemented

            shape_item = na.broadcast_shapes(item_nominal.shape, item_distribution.shape)

            if not set(shape_item).issubset(shape_array_distribution):
                raise ValueError(
                    f"the axes in item, {tuple(shape_item)}, must be a subset of the axes in array, "
                    f"{tuple(shape_array_distribution)}"
                )

            if not all(shape_item[ax] == shape_array_distribution[ax] for ax in shape_item):
                raise ValueError(
                    f"the shape of item, {shape_item}, must be consistent with the shape of the array, "
                    f"{shape_array_distribution}"
                )

            shape_nominal = na.broadcast_shapes(nominal.shape, item_nominal.shape)
            shape_distribution = na.broadcast_shapes(distribution.shape, item_distribution.shape)

            nominal = na.broadcast_to(nominal, shape_nominal)
            distribution = na.broadcast_to(distribution, shape_distribution)

        elif isinstance(item, dict):

            if not set(item).issubset(shape_array_distribution):
                raise ValueError(f"the axes in item, {tuple(item)}, must be a subset of the axes in the array, {array.axes}")

            item_nominal = dict()
            item_distribution = dict()
            for ax in item:
                if isinstance(item[ax], na.AbstractArray):
                    if isinstance(item[ax], AbstractUncertainScalarArray):
                        item_nominal[ax] = item[ax].nominal
                        item_distribution[ax] = item[ax].distribution
                    elif isinstance(item[ax], na.AbstractScalarArray):
                        item_nominal[ax] = item_distribution[ax] = item[ax]
                    else:
                        return NotImplemented
                elif isinstance(item[ax], (int, slice)):
                    item_nominal[ax] = item_distribution[ax] = item[ax]
                else:
                    return NotImplemented

                if ax not in nominal.axes:
                    item_nominal.pop(ax)
                if ax not in distribution.axes:
                    item_distribution.pop(ax)

        else:
            return NotImplemented

        result = UncertainScalarArray(
            nominal=nominal[item_nominal],
            distribution=distribution[item_distribution],
        )

        return result

    def _getitem_reversed(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | AbstractUncertainScalarArray] | AbstractUncertainScalarArray
    ):
        if isinstance(array, AbstractUncertainScalarArray):
            pass
        elif isinstance(array, na.AbstractScalarArray):
            if isinstance(item, dict):
                num_distribution = item[self.axis_distribution].distribution.max().ndarray + 1
            else:
                num_distribution = item.num_distribution
            shape_distribution = na.broadcast_shapes(array.shape, {self.axis_distribution: num_distribution})
            array = UncertainScalarArray(
                nominal=array,
                distribution=array.broadcast_to(shape_distribution),
            )
        else:
            return NotImplemented

        return array._getitem(item)

    def __bool__(self):
        result = super().__bool__()
        nominal = bool(self.nominal)
        distribution = self.distribution
        if self.axis_distribution in na.shape(distribution):
            distribution = np.all(distribution, axis=self.axis_distribution)
        distribution = bool(distribution)
        return result and nominal and distribution

    def __mul__(self, other: na.ArrayLike | u.Unit) -> UncertainScalarArray:
        if isinstance(other, u.UnitBase):
            return UncertainScalarArray(
                nominal=self.nominal * other,
                distribution=self.distribution * other,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self, other: na.ArrayLike | u.Unit) -> UncertainScalarArray:
        if isinstance(other, u.UnitBase):
            return UncertainScalarArray(
                nominal=self.nominal << other,
                distribution=self.distribution << other,
            )
        else:
            return super().__lshift__(other)

    def __truediv__(self, other: na.ArrayLike | u.Unit) -> UncertainScalarArray:
        if isinstance(other, u.UnitBase):
            return UncertainScalarArray(
                nominal=self.nominal / other,
                distribution=self.distribution / other,
            )
        else:
            return super().__truediv__(other)

    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | UncertainScalarArray | tuple[UncertainScalarArray, ...]:

        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

        nout = function.nout

        inputs_nominal = []
        inputs_distribution = []
        for inp in inputs:
            if isinstance(inp, na.AbstractArray):
                if isinstance(inp, AbstractUncertainScalarArray):
                    inp_nominal = inp.nominal
                    inp_distribution = inp.distribution
                elif isinstance(inp, na.AbstractScalarArray):
                    inp_nominal = inp_distribution = inp
                else:
                    return NotImplemented
            else:
                inp_nominal = inp_distribution = inp
            inputs_nominal.append(inp_nominal)
            inputs_distribution.append(inp_distribution)

        kwargs_nominal = dict()
        kwargs_distribution = dict()

        if "where" in kwargs:
            where = kwargs.pop("where")
            if isinstance(where, na.AbstractArray):
                if isinstance(where, na.AbstractScalar):
                    if isinstance(where, AbstractUncertainScalarArray):
                        where_nominal = where.nominal
                        where_distribution = where.distribution
                    else:
                        where_nominal = where_distribution = where
                else:
                    return NotImplemented
            else:
                where_nominal = where_distribution = where
            kwargs_nominal["where"] = where_nominal
            kwargs_distribution["where"] = where_distribution

        if "out" in kwargs:
            out = kwargs.pop("out")
            out_nominal = list()
            out_distribution = list()
            for o in out:
                if o is not None:
                    if isinstance(o, UncertainScalarArray):
                        types = (np.ndarray, na.AbstractArray)
                        o_nominal = o.nominal if isinstance(o.nominal, types) else None
                        o_distribution = o.distribution if isinstance(o.distribution, types) else None
                    else:
                        raise ValueError(
                            f"`out` must be `None` or an instance of `{self.type_explicit}`, "
                            f"got {tuple(type(x) for x in out)}"
                        )
                else:
                    o_nominal = o_distribution = None
                out_nominal.append(o_nominal)
                out_distribution.append(o_distribution)
            if nout == 1:
                out_nominal = out_nominal[0]
                out_distribution = out_distribution[0]
            else:
                out_nominal = tuple(out_nominal)
                out_distribution = tuple(out_distribution)
            kwargs_nominal["out"] = out_nominal
            kwargs_distribution["out"] = out_distribution
        else:
            out = (None, ) * nout

        result_nominal = getattr(function, method)(*inputs_nominal, **kwargs_nominal, **kwargs)
        result_distribution = getattr(function, method)(*inputs_distribution, **kwargs_distribution, **kwargs)

        if nout == 1:
            result_nominal = (result_nominal, )
            result_distribution = (result_distribution, )

        result = list(
            UncertainScalarArray(result_nominal[i], result_distribution[i])
            for i in range(nout)
        )

        for i in range(nout):
            if out[i] is not None:
                out[i].nominal = result[i].nominal
                out[i].distribution = result[i].distribution
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

        from . import uncertainties_array_functions

        if func in uncertainties_array_functions.SINGLE_ARG_FUNCTIONS:
            return uncertainties_array_functions.array_functions_single_arg(func, *args, **kwargs)

        if func in uncertainties_array_functions.ARRAY_CREATION_LIKE_FUNCTIONS:
            return uncertainties_array_functions.array_function_array_creation_like(func, *args, **kwargs)

        if func in uncertainties_array_functions.SEQUENCE_FUNCTIONS:
            return uncertainties_array_functions.array_function_sequence(func, *args, **kwargs)

        if func in uncertainties_array_functions.DEFAULT_FUNCTIONS:
            return uncertainties_array_functions.array_function_default(func, *args, **kwargs)

        if func in uncertainties_array_functions.PERCENTILE_LIKE_FUNCTIONS:
            return uncertainties_array_functions.array_function_percentile_like(func, *args, **kwargs)

        if func in uncertainties_array_functions.ARG_REDUCE_FUNCTIONS:
            return uncertainties_array_functions.array_function_arg_reduce(func, *args, **kwargs)

        if func in uncertainties_array_functions.FFT_LIKE_FUNCTIONS:
            return uncertainties_array_functions.array_function_fft_like(func, *args, **kwargs)

        if func in uncertainties_array_functions.FFTN_LIKE_FUNCTIONS:
            return uncertainties_array_functions.array_function_fftn_like(func, *args, **kwargs)

        if func in uncertainties_array_functions.EMATH_FUNCTIONS:
            return uncertainties_array_functions.array_function_emath(func, *args, **kwargs)

        if func in uncertainties_array_functions.STACK_LIKE_FUNCTIONS:
            return uncertainties_array_functions.array_function_stack_like(func, *args, **kwargs)

        if func in uncertainties_array_functions.HANDLED_FUNCTIONS:
            return uncertainties_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def __named_array_function__(self, func, *args, **kwargs):
        result = super().__named_array_function__(func, *args, **kwargs)
        if result is not NotImplemented:
            return result

        from . import uncertainties_named_array_functions

        if func in uncertainties_named_array_functions.ASARRAY_LIKE_FUNCTIONS:
            return uncertainties_named_array_functions.asarray_like(func=func, *args, **kwargs)

        if func in uncertainties_named_array_functions.RANDOM_FUNCTIONS:
            return uncertainties_named_array_functions.random(func=func, *args, **kwargs)

        if func in uncertainties_named_array_functions.PLT_PLOT_LIKE_FUNCTIONS:
            return uncertainties_named_array_functions.plt_plot_like(func, *args, **kwargs)

        if func in uncertainties_named_array_functions.HANDLED_FUNCTIONS:
            return uncertainties_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarArray(
    AbstractUncertainScalarArray,
    na.AbstractExplicitArray,
    Generic[NominalArrayT, DistributionArrayT],
):
    nominal: NominalArrayT = 0
    distribution: DistributionArrayT = 0

    def __post_init__(self):
        if self.axis_distribution in na.shape(self.nominal):
            raise ValueError(
                f"`axis_distribution`, '{self.axis_distribution}' should not be in `nominal` array with "
                f"shape {na.shape(self.nominal)}"
            )

    @classmethod
    def from_scalar_array(
            cls: type[Self],
            a: float | u.Quantity | na.AbstractScalarArray,
            like: None | Self = None,
    ) -> Self:

        self = super().from_scalar_array(a=a, like=like)

        if isinstance(a, na.AbstractArray):
            if not isinstance(a, na.AbstractScalarArray):
                raise TypeError(
                    f"If `a` is an instance of `{na.AbstractArray.__name__}`, it must be an instance of "
                    f"`{na.AbstractScalarArray.__name__}`, got `{type(a).__name__}`."
                )

        if like is None:
            self.nominal = a
            self.distribution = a
        else:
            if isinstance(like.nominal, na.AbstractArray):
                self.nominal = like.nominal.from_scalar_array(a=a, like=like.nominal)
            else:
                self.nominal = a

            if isinstance(like.distribution, na.AbstractArray):
                self.distribution = like.distribution.from_scalar_array(a=a, like=like.distribution)
            else:
                self.distribution = a

        return self

    @property
    def num_distribution(self: Self) -> int:
        return self.distribution.shape[self.axis_distribution]

    @property
    def axes(self: Self) -> tuple[str, ...]:
        return tuple(self.shape.keys())

    @property
    def shape(self) -> dict[str, int]:
        shape = self.shape_distribution
        if self.axis_distribution in shape:
            shape.pop(self.axis_distribution)
        return shape

    @property
    def ndim(self: Self) -> int:
        return len(self.shape)

    @property
    def size(self: Self) -> int:
        return int(np.array(tuple(self.shape.values())).prod())

    @property
    def explicit(self) -> Self:
        return self.copy_shallow()

    def __setitem__(
            self,
            item: dict[str, int | slice | na.AbstractScalar] | na.AbstractScalar,
            value: int | float | u.Quantity | na.AbstractScalar,
    ):
        shape_self = self.shape

        if isinstance(item, na.AbstractArray):

            item = item.explicit
            if not set(item.shape).issubset(shape_self):
                raise ValueError(
                    f"if `item` is an instance of `{na.AbstractArray.__name__}`, "
                    f"`item.axes`, {item.axes}, should be a subset of `self.axes`, {self.axes}"
                )

            if isinstance(item, na.AbstractUncertainScalarArray):
                item_nominal = item_distribution = item.nominal & np.all(item.distribution, axis=self.axis_distribution)
            elif isinstance(item, na.AbstractScalarArray):
                item_nominal = item_distribution = item
            else:
                raise TypeError(
                    f"if `item` is an instance of `{na.AbstractArray.__name__}`, "
                    f"it must be an instance of `{na.AbstractScalar.__name__}`, "
                    f"got `{type(item)}`"
                )
        elif isinstance(item, dict):

            if not set(item).issubset(shape_self):
                raise ValueError(
                    f"if `item` is a `{dict.__name__}`, the keys in `item`, {tuple(item)}, "
                    f"must be a subset of `self.axes`, {self.axes}"
                )

            item_nominal = dict()
            item_distribution = dict()
            for axis in item:
                item_axis = item[axis]
                if isinstance(item_axis, na.AbstractArray):
                    if isinstance(item_axis, na.AbstractUncertainScalarArray):
                        item_nominal[axis] = item_axis.nominal
                        item_distribution[axis] = item_axis.distribution
                    elif isinstance(item_axis, na.AbstractScalarArray):
                        item_nominal[axis] = item_distribution[axis] = item_axis
                    else:
                        raise TypeError(
                            f"if a value in `item` is an instance of `{na.AbstractArray.__name__}`, "
                            f"it must be an instance of `{na.AbstractScalar.__name__}`, "
                            f"got `{type(item_axis)}`"
                        )
                else:
                    item_nominal[axis] = item_distribution[axis] = item_axis

        else:
            raise TypeError(
                f"`item` must be an instance of `{na.AbstractArray.__name__}` or {dict.__name__}, "
                f"got `{type(item)}`"
            )

        if isinstance(value, na.AbstractArray):
            if isinstance(value, na.AbstractUncertainScalarArray):
                value_nominal = value.nominal
                value_distribution = value.distribution
            elif isinstance(value, na.AbstractScalarArray):
                value_nominal = value_distribution = value
            else:
                raise TypeError(
                    f"if `value` is an instance of `{na.AbstractArray.__name__}`, "
                    f"it must be an instance of `{na.AbstractScalar.__name__}`, "
                    f"got {type(value)}"
                )
        else:
            value_nominal = value_distribution = value

        self.nominal[item_nominal] = value_nominal
        self.distribution[item_distribution] = value_distribution


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitUncertainScalarArray(
    AbstractUncertainScalarArray,
    na.AbstractImplicitArray,
):

    def _attr_normalized(self, name: str) -> UncertainScalarArray:

        attr = getattr(self, name)

        if isinstance(attr, na.AbstractArray):
            if isinstance(attr, na.AbstractScalar):
                if isinstance(attr, na.AbstractUncertainScalarArray):
                    result = attr
                else:
                    result = UncertainScalarArray(attr, attr)
            else:
                raise TypeError(
                    f"if `{name}` is an instance of `AbstractArray`, it must be an instance of `AbstractScalar`, "
                    f"got {type(attr)}"
                )
        else:
            result = UncertainScalarArray(attr, attr)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class UniformUncertainScalarArray(
    AbstractImplicitUncertainScalarArray,
    na.AbstractRandomMixin,
    Generic[NominalArrayT, WidthT],
):
    nominal: NominalArrayT = dataclasses.MISSING
    width: WidthT = dataclasses.MISSING
    num_distribution: int = _num_distribution_default
    seed: None | int = None

    @property
    def distribution(self: Self) -> na.ScalarUniformRandomSample:
        return na.ScalarUniformRandomSample(
            start=self.nominal - self.width,
            stop=self.nominal + self.width,
            shape_random={self.axis_distribution: self.num_distribution},
            seed=self.seed,
        )

    @property
    def explicit(self) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.explicit(self.nominal),
            distribution=na.explicit(self.distribution),
        )


@dataclasses.dataclass(eq=False, repr=False)
class NormalUncertainScalarArray(
    AbstractImplicitUncertainScalarArray,
    na.AbstractRandomMixin,
    Generic[NominalArrayT, WidthT],
):
    nominal: NominalArrayT = dataclasses.MISSING
    width: WidthT = dataclasses.MISSING
    num_distribution: int = _num_distribution_default
    seed: None | int = None

    @property
    def distribution(self: Self) -> na.ScalarNormalRandomSample:
        return na.ScalarNormalRandomSample(
            center=self.nominal,
            width=self.width,
            shape_random={self.axis_distribution: self.num_distribution},
            seed=self.seed,
        )

    @property
    def explicit(self) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.explicit(self.nominal),
            distribution=na.explicit(self.distribution),
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractUncertainScalarRandomSample(
    AbstractImplicitUncertainScalarArray,
    na.AbstractRandomSample,
):
    @property
    def nominal(self) -> float | u.Quantity | na.AbstractScalarArray:
        return self.explicit.nominal

    @property
    def distribution(self) -> na.AbstractScalarArray:
        return self.explicit.distribution

    @property
    def num_distribution(self) -> int:
        return self.explicit.num_distribution


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarUniformRandomSample(
    AbstractUncertainScalarRandomSample,
    na.AbstractUniformRandomSample[UncertainScalarStartT, UncertainScalarStopT],
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
class UncertainScalarNormalRandomSample(
    AbstractUncertainScalarRandomSample,
    na.AbstractNormalRandomSample[UncertainScalarCenterT, UncertainScalarWidthT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarPoissionRandomSample(
    AbstractUncertainScalarRandomSample,
    na.AbstractPoissonRandomSample[UncertainScalarCenterT],
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedUncertainScalarArray(
    AbstractImplicitUncertainScalarArray,
    na.AbstractParameterizedArray,
):
    @property
    def nominal(self) -> float | u.Quantity | na.AbstractScalarArray:
        return self.explicit.nominal

    @property
    def distribution(self) -> na.AbstractScalarArray:
        return self.explicit.distribution

    @property
    def num_distribution(self) -> int:
        return self.explicit.num_distribution


@dataclasses.dataclass(eq=False, repr=False)
class AbstractUncertainScalarSpace(
    AbstractParameterizedUncertainScalarArray,
    na.AbstractSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarLinearSpace(
    AbstractUncertainScalarSpace,
    na.AbstractLinearSpace,
):
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


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarStratifiedRandomSpace(
    UncertainScalarLinearSpace,
    na.AbstractStratifiedRandomSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarLogarithmicSpace(
    AbstractUncertainScalarSpace,
    na.AbstractLogarithmicSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarGeometricSpace(
    AbstractUncertainScalarSpace,
    na.AbstractGeometricSpace,
):
    pass
