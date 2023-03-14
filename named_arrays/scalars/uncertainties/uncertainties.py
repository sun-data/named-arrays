from __future__ import annotations
from typing import TypeVar, Generic, ClassVar, Type, Sequence, Callable, Collection, Any
from typing_extensions import Self

import abc
import dataclasses
import numpy as np
import astropy.units as u

import named_arrays as na

__all__ = [
    'AbstractUncertainScalarArray',
    'UncertainScalarArray',
    'UniformUncertainScalarArray',
    'NormalUncertainScalarArray',
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

_axis_distribution_default = "_distribution"
_num_distribution_default = 11


@dataclasses.dataclass(eq=False, repr=False)
class AbstractUncertainScalarArray(
    na.AbstractScalar
):
    __named_array_priority__: ClassVar[int] = 10 * na.AbstractScalarArray.__named_array_priority__

    axis_distribution: ClassVar[str] = "_distribution"

    @property
    def type_array(self) -> Type[UncertainScalarArray]:
        return UncertainScalarArray

    @property
    def type_array_abstract(self) -> Type[AbstractUncertainScalarArray]:
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
    def unit(self: Self) -> None | u.Unit:
        return na.unit(self.nominal)

    @property
    @abc.abstractmethod
    def array(self) -> UncertainScalarArray[na.ScalarArray, na.ScalarArray]:
        pass

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

    def to(self, unit: u.UnitBase) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).to(unit),
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
        array = self.array
        shape_array = array.shape
        shape_array_distribution = array.shape_distribution

        nominal = na.as_named_array(array.nominal)
        distribution = na.as_named_array(array.distribution)

        if isinstance(item, na.AbstractArray):
            item = item.array
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
        if self.axis_distribution in distribution.shape:
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
                            f"`out` must be `None` or an instance of `{self.type_array}`, "
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

        if func in uncertainties_array_functions.STACK_LIKE_FUNCTIONS:
            return uncertainties_array_functions.array_function_stack_like(func, *args, **kwargs)

        if func in uncertainties_array_functions.HANDLED_FUNCTIONS:
            return uncertainties_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    @property
    def broadcasted(self) -> na.UncertainScalarArray:
        a = self.array
        return na.UncertainScalarArray(
            nominal=na.broadcast_to(a.nominal, a.shape),
            distribution=na.broadcast_to(a.distribution, a.shape_distribution),
        )


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarArray(
    AbstractUncertainScalarArray,
    na.AbstractExplicitArray,
    Generic[NominalArrayT, DistributionArrayT],
):
    nominal: NominalArrayT = dataclasses.MISSING
    distribution: DistributionArrayT = dataclasses.MISSING

    def __post_init__(self):
        if self.axis_distribution not in na.shape(self.distribution):
            raise ValueError(
                f"`axis_distribution`, '{self.axis_distribution}' not in `distribution` array with "
                f"shape {na.shape(self.distribution)}"
            )

    @classmethod
    def empty(
            cls: Type[Self],
            shape: dict[str, int],
            dtype: Type = float,
            axis_distribution: str = _axis_distribution_default,
            num_distribution: int = _num_distribution_default,
    ) -> Self:
        shape_distribution = shape | {axis_distribution: num_distribution}
        return UncertainScalarArray(
            nominal=na.ScalarArray.empty(shape=shape, dtype=dtype),
            distribution=na.ScalarArray.empty(shape=shape_distribution, dtype=dtype),
        )

    @classmethod
    def zeros(
            cls: Type[Self],
            shape: dict[str, int],
            dtype: Type = float,
            axis_distribution: str = _axis_distribution_default,
            num_distribution: int = _num_distribution_default,
    ) -> Self:
        shape_distribution = shape | {axis_distribution: num_distribution}
        return UncertainScalarArray(
            nominal=na.ScalarArray.zeros(shape=shape, dtype=dtype),
            distribution=na.ScalarArray.zeros(shape=shape_distribution, dtype=dtype),
        )

    @classmethod
    def ones(
            cls: Type[Self],
            shape: dict[str, int],
            dtype: Type = float,
            axis_distribution: str = _axis_distribution_default,
            num_distribution: int = _num_distribution_default,
    ) -> Self:
        shape_distribution = shape | {axis_distribution: num_distribution}
        return UncertainScalarArray(
            nominal=na.ScalarArray.ones(shape=shape, dtype=dtype),
            distribution=na.ScalarArray.ones(shape=shape_distribution, dtype=dtype),
        )

    @property
    def num_distribution(self: Self) -> int:
        return self.distribution.shape[self.axis_distribution]

    @property
    def axes(self: Self) -> tuple[str, ...]:
        return tuple(self.shape.keys())

    @property
    def shape(self) -> dict[str, int]:
        shape = self.shape_distribution
        shape.pop(self.axis_distribution)
        return shape

    @property
    def ndim(self: Self) -> int:
        return len(self.shape)

    @property
    def size(self: Self) -> int:
        return int(np.array(tuple(self.shape.values())).prod())

    @property
    def array(self) -> Self:
        return self.copy_shallow()

    @property
    def centers(self) -> Self:
        return self


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitUncertainScalarArray(
    AbstractUncertainScalarArray,
    na.AbstractImplicitArray,
):
    pass


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
    def array(self) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).array,
            distribution=na.as_named_array(self.distribution).array,
        )

    @property
    def centers(self) -> Self:
        return self


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
    def array(self) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).array,
            distribution=na.as_named_array(self.distribution).array,
        )

    @property
    def centers(self) -> Self:
        return self

