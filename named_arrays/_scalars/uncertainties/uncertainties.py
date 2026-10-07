from __future__ import annotations
from typing import Mapping, TYPE_CHECKING, TypeVar, Generic, ClassVar, Type, Sequence, Callable, Collection, Any, overload
from typing import Self

import abc
import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na
from named_arrays._core import _required, _array_function_handlers

__all__ = [
    "nominal",
    "UncertainScalarStartT",
    "UncertainScalarStopT",
    "UncertainScalarTypeError",
    "AbstractUncertainScalarArray",
    "UncertainScalarArray",
    "AbstractImplicitUncertainScalarArray",
    "UniformUncertainScalarArray",
    "NormalUncertainScalarArray",
    "AbstractUncertainScalarRandomSample",
    "UncertainScalarUniformRandomSample",
    "UncertainScalarNormalRandomSample",
    "UncertainScalarPoissionRandomSample",
    "AbstractParameterizedUncertainScalarArray",
    "AbstractUncertainScalarSpace",
    "UncertainScalarLinearSpace",
    "UncertainScalarStratifiedRandomSpace",
    "UncertainScalarLogarithmicSpace",
    "UncertainScalarGeometricSpace",
]

NominalArrayT = TypeVar(
    'NominalArrayT',
    bound=None | float | complex | np.ndarray | u.Quantity | na.AbstractScalarArray,
    covariant=True,
)
DistributionArrayT = TypeVar(
    'DistributionArrayT',
    bound=None | float | complex | np.ndarray | u.Quantity | na.AbstractScalarArray,
    covariant=True,
)
WidthT = TypeVar('WidthT', bound=int | float | np.ndarray | u.Quantity | na.AbstractScalarArray, covariant=True)
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


def _mask_union(
    mask: AbstractUncertainScalarArray,
) -> tuple[na.AbstractScalarArray, bool]:
    """
    Reduce an uncertain boolean mask to the elements selected by the nominal
    value or by any sample of the distribution.

    Parameters
    ----------
    mask
        An uncertain boolean array used to select elements of another array.

    Returns
    -------
    union
        :obj:`True` wherever the nominal value or any sample of `mask` is
        :obj:`True`.
    varies
        Whether the nominal value and the samples of `mask` disagree about
        which elements are selected.
    """
    axis = mask.axis_distribution

    mask_nominal = na.as_named_array(mask.nominal)
    mask_distribution = na.as_named_array(mask.distribution)

    varies = _varies(mask)

    if axis in mask_distribution.shape:
        mask_distribution = np.any(mask_distribution, axis=axis)

    union = mask_nominal | mask_distribution

    return union, varies


def _varies(item: Any) -> bool:
    """
    Whether an item used to index an array selects different elements in
    the nominal value and in the samples of the distribution.

    Parameters
    ----------
    item
        A boolean mask, an array of indices, or a :class:`dict` of indices
        along each axis.
    """
    if isinstance(item, dict):
        return any(_varies(item[axis]) for axis in item)
    if isinstance(item, AbstractUncertainScalarArray):
        item = item.explicit
        nominal = na.as_named_array(item.nominal)
        distribution = na.as_named_array(item.distribution)
        return bool(np.any(distribution != nominal))
    return False


def _certain(item: AbstractUncertainScalarArray) -> na.AbstractScalarArray:
    """
    The plain array which selects the same elements as an uncertain item,
    for indexing an array without a distribution.

    Parameters
    ----------
    item
        A boolean mask or an array of indices which selects the same
        elements in its nominal value and in every sample of its
        distribution.
    """
    if _varies(item):
        raise ValueError(
            "`item` selects different elements in the nominal value and in the samples of its "
            "distribution, which an array without a distribution cannot store, so convert "
            "this array to an uncertain array first, or index with `item.nominal` to select "
            "using only the nominal value."
        )
    item = item.explicit
    return na.broadcast_to(na.as_named_array(item.nominal), item.shape)


def _as_uncertain(
    array: Any,
    item: Any,
    value: Any,
) -> Any:
    """
    Prepare a part of a vector or function, such as a component, for an
    assignment which an array without a distribution cannot store.

    Parameters
    ----------
    array
        The part of the vector or function which is assigned to.
    item
        The item which selects the elements of `array` to assign to.
    value
        The value assigned to the selected elements.

    Returns
    -------
    An uncertain copy of `array` if it is a plain array and either `item`
    selects different elements in different samples or `value` is uncertain,
    and otherwise `array` itself.
    The copy shares no memory with `array`, so assigning to it never changes
    `array`.
    """
    if isinstance(array, na.ScalarArray):
        if _varies(item) or isinstance(value, AbstractUncertainScalarArray):
            return UncertainScalarArray(array.copy(), array.copy())
    return array


def _fill_unselected(
    dtype: np.dtype,
    dtype_other: np.dtype,
) -> None | bool | np.ndarray:
    """
    The value of the elements which a sample of an uncertain mask did not
    select, for the nominal value or the distribution of an array.

    Parameters
    ----------
    dtype
        The data type of the indexed nominal value or distribution.
    dtype_other
        The data type of the other one, so that an integer nominal value of
        a floating-point distribution can be filled with NaN.

    Returns
    -------
    NaN, or False for boolean arrays, or :obj:`None` if no value of `dtype`
    can mark an element as not selected.
    """
    if np.issubdtype(dtype, np.bool_):
        return False
    if np.issubdtype(dtype, np.inexact):
        return np.array(np.nan, dtype=dtype)
    if np.issubdtype(dtype, np.integer) and np.issubdtype(dtype_other, np.inexact):
        return np.array(np.nan, dtype=np.result_type(dtype, dtype_other))
    return None


def nominal(
    a: Any | na.UncertainScalarArray[NominalArrayT, DistributionArrayT],
) -> na.AbstractExplicitArray:
    """
    Isolate the `nominal` attribute of an uncertain array.
    If `a` is a nested object, such as a vector, or an arbitrarily-nested
    structure (:class:`dict`, :class:`list`, :class:`tuple`, or
    :mod:`dataclasses` instance), this function is recursively applied to all
    the attributes/elements of the nested object, leaving non-array values
    unchanged.

    Parameters
    ----------
    a
        An array to isolate the `nominal` attribute of.

    Examples
    --------
    If we define an uncertain scalar `x`,

    .. jupyter-execute::

        import named_arrays as na

        x = na.UniformUncertainScalarArray(5, 4, num_distribution=5).explicit
        x

    then we can isolate the `nominal` attribute of `x` using this function

    .. jupyter-execute::

        na.nominal(x)

    If we use this scalar to define a vector `a`,

    .. jupyter-execute::

        a = na.Cartesian2dVectorArray(x, 2)
        a

    then we can also isolate the `nominal` attribute of every component of `a`

    .. jupyter-execute::

        na.nominal(a)
    """
    if na.named_array_like(a):
        try:
            return na._named_array_function(
                func=nominal,
                a=a,
            )
        except TypeError:
            return a
    elif isinstance(a, dict):
        return {key: nominal(a[key]) for key in a}
    elif isinstance(a, list):
        return [nominal(a_i) for a_i in a]
    elif isinstance(a, tuple):
        return tuple(nominal(a_i) for a_i in a)
    elif dataclasses.is_dataclass(a):
        return dataclasses.replace(a, **{
            field.name: nominal(getattr(a, field.name))
            for field in dataclasses.fields(a)
            if field.init
        })
    else:
        return a


@functools.cache
def _array_function_dispatch() -> tuple[dict[Callable, Callable], dict[Callable, Callable]]:
    """
    The handlers of the :mod:`numpy` functions which uncertain arrays support,
    built on first use since
    :mod:`named_arrays._scalars.uncertainties.uncertainties_array_functions`
    imports this module.

    Returns
    -------
    A dictionary mapping each function in a category of
    :mod:`~named_arrays._scalars.uncertainties.uncertainties_array_functions`
    to the handler of that category, and the ``HANDLED_FUNCTIONS`` of that module.
    """
    from . import uncertainties_array_functions as f
    handlers = _array_function_handlers([
        (f.SINGLE_ARG_FUNCTIONS, f.array_functions_single_arg),
        (f.ARRAY_CREATION_LIKE_FUNCTIONS, f.array_function_array_creation_like),
        (f.SEQUENCE_FUNCTIONS, f.array_function_sequence),
        (f.DEFAULT_FUNCTIONS, f.array_function_default),
        (f.CUMULATIVE_REDUCE_FUNCTIONS, f.array_function_cumulative_reduce),
        (f.PERCENTILE_LIKE_FUNCTIONS, f.array_function_percentile_like),
        (f.ARG_REDUCE_FUNCTIONS, f.array_function_arg_reduce),
        (f.FFT_LIKE_FUNCTIONS, f.array_function_fft_like),
        (f.FFTN_LIKE_FUNCTIONS, f.array_function_fftn_like),
        (f.EMATH_FUNCTIONS, f.array_function_emath),
        (f.STACK_LIKE_FUNCTIONS, f.array_function_stack_like),
    ])
    return handlers, f.HANDLED_FUNCTIONS


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

    def to_value(
        self,
        unit: u.UnitBase,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = None,
    ) -> UncertainScalarArray:
        return UncertainScalarArray(
            nominal=na.as_named_array(self.nominal).to_value(
                unit=unit,
                equivalencies=equivalencies,
            ),
            distribution=self.distribution.to_value(unit),
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

    def matrix_inverse(
            self,
            axis_rows: str,
            axis_columns: str,
    ) -> UncertainScalarArray:
        """
        Compute the inverse of this array, treating it as a matrix with the
        given row and column axes.

        The nominal value and every sample of the distribution are inverted
        separately, since the distribution axis is independent of the axes of
        the matrix.

        Parameters
        ----------
        axis_rows
            The axis representing the rows of the matrix.
        axis_columns
            The axis representing the columns of the matrix.
        """

        shape = self.shape
        shape_base = {ax: shape[ax] for ax in (axis_rows, axis_columns)}

        nominal = na.broadcast_to(self.nominal, na.shape(self.nominal) | shape_base)
        distribution = na.broadcast_to(self.distribution, na.shape(self.distribution) | shape_base)

        return UncertainScalarArray(
            nominal=nominal.matrix_inverse(axis_rows=axis_rows, axis_columns=axis_columns),
            distribution=distribution.matrix_inverse(axis_rows=axis_rows, axis_columns=axis_columns),
        )

    def to_string_array(
        self,
        format_value: str = "%.2f",
        format_unit: str = "latex_inline",
        pad_unit: str = r"$\,$",
    ) -> UncertainScalarArray:
        kwargs = dict(
            format_value=format_value,
            format_unit=format_unit,
            pad_unit=pad_unit,
        )
        return self.type_explicit(
            nominal=na.as_named_array(self.nominal).to_string_array(**kwargs),
            distribution=na.as_named_array(self.distribution).to_string_array(**kwargs),
        )

    def _getitem(
            self,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        array = self.explicit
        shape_array_distribution = array.shape_distribution

        nominal = na.as_named_array(array.nominal)
        distribution = na.as_named_array(array.distribution)

        # An uncertain mask whose samples select different elements, which is
        # applied after every selected element has been gathered.
        mask_varying = None

        if isinstance(item, na.AbstractArray):
            item = item.explicit
            if isinstance(item, AbstractUncertainScalarArray):
                # Select every element chosen by the nominal value or by any
                # sample, and blank out each element in the realizations which
                # did not choose it, since a sample-dependent number of
                # elements has no fixed-shape representation otherwise.
                union, varies = _mask_union(item)
                if varies:
                    dtype_nominal = na.get_dtype(nominal)
                    dtype_distribution = na.get_dtype(distribution)
                    fill_nominal = _fill_unselected(dtype_nominal, dtype_distribution)
                    fill_distribution = _fill_unselected(dtype_distribution, dtype_nominal)
                    if fill_nominal is None or fill_distribution is None:
                        dtype = dtype_nominal if fill_nominal is None else dtype_distribution
                        raise ValueError(
                            "`item` selects different elements in the nominal value and in the "
                            "samples of its distribution, so the elements which a sample did not "
                            "select are filled with NaN, or with False for boolean arrays, which "
                            f"is not possible for {dtype=}. Use `numpy.where()` to combine arrays "
                            "elementwise instead, or index with `item.nominal` to select using "
                            "only the nominal value."
                        )
                    mask_varying = item
                item_nominal = item_distribution = union
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
                elif isinstance(item[ax], slice):
                    item_nominal[ax] = item_distribution[ax] = item[ax]
                elif np.issubdtype(type(item[ax]), np.integer):
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

        if mask_varying is not None:
            mask = mask_varying[union]
            result = UncertainScalarArray(
                nominal=np.where(mask.nominal, result.nominal, fill_nominal),
                distribution=np.where(mask.distribution, result.distribution, fill_distribution),
            )

        return result

    def _getitem_reversed(
            self,
            array: na.AbstractArray,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        if isinstance(array, AbstractUncertainScalarArray):
            pass
        elif isinstance(array, na.AbstractScalarArray):
            if isinstance(item, AbstractUncertainScalarArray):
                # An array without a distribution has the same values in every
                # sample, so it stays certain unless the mask selects
                # different elements in different samples.
                item = item.explicit
                union, varies = _mask_union(item)
                result = array[union]
                dtype = na.get_dtype(array)
                fill = _fill_unselected(dtype, dtype)
                if not varies or fill is None:
                    # Elements which a sample did not select are left in place
                    # where they cannot be filled, such as the integer
                    # components of a vector or the labels of a function's
                    # inputs. The arrays which this one accompanies mark those
                    # elements instead.
                    return result
                mask = item[union]
                return UncertainScalarArray(
                    nominal=np.where(mask.nominal, result, fill),
                    distribution=np.where(mask.distribution, result, fill),
                )
            axis = self.axis_distribution
            if isinstance(item, dict) and axis not in item and not _varies(item):
                # Indices whose samples all agree with their nominal value
                # select the same elements in every sample, so an array
                # without a distribution stays certain.
                return array[{
                    ax: _certain(item[ax]) if isinstance(item[ax], AbstractUncertainScalarArray) else item[ax]
                    for ax in item
                }]
            shape_distribution = array.shape
            if isinstance(item, dict) and axis in item:
                num_distribution = item[axis].distribution.max().ndarray + 1
                shape_distribution = na.broadcast_shapes(shape_distribution, {axis: num_distribution})
            elif axis in na.shape(self.distribution):
                # `self` is the uncertain part of `item` which could not be
                # applied to a plain array, such as indices which differ
                # between samples
                num_distribution = na.shape(self.distribution)[axis]
                shape_distribution = na.broadcast_shapes(shape_distribution, {axis: num_distribution})
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
            ufunc: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | UncertainScalarArray | tuple[UncertainScalarArray, ...]:

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

        nout = ufunc.nout

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

        result_nominal = getattr(ufunc, method)(*inputs_nominal, **kwargs_nominal, **kwargs)
        result_distribution = getattr(ufunc, method)(*inputs_distribution, **kwargs_distribution, **kwargs)

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

        handlers, handled = _array_function_dispatch()

        if func in handlers:
            return handlers[func](func, *args, **kwargs)

        if func in handled:
            return handled[func](*args, **kwargs)

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

        if func in uncertainties_named_array_functions.NDFILTER_FUNCTIONS:
            return uncertainties_named_array_functions.ndfilter(func, *args, **kwargs)

        if func in uncertainties_named_array_functions.HANDLED_FUNCTIONS:
            return uncertainties_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented


@dataclasses.dataclass(eq=False, repr=False)
class UncertainScalarArray(
    AbstractUncertainScalarArray,
    na.AbstractExplicitScalarArray,
    Generic[NominalArrayT, DistributionArrayT],
):
    """
    A scalar array with uncertainty, represented by a nominal value and a
    distribution of samples along the axis ``_distribution``.

    Each sample of the distribution is one possible realization of the array.
    Every operation is applied to the nominal value and to each realization
    separately, which propagates the uncertainty through a calculation.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na

        x = na.UncertainScalarArray(
            nominal=na.ScalarArray(np.array([-1, 0.1, 2]), axes="x"),
            distribution=na.ScalarArray(
                ndarray=np.array([
                    [-1.1, -0.9, -1.0],
                    [-0.2, 0.3, 0.1],
                    [2.1, 1.9, 2.2],
                ]),
                axes=("x", "_distribution"),
            ),
        )

    A mask computed from an uncertain array is uncertain too, and can select a
    different number of elements in each realization.
    Indexing with such a mask keeps every element selected by the nominal value
    or by any sample, and each realization which did not select an element
    holds NaN there.

    .. jupyter-execute::

        print(x[x > 0])

    Reductions which ignore NaN then give the same result as applying the mask
    to each realization separately.

    .. jupyter-execute::

        print(np.nansum(x[x > 0]))
        print(np.sum(x, where=x > 0))

    A realization which selects nothing holds only NaN, so reductions like
    :func:`numpy.nanmean` warn about an empty slice for it.
    Boolean arrays hold False instead of NaN, which leaves :func:`numpy.any`
    and :func:`numpy.sum` unaffected, but not :func:`numpy.all`, so use
    ``np.all(b, where=m)`` rather than ``np.all(b[m])``.
    Uncertain arrays which can hold neither, such as integer arrays, raise an
    error, as does :func:`numpy.nonzero`, since such a mask has no single set
    of indices.
    Arrays without a distribution are filled the same way, so they become
    uncertain, except that those which can hold neither, like the integer
    component of a vector or the labels of a function's inputs, keep every
    selected element and stay certain, since the uncertain arrays which they
    accompany mark the elements a realization did not select.
    If every sample of the mask agrees with its nominal value, nothing is
    filled and arrays without a distribution stay certain.
    To select using only the nominal value of the mask, index with
    ``x[(x > 0).nominal]``.

    Assigning through an uncertain mask only changes the elements each
    realization selected,

    .. jupyter-execute::

        y = x.copy()
        y[y < 0] = 0
        print(y)

    and the NaN values returned by indexing are never written back, so an update like
    ``y[m] = -y[m]`` changes each realization correctly.
    If the nominal value or the distribution cannot store the result, because
    it does not vary along an axis of the mask or the distribution has no
    sample axis, both are replaced by broadcasted copies, so other arrays
    which share them are left unchanged.
    An array without a distribution cannot store a selection which differs
    between samples, or an uncertain value, so assigning one to it raises an
    error, except that a vector or function replaces its components, inputs,
    or outputs which have no distribution with uncertain copies.
    The arrays they replace are never written to, so other references to
    them are left unchanged.

    Similarly, :func:`numpy.sort` sorts the nominal value and each sample
    separately, and :func:`numpy.argsort` returns indices which differ between
    samples, so that indexing with them gathers each sample in its own order.
    """

    nominal: NominalArrayT = 0
    """The nominal value of the array."""

    distribution: DistributionArrayT = 0
    """Distribution of possible values of the array."""

    # The operators declared on `AbstractArray` can only promise the widest
    # array type. The result of an operation is the explicit array of the
    # highest family involved, so the result is `Self` unless a higher family
    # absorbs it. Declarations only; the implementation is inherited.
    if TYPE_CHECKING:  # pragma: nocover

        @overload
        def __add__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __add__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __add__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __sub__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __sub__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __sub__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __floordiv__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __floordiv__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __floordiv__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __mod__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __mod__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __mod__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __pow__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __pow__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __pow__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __radd__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __radd__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __radd__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __rsub__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __rsub__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __rsub__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __rmul__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __rmul__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __rmul__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __rtruediv__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __rtruediv__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __rtruediv__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __rfloordiv__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __rfloordiv__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __rfloordiv__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __rmod__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __rmod__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __rmod__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __rpow__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __rpow__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __rpow__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __lt__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __lt__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __lt__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __le__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __le__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __le__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __gt__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __gt__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __gt__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __ge__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __ge__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __ge__(self, other: na.ArrayLike) -> Self: ...

        @overload
        def __mul__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __mul__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __mul__(self, other: na.ArrayLike | u.UnitBase) -> Self: ...

        @overload
        def __truediv__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __truediv__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __truediv__(self, other: na.ArrayLike | u.UnitBase) -> Self: ...

        @overload
        def __lshift__(self, other: na.AbstractFunctionArray) -> na.AbstractFunctionArray: ...

        @overload
        def __lshift__(self, other: na.AbstractVectorArray) -> na.AbstractVectorArray: ...

        @overload
        def __lshift__(self, other: na.ArrayLike | u.UnitBase) -> Self: ...

        def __neg__(self) -> Self: ...

        def __pos__(self) -> Self: ...

        def __abs__(self) -> Self: ...


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
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray,
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
                # Assign to every element chosen by the nominal value or by any
                # sample, keeping the current value in each realization which
                # did not choose it. This also accepts `value` in the form
                # returned by `self[item]`, whose unchosen elements are NaN.
                union, varies = _mask_union(item)
                if varies:
                    value = np.where(item[union], value, self[union])
                item_nominal = item_distribution = union
            elif isinstance(item, na.AbstractScalarArray):
                item_nominal = item_distribution = item
            else:
                raise TypeError(
                    f"if `item` is an instance of `{na.AbstractArray.__name__}`, "
                    f"it must be an instance of `{na.AbstractScalar.__name__}`, "
                    f"got `{type(item)}`"
                )

            axes_item = set(item.shape)
            items_distribution = [item_distribution]

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

            # An index array which varies along an axis of this array also
            # indexes along that axis.
            axes_item = set(item).union(*(na.shape(item[axis]) for axis in item))
            items_distribution = list(item_distribution.values())

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

        # The nominal value and the distribution need every axis which the
        # assignment indexes or the value varies along, and the distribution
        # needs a sample axis if the indices or the value differ between
        # samples, as they do where an uncertain mask varies.
        axis_distribution = self.axis_distribution
        axes = axes_item.union(na.shape(value_nominal), na.shape(value_distribution))
        shape = {ax: shape_self[ax] for ax in shape_self if ax in axes}
        shape_nominal = na.shape(self.nominal)
        shape_nominal_new = na.broadcast_shapes(shape_nominal, shape)
        for a in [value_distribution, *items_distribution]:
            shape_a = na.shape(a)
            if axis_distribution in shape_a:
                shape = na.broadcast_shapes(shape, {axis_distribution: shape_a[axis_distribution]})
        shape_distribution = na.shape(self.distribution)
        shape_distribution_new = na.broadcast_shapes(shape_distribution, shape)

        if shape_nominal_new != shape_nominal or shape_distribution_new != shape_distribution:
            # Replace both, rather than only the one which needs to grow, so
            # that an array which shares them keeps a consistent nominal value
            # and distribution.
            nominal = na.as_named_array(self.nominal)
            distribution = na.as_named_array(self.distribution)
            self.nominal = na.broadcast_to(nominal, shape_nominal_new).copy()
            self.distribution = na.broadcast_to(distribution, shape_distribution_new).copy()

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
    nominal: NominalArrayT = _required()
    width: WidthT = _required()
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
    nominal: NominalArrayT = _required()
    width: WidthT = _required()
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
    def volume_cell(self, axis: None | str | Sequence[str]) -> na.AbstractScalar:
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
    def volume_cell(self, axis: None | str | Sequence[str]) -> na.AbstractScalar:
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
