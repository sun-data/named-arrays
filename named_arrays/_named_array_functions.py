from __future__ import annotations
from typing import Sequence, overload, Type, Any, Callable, TypeVar
import functools
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    '_named_array_function',
    'asarray',
    'asanyarray',
    'arange',
    'linspace',
    'logspace',
    'geomspace',
    'ndim',
    'shape',
    'unit',
    'unit_normalized',
    'broadcast_to',
    'stack',
    'concatenate',
    'add_axes',
    "interp",
    'jacobian',
]

ArrayT = TypeVar("ArrayT")
LikeT = TypeVar("LikeT", bound="None | na.AbstractArray")
AxisT = TypeVar("AxisT", bound="str | na.AbstractArray")
NumT = TypeVar("NumT", bound="int | na.AbstractArray")
BaseT = TypeVar("BaseT", bound="int | na.AbstractArray")
InputT = TypeVar("InputT", bound="float | u.Quantity | na.AbstractScalarArray")
OutputT = TypeVar("OutputT", bound="float | u.Quantity | na.AbstractScalarArray")


def _is_subclass(a: Any, b: Any):
    if type(a) == type(b):
        return 0
    elif isinstance(a, type(b)):
        return 1
    elif isinstance(b, type(a)):
        return -1
    else:
        return 0


def _named_array_function(func: Callable, *args, **kwargs):
    arrays_args = tuple(arg for arg in args if isinstance(arg, na.AbstractArray))
    arrays_kwargs = tuple(kwargs[k] for k in kwargs if isinstance(kwargs[k], na.AbstractArray))
    arrays = arrays_args + arrays_kwargs

    arrays = sorted(arrays, key=functools.cmp_to_key(_is_subclass))

    for array in arrays:
        res = array.__named_array_function__(func, *args, **kwargs)
        if res is not NotImplemented:
            return res

    raise TypeError(
        f"all types, {[type(a).__qualname__ for a in arrays]}, returned `NotImplemented` "
        f"for function `{func.__name__}`"
    )


def _asarray_like(
        func: Callable,
        a: ArrayT,
        dtype: None | type | np.dtype | str = None,
        order: None | str = None,
        *,
        like: None | LikeT = None,
) -> ArrayT | LikeT:

    if a is None:
        return None

    if like is None:
        like = na.ScalarArray(None)
    elif not na.named_array_like(like):
        like = func(like)

    return _named_array_function(
        func=func,
        a=a,
        dtype=dtype,
        order=order,
        like=like,
    )


@overload
def asarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: None = ...
) -> ArrayT:
    ...


@overload
def asarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: LikeT = ...
) -> LikeT:
    ...


def asarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = None,
        order: None | str = None,
        *,
        like: None | LikeT = None,
) -> ArrayT | LikeT:
    """
    Converts the input to use only instances of :class:`numpy.ndarray` as the underlying data.

    This function does not convert an instance of :class:`named_arrays.AbstractArray` to an instance of
    :class:`numpy.ndarray` like you might expect from the documentation of :func:`numpy.asarray`.
    Instead, it recursively inspects the input, converting instances of :class:`named_arrays.AbstractImplicitArray`
    to `named_arrays.AbstractExplicitArray`, and calling :func:`numpy.asarray` on the underlying data.

    Parameters
    ----------
    a
        Input array to be converted
    dtype
        Data type of output, usually inferred from the input.
    order
        Memory layout. See the documentation of :func:`numpy.asarray` for more information.
    like
        Optional reference object.
        If provided, the result will be defined by this object.

    Returns
    -------
    out
        Standardized interpretation of ``a``, with all the underlying data expressed as instances of
        :class:`numpy.ndarray`.

    Examples
    --------

    Standardize a :class:`float`

    .. jupyter-execute::

        import named_arrays as na

        na.asarray(2)

    Standardize a :class:`named_arrays.ScalarArray` of :class:`float`

    .. jupyter-execute::

        na.asarray(na.ScalarArray(2))

    Standardize a :class:`named_arrays.Cartesian2dVectorArray` of :class:`float`

    .. jupyter-execute::

        na.asarray(na.Cartesian2dVectorArray(2, 3))

    See Also
    --------
    :func:`numpy.asarray` : Equivalent Numpy function
    :func:`named_arrays.asanyarray` : Similar to this function, but allows `numpy.ndarray` subclasses to pass through.
    """
    return _asarray_like(
        func=asarray,
        a=a,
        dtype=dtype,
        order=order,
        like=like,
    )


@overload
def asanyarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: None = ...
) -> ArrayT:
    ...


@overload
def asanyarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: LikeT = ...
) -> LikeT:
    ...


def asanyarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = None,
        order: None | str = None,
        *,
        like: None | LikeT = None,
) -> ArrayT | LikeT:
    """
    Converts the input to use only instances of :class:`numpy.ndarray` subclasses as the underlying data.

    This function does not convert an instance of :class:`named_arrays.AbstractArray` to an instance of a
    :class:`numpy.ndarray` subclass like you might expect from the documentation of :func:`numpy.asanyarray`.
    Instead, it recursively inspects the input, converting instances of :class:`named_arrays.AbstractImplicitArray`
    to `named_arrays.AbstractExplicitArray`, and calling :func:`numpy.asanyarray` on the underlying data.

    Parameters
    ----------
    a
        Input array to be converted
    dtype
        Data type of output, usually inferred from the input.
    order
        Memory layout. See the documentation of :func:`numpy.asanyarray` for more information.
    like
        Optional reference object.
        If provided, the result will be defined by this object.

    Returns
    -------
    out
        Standardized interpretation of ``a``, with all the underlying data expressed as instances of
        :class:`numpy.ndarray` subclasses.

    Examples
    --------

    Standardize an instance of :class:`astropy.units.Quantity`

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na

        na.asanyarray(2 * u.mm)

    See Also
    --------
    :func:`numpy.asanyarray` : Equivalent Numpy function
    :func:`named_arrays.asarray`: Similar to this function but converts instances of :class:`numpy.ndarray` subclasses
        back to instances of :class:`numpy.ndarray`.
    """

    return _asarray_like(
        func=asanyarray,
        a=a,
        dtype=dtype,
        order=order,
        like=like,
    )


def arange(
        start: float | complex | u.Quantity | na.AbstractArray,
        stop: float | complex | u.Quantity | na.AbstractArray,
        axis: str | na.AbstractArray,
        step: int | na.AbstractArray = 1,
) -> na.AbstractExplicitArray:
    """
    Redefined version of :func:`numpy.arange` with an ``axis`` parameter.

    Parameters
    ----------
    start
        starting value of the sequence
    stop
        ending value of the sequence
    axis
        name of the new sequence axis
    step
        step size between consecutive elements of the sequence

    See Also
    --------
    :func:`numpy.arange` : Equivalent numpy function

    :class:`named_arrays.ScalarArrayRange` : Implicit array version of this function for scalars
    """
    return _named_array_function(
        func=arange,
        start=na.as_named_array(start),
        stop=na.as_named_array(stop),
        axis=axis,
        step=step,
    )


def linspace(
        start: na.StartT,
        stop: na.StopT,
        axis: AxisT,
        num: NumT = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
) -> na.StartT | na.StopT | AxisT | NumT:
    """
    Create an array of evenly-spaced numbers between :attr:`start` and :attr:`stop`

    Parameters
    ----------
    start
        The starting value of the sequence
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`
    axis
        The name of the new axis corresponding to the sequence
    num
        Number of values in the sequence
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    dtype
        :class:`numpy.dtype` of the result

    See Also
    --------
    :func:`numpy.linspace` :  Corresponding numpy function.

    :class:`named_arrays.ScalarLinearSpace` : Corresponding implicit scalar array.

    :class:`named_arrays.UncertainScalarLinearSpace`: Corresponding implicit uncertain scalar array.
    """
    return np.linspace(
        start=na.as_named_array(start),
        stop=stop,
        axis=axis,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
    )


def logspace(
        start: na.StartT,
        stop: na.StopT,
        axis: AxisT,
        num: NumT = 50,
        endpoint: bool = True,
        base: BaseT = 10,
        dtype: None | type | np.dtype = None,
) -> na.StartT | na.StopT | AxisT | NumT:
    """
    Create an array of evenly-spaced numbers on a log scale between :attr:`start` and :attr:`stop`

    Parameters
    ----------
    start
        The starting value of the sequence
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`
    axis
        The name of the new axis corresponding to the sequence
    num
        Number of values in the sequence
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    base
        The base of the logarithm
    dtype
        :class:`numpy.dtype` of the result

    See Also
    --------
    :func:`numpy.logspace` :  Corresponding numpy function.

    :class:`named_arrays.ScalarLogarithmicSpace` : Corresponding implicit scalar array.

    :class:`named_arrays.UncertainScalarLogarithmicSpace`: Corresponding implicit uncertain scalar array.
    """
    return np.logspace(
        start=na.as_named_array(start),
        stop=stop,
        axis=axis,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
    )


def geomspace(
        start: na.StartT,
        stop: na.StopT,
        axis: AxisT,
        num: NumT = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
) -> na.StartT | na.StopT | AxisT | NumT:
    """
    Create an array of a geometric progression of numbers between :attr:`start` and :attr:`stop`

    Parameters
    ----------
    start
        The starting value of the sequence
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`
    axis
        The name of the new axis corresponding to the sequence
    num
        Number of values in the sequence
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    dtype
        :class:`numpy.dtype` of the result

    See Also
    --------
    :func:`numpy.geomspace` :  Corresponding numpy function.

    :class:`named_arrays.ScalarGeometricSpace` : Corresponding implicit scalar array.

    :class:`named_arrays.UncertainScalarGeometricSpace`: Corresponding implicit uncertain scalar array.
    """
    return np.geomspace(
        start=na.as_named_array(start),
        stop=stop,
        axis=axis,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
    )


def ndim(a: na.AbstractArray) -> int:
    return np.ndim(a)


def shape(a: na.ArrayLike) -> dict[str, int]:
    if not isinstance(a, na.AbstractArray):
        a = na.ScalarArray(a)
    return np.shape(a)


def unit(
        a: Any,
        squeeze: bool = True,
) -> None | u.UnitBase | na.AbstractArray:
    """
    Isolate the physical units associated with the given object.

    If the array has no physical units, this function returns :obj:`None`.

    Parameters
    ----------
    a
        object to isolate the units of
    squeeze
        If the result is an instance of :class:`named_arrays.AbstractVectorArray`,
        and all the components are the same, simplify the result into a single
        :class:`astropy.units.Unit` instance.

    See Also
    --------
    :func:`unit_normalized` : version of this function that returns :obj:`astropy.units.dimensionless_unscaled`
        instead of :obj:`None` if there is no unit associated with the given object
    """
    if isinstance(a, u.UnitBase):
        return a
    elif isinstance(a, u.Quantity):
        return a.unit
    elif isinstance(a, na.AbstractArray):
        return na._named_array_function(
            func=unit,
            a=a,
            squeeze=squeeze,
        )
    else:
        return None


def unit_normalized(
        a: Any,
        squeeze: bool = True,
) -> u.UnitBase | na.AbstractArray:
    """
    Isolate the physical units associated with a given object, normalizing to dimensionless units
    if the object does not have associated units.


    Parameters
    ----------
    a
        object to isolate the units of
    squeeze
        If the result is an instance of :class:`named_arrays.AbstractVectorArray`,
        and all the components are the same, simplify the result into a single
        :class:`astropy.units.Unit` instance.

    See Also
    --------
    :func:`unit` : version of this function that returns :obj:`None` instead of
        :obj:`astropy.units.dimensionless_unscaled` if there is no unit associated with the given object.

    """
    if isinstance(a, u.UnitBase):
        return a
    elif isinstance(a, u.Quantity):
        return a.unit
    elif isinstance(a, na.AbstractArray):
        return na._named_array_function(
            func=unit_normalized,
            a=a,
            squeeze=squeeze,
        )
    else:
        return u.dimensionless_unscaled


@overload
def broadcast_to(array: float | complex | np.ndarray | u.Quantity, shape: dict[str, int]) -> na.ScalarArray:
    ...


@overload
def broadcast_to(array: na.AbstractScalarArray, shape: dict[str, int]) -> na.ScalarArray:
    ...


def broadcast_to(array, shape):
    if not isinstance(array, na.AbstractArray):
        array = na.ScalarArray(array)
    return np.broadcast_to(array=array, shape=shape)


def stack(
        arrays: Sequence[na.ArrayLike],
        axis: str,
        out: None | na.AbstractExplicitArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: None | str = "same_kind",
) -> na.AbstractArray:
    if not any(isinstance(a, na.AbstractArray) for a in arrays):
        arrays = list(arrays)
        arrays[0] = na.ScalarArray(arrays[0])
    return np.stack(
        arrays=arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
    )


def concatenate(
        arrays: Sequence[na.ArrayLike],
        axis: str,
        out: None | na.AbstractExplicitArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: None | str = "same_kind",
) -> na.AbstractArray:
    if not any(isinstance(a, na.AbstractArray) for a in arrays):
        arrays = list(arrays)
        arrays[0] = na.ScalarArray(arrays[0])
    return np.concatenate(
        arrays=arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
    )


def add_axes(array: na.ArrayLike, axes: str | Sequence[str]):
    if not isinstance(array, na.AbstractArray):
        array = na.ScalarArray(array)
    return array.add_axes(axes)


def interp(
        x: float | u.Quantity | na.AbstractArray,
        xp:  na.AbstractArray,
        fp: na.AbstractArray,
        axis: None | str = None,
        left: None | float | u.Quantity | na.AbstractArray = None,
        right: None | float | u.Quantity | na.AbstractArray = None,
        period: None | float | u.Quantity | na.AbstractArray = None,
) -> na.AbstractArray:
    """
    Thin wrapper around :func:`numpy.interp`.

    Performs 1D interpolation on monotonically-increasing sample points along
    the specified axes.

    This function adds an ``axis`` argument to allow for interpolating
    :math:`n`-dimensional arrays.

    Parameters
    ----------
    x
         The new :math:`x` coordinates where the interpolant will be evaluated.
    xp
        The :math:`x` coordinates of the data points.
    fp
        The :math:`y` coordinates of the data points.
    axis
        The logical axis along which to interpolate.
    left
        Value to return for points less than ``xp[{axis: 0}]``.
        Default is ``fp[{axis: 0}]``
    right
        Value to return for points larger than ``xp[{axis: ~0}]``
    period
        A period for the :math:`x` coordinates.
        This parameter allows for proper interpolation of angular coordinates
    """
    return _named_array_function(
        func=interp,
        x=x,
        xp=xp,
        fp=fp,
        axis=axis,
        left=left,
        right=right,
        period=period,
    )


def jacobian(
        function: Callable[[InputT], OutputT],
        x: InputT,
        dx: None | InputT = None,
) -> na.AbstractMatrixArray:
    """
    Compute the Jacobian of the given function using the first-order finite difference method.

    Parameters
    ----------
    function
        The function to compute the Jacobian of
    x
        The point to evaluate the function
    step_size
        The distance that ``x`` will be perturbed by to compute the slope
    """

    if dx is None:
        dx = 1e-10
        unit_x = na.unit(x)
        if unit_x is not None:
            dx = dx * unit_x
    dx = na.asanyarray(dx, like=x)

    f_x = function(x)

    return na._named_array_function(
        func=jacobian,
        function=function,
        x=x,
        dx=dx,
        like=f_x,
    )
