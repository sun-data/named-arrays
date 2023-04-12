from __future__ import annotations
from typing import Sequence, overload, Type, Any, Callable, TypeVar
import functools
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    '_named_array_function',
    'arange',
    'linspace',
    'logspace',
    'geomspace',
    'ndim',
    'shape',
    'broadcast_to',
    'stack',
    'concatenate',
    'add_axes',
]

AxisT = TypeVar("AxisT", bound="str | na.AbstractArray")
NumT = TypeVar("NumT", bound="int | na.AbstractArray")
BaseT = TypeVar("BaseT", bound="int | na.AbstractArray")


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

    raise TypeError("all types returned `NotImplemented`")


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
