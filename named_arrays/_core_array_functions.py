from typing import Callable, Sequence

import numpy as np
import named_arrays as na

__all__ = [
    'HANDLED_FUNCTIONS',
]

HANDLED_FUNCTIONS = dict()


def implements(numpy_function: Callable):
    """Register an ``__array_function__`` implementation for :class:`named_array.AbstractArray` objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.shape)
def shape(
        a: na.AbstractScalarArray,
) -> dict[str, int]:
    """
    Compute the shape of the given array.

    In :mod:`numpy`, the shape of an array is a :class:`tuple` of integers.
    For this package, each axis is characterized by a name instead of
    its position, so the shape is a :class:`dict` where the keys are
    the axis names and the values are number of elements along each axis.

    Parameters
    ----------
    a
        The array to compute the shape of.
    """
    return a.shape


@implements(np.broadcast_to)
def broadcast_to(
    array: na.AbstractArray,
    shape: dict[str, int],
) -> na.AbstractExplicitArray:
    """
    Broadcast the given array to the requested shape.

    Parameters
    ----------
    array
        The array to broadcast.
    shape
        The requested shape of the result.

    See Also
    --------
    :func:`numpy.broadcast_to`: Equivalent :mod:`numpy` function.
    """
    return na.broadcast_to(
        array=array,
        shape=shape,
    )


def _axes_normalized(
    shape: dict[str, int],
    axis: None | str | Sequence[str],
) -> tuple[str, ...]:
    """
    Convert the `axis` argument of a structural function into a tuple of names.

    Unlike :func:`named_arrays.axis_normalized`, this helper works on a shape
    instead of an array, and drops the axes which are not present in that shape.

    Parameters
    ----------
    shape
        The shape of the array being operated on.
    axis
        The axes to normalize.
        If :obj:`None`, every axis of `shape` is used.
    """
    if axis is None:
        result = tuple(shape)
    elif isinstance(axis, str):
        result = (axis,)
    else:
        result = tuple(axis)

    return tuple(ax for ax in result if ax in shape)


@implements(np.expand_dims)
def expand_dims(
    a: na.AbstractArray,
    axis: str | Sequence[str],
) -> na.AbstractExplicitArray:
    """
    Add new axes of length one to the given array.

    Parameters
    ----------
    a
        The array to add axes to.
    axis
        Either a single axis name, or a sequence of axis names, to add.

    Raises
    ------
    ValueError
        If `axis` repeats a name, or names an axis which `a` already has.

    See Also
    --------
    :func:`numpy.expand_dims`: Equivalent :mod:`numpy` function.
    :func:`named_arrays.add_axes`: The function used to implement this one.
    """
    shape = na.shape(a)

    axes = (axis,) if isinstance(axis, str) else tuple(axis)

    if len(set(axes)) != len(axes):
        raise ValueError(f"the requested axes, {axes}, must not contain duplicates")

    axes_existing = tuple(ax for ax in axes if ax in shape)
    if axes_existing:
        raise ValueError(
            f"the requested axes, {axes_existing}, are already axes of `a`, {shape}"
        )

    return na.add_axes(a, axes)


@implements(np.squeeze)
def squeeze(
    a: na.AbstractArray,
    axis: None | str | Sequence[str] = None,
) -> na.AbstractExplicitArray:
    """
    Remove axes of length one from the given array.

    Parameters
    ----------
    a
        The array to remove axes from.
    axis
        The axes to consider removing.
        If :obj:`None` (the default), every axis of length one is removed.
        Axes not present in `a` are ignored.

    Raises
    ------
    ValueError
        If `axis` names an axis of `a` whose length is not one.

    See Also
    --------
    :func:`numpy.squeeze`: Equivalent :mod:`numpy` function.
    :func:`named_arrays.debroadcast`: Remove the axes along which an array is
        constant, whatever their length.

    Notes
    -----
    An axis which `a` does not have is ignored rather than an error, since an
    array broadcasts along such an axis as though it had length one, and
    removing an axis of length one is what this function does.
    """
    a = a.explicit
    shape = a.shape

    if axis is None:
        axes = tuple(ax for ax in shape if shape[ax] == 1)
    else:
        axes = _axes_normalized(shape, axis)
        axes_wrong = {ax: shape[ax] for ax in axes if shape[ax] != 1}
        if axes_wrong:
            raise ValueError(
                f"the requested axes, {axes_wrong}, do not have a length of one"
            )

    result = a[{ax: 0 for ax in axes}]

    axes_remaining = tuple(ax for ax in axes if ax in result.shape)
    if axes_remaining:
        raise ValueError(
            f"the requested axes, {axes_remaining}, could not be removed from an "
            f"instance of {type(a)}, since indexing them does not reduce the "
            f"number of dimensions of the array"
        )

    return result


@implements(np.flip)
def flip(
    m: na.AbstractArray,
    axis: None | str | Sequence[str] = None,
) -> na.AbstractExplicitArray:
    """
    Reverse the order of the elements of the given array along the given axes.

    Parameters
    ----------
    m
        The array to reverse.
    axis
        The axes to reverse along.
        If :obj:`None` (the default), every axis of `m` is reversed.
        Axes not present in `m` are ignored.

    See Also
    --------
    :func:`numpy.flip`: Equivalent :mod:`numpy` function.

    Notes
    -----
    An axis which `m` does not have is ignored rather than an error, since an
    array broadcasts along such an axis as though it had length one, and
    reversing an axis of length one leaves the array unchanged.
    """
    m = m.explicit

    axes = _axes_normalized(m.shape, axis)

    return m[{ax: slice(None, None, -1) for ax in axes}]


@implements(np.roll)
def roll(
    a: na.AbstractArray,
    shift: int | Sequence[int],
    axis: None | str | Sequence[str] = None,
) -> na.AbstractExplicitArray:
    """
    Shift the elements of the given array along the given axes, with the
    elements shifted off the end reappearing at the start.

    Parameters
    ----------
    a
        The array to shift.
    shift
        The number of places to shift by.
        If `axis` names more than one axis, this may be one number for each of
        them, or one number used for all of them.
    axis
        The axes to shift along.
        Axes not present in `a` are ignored.

    Raises
    ------
    ValueError
        If `axis` is :obj:`None`, or if `shift` and `axis` are sequences of
        different lengths.

    See Also
    --------
    :func:`numpy.roll`: Equivalent :mod:`numpy` function.

    Notes
    -----
    Unlike :func:`numpy.roll`, `axis` is required. The :mod:`numpy` version
    shifts the flattened array when given no axis, and this package has no
    positional order to flatten along.

    An axis which `a` does not have is ignored, since an array broadcasts along
    such an axis as though it had length one, and shifting an axis of length one
    leaves the array unchanged.

    The shift is expressed as indexing `a` by an array of indices, so an array
    which does not support that kind of indexing along `axis` cannot be shifted
    along it either. The vertex axis of a
    :class:`named_arrays.FunctionArray` is the case in point, since it has one
    more input than it has outputs and no single index applies to both.
    """
    if axis is None:
        raise ValueError(
            "`axis` is required, since there is no positional order to flatten "
            "along. Name the axes to shift along, or use `combine_axes` first."
        )

    axes = (axis,) if isinstance(axis, str) else tuple(axis)
    shifts = (shift,) * len(axes) if isinstance(shift, int) else tuple(shift)

    if len(shifts) != len(axes):
        raise ValueError(
            f"`shift` and `axis` must have the same length, "
            f"got {len(shifts)} and {len(axes)}"
        )

    a = a.explicit
    shape = a.shape

    index = dict()
    for ax, sh in zip(axes, shifts):
        if ax not in shape:
            continue
        num = shape[ax]
        index[ax] = (na.arange(0, num, axis=ax) - sh) % num

    return a[index]
