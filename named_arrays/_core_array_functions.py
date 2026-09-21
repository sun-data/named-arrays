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


def _coordinates(
    spacing: na.AbstractArray,
    axis: str,
    num: int,
) -> na.AbstractArray:
    """
    The coordinates along `axis` described by a spacing argument.

    :func:`numpy.gradient` reads a spacing either as the coordinates
    themselves, when it runs the length of the axis, or as the constant gap
    between them otherwise. This applies the same rule.
    """
    if na.shape(spacing).get(axis, None) == num:
        return spacing

    return spacing * na.arange(0, num, axis=axis)


def _gradient(
    f: na.AbstractArray,
    x: na.AbstractArray,
    axis: str,
    edge_order: int = 1,
) -> na.AbstractExplicitArray:
    """
    Differentiate `f` with respect to `x` along `axis`, one element at a time.

    The interior uses second-order central differences, weighting each
    neighbor by the gap on the opposite side so that unevenly spaced samples
    are handled, and the two ends use one-sided differences of order
    `edge_order`.

    :func:`numpy.gradient` takes the coordinates along an axis as a
    one-dimensional array, so it cannot express a coordinate which varies
    along any other axis. That is what a distorted grid gives, and what the
    distribution of an uncertain coordinate always is, since it carries the
    distribution axis. Differencing elementwise removes the restriction, at
    the cost of a few temporary arrays, and agrees with
    :func:`numpy.gradient` wherever both apply.

    Parameters
    ----------
    f
        The values to differentiate.
    x
        The coordinates to differentiate against, which must be a scalar and
        must vary along `axis`.
    axis
        The axis to differentiate along.
    edge_order
        The order of the one-sided differences at the two ends, 1 or 2.
    """
    if edge_order not in (1, 2):
        raise ValueError(f"{edge_order=} must be either 1 or 2")

    # the gaps, their squares, and their product all overflow in a narrow
    # integer type, which would corrupt the interior of the result without
    # any warning, so an integer variable is promoted first, as
    # :func:`numpy.gradient` does
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float)

    shape_x = na.shape(x)

    if axis not in shape_x:
        raise ValueError(
            f"the differentiation variable does not vary along {axis=}, so "
            f"the derivative against it is undefined there. The variable has "
            f"shape {shape_x}."
        )

    num = shape_x[axis]

    if num < edge_order + 1:
        raise ValueError(
            f"at least {edge_order + 1} points are required along {axis=} "
            f"for {edge_order=}, got {num}"
        )

    shape_f = na.shape(f)
    if shape_f.get(axis, 1) != num:
        f = na.broadcast_to(f, na.broadcast_shapes(shape_f, {axis: num}))

    def _slice(a: na.AbstractArray, s: slice) -> na.AbstractArray:
        return a[{axis: s}]

    # second-order central differences in the interior, which for uneven
    # spacing weight each neighbor by the gap on the opposite side
    gap_behind = _slice(x, slice(1, ~0)) - _slice(x, slice(None, -2))
    gap_ahead = _slice(x, slice(2, None)) - _slice(x, slice(1, ~0))
    interior = (
        np.square(gap_behind) * _slice(f, slice(2, None))
        + (np.square(gap_ahead) - np.square(gap_behind)) * _slice(f, slice(1, ~0))
        - np.square(gap_ahead) * _slice(f, slice(None, -2))
    ) / (gap_behind * gap_ahead * (gap_ahead + gap_behind))

    if edge_order == 1:
        first = (
            _slice(f, slice(1, 2)) - _slice(f, slice(0, 1))
        ) / (_slice(x, slice(1, 2)) - _slice(x, slice(0, 1)))
        last = (
            _slice(f, slice(-1, None)) - _slice(f, slice(-2, -1))
        ) / (_slice(x, slice(-1, None)) - _slice(x, slice(-2, -1)))
    else:
        d1 = _slice(x, slice(1, 2)) - _slice(x, slice(0, 1))
        d2 = _slice(x, slice(2, 3)) - _slice(x, slice(1, 2))
        first = (
            -(2 * d1 + d2) / (d1 * (d1 + d2)) * _slice(f, slice(0, 1))
            + (d1 + d2) / (d1 * d2) * _slice(f, slice(1, 2))
            - d1 / (d2 * (d1 + d2)) * _slice(f, slice(2, 3))
        )
        d1 = _slice(x, slice(-2, -1)) - _slice(x, slice(-3, -2))
        d2 = _slice(x, slice(-1, None)) - _slice(x, slice(-2, -1))
        last = (
            d2 / (d1 * (d1 + d2)) * _slice(f, slice(-3, -2))
            - (d2 + d1) / (d1 * d2) * _slice(f, slice(-2, -1))
            + (2 * d2 + d1) / (d2 * (d1 + d2)) * _slice(f, slice(-1, None))
        )

    return np.concatenate([first, interior, last], axis=axis)
