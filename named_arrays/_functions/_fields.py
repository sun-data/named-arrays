"""
The extra fields of a subclass of :class:`named_arrays.FunctionArray`.

A subclass can carry fields beyond its inputs and outputs, such as the
exposure time of each image. The fields whose values are named arrays are
treated as samples along the axes of the function. The operations which
select, rearrange, or combine elements along an axis do the same to them, and
a reduction which would remove an axis that one of them varies along raises an
error, since only the subclass knows whether the field should be summed,
averaged, or dropped. Fields which are not named arrays, such as strings or
nested models, are carried over unchanged.
"""

from typing import Any, Callable, Sequence, cast
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "names",
    "apply",
    "stack_like",
    "check_reduction",
]


def names(a: Any) -> tuple[str, ...]:
    """
    The names of the fields of `a`, other than its inputs and outputs, whose
    values are named arrays.

    Parameters
    ----------
    a
        A function array, usually an instance of a subclass of
        :class:`named_arrays.FunctionArray`.
    """
    if not dataclasses.is_dataclass(a) or isinstance(a, type):
        return ()
    return tuple(
        field.name
        for field in dataclasses.fields(a)
        if field.init
        and field.name not in ("inputs", "outputs")
        and isinstance(getattr(a, field.name), na.AbstractArray)
    )


def apply(
    a: Any,
    func: Callable[["na.AbstractArray"], "na.AbstractArray"],
    axes: None | Sequence[str] = None,
) -> dict[str, "na.AbstractArray"]:
    """
    Apply `func` to each named-array field of `a`.

    The result can be passed to :meth:`named_arrays.AbstractArray.replace`.

    Parameters
    ----------
    a
        A function array.
    func
        The operation to apply to each field.
    axes
        If given, only the fields which vary along at least one of these axes
        are changed, since the others are unaffected by an operation along
        them.
    """
    result = dict()
    for name in names(a):
        value = getattr(a, name)
        if axes is not None and set(axes).isdisjoint(value.shape):
            continue
        result[name] = func(value)
    return result


def stack_like(
    func: Callable,
    arrays: Sequence[Any],
    axis: str,
    shapes: Sequence[dict[str, int]],
) -> dict[str, "na.AbstractArray"]:
    """
    Stack or concatenate the named-array fields of `arrays`, the way
    `func` stacks or concatenates the arrays themselves.

    Parameters
    ----------
    func
        Either :func:`numpy.stack` or :func:`numpy.concatenate`.
    arrays
        The function arrays being stacked or concatenated.
    axis
        The axis along which the arrays are stacked or concatenated.
    shapes
        The shape of each array, used to broadcast a field which does not vary
        along `axis` when the arrays are concatenated.
    """
    result = dict()
    for name in dict.fromkeys(n for a in arrays for n in names(a)):
        values = [_value(a, name, func) for a in arrays if hasattr(a, name)]

        if all(axis not in v.shape for v in values) and _all_equal(values):
            # the same in every array which has it, and constant along the
            # axis, so it still is
            continue

        if len(values) < len(arrays):
            raise ValueError(
                f"`{name}` differs between the arrays combined with "
                f"{func.__name__}, but some of them do not have it, got "
                f"{[type(a).__name__ for a in arrays]}"
            )

        if func is np.concatenate:
            shape = na.broadcast_shapes(*[
                {ax: n for ax, n in v.shape.items() if ax != axis} for v in values
            ])
            values = [
                na.broadcast_to(v, shape | {axis: s[axis]})
                for v, s in zip(values, shapes)
            ]
        else:
            shape = na.broadcast_shapes(*[v.shape for v in values])
            values = [na.broadcast_to(v, shape) for v in values]

        result[name] = func(values, axis=axis)
    return result


def _value(a: Any, name: str, func: Callable) -> "na.AbstractArray":
    """
    The field `name` of `a` as a named array, so that it can be combined with
    the fields of other arrays.

    A scalar, such as a default exposure time of ``0 * u.s``, is a value
    which is the same for every element.
    """
    value = getattr(a, name)
    if isinstance(value, na.AbstractArray):
        return value
    if isinstance(value, (int, float, complex, np.number, u.Quantity)) and np.ndim(value) == 0:
        return cast("na.AbstractArray", na.as_named_array(value))
    raise ValueError(
        f"`{name}` must be a named array or a scalar in every array combined "
        f"with {func.__name__}, got {type(value).__name__}"
    )


def _all_equal(values: Sequence["na.AbstractArray"]) -> bool:
    """Whether every one of `values` is the same as the first."""
    first = values[0]
    for v in values[1:]:
        if v is first:
            continue
        if v.shape != first.shape or not bool(np.all(v == first)):
            return False
    return True


def check_reduction(
    a: Any,
    axis: Sequence[str],
    operation: str,
    hint: str = "",
) -> None:
    """
    Raise an error if a reduction would remove an axis along which a
    named-array field of `a` varies.

    Parameters
    ----------
    a
        The function array being reduced.
    axis
        The axes which the reduction removes.
    operation
        The name of the reduction, for the error message.
    hint
        Another way to avoid the error, appended to the message.
    """
    for name in names(a):
        value = getattr(a, name)
        axes = tuple(ax for ax in axis if ax in value.shape)
        if axes:
            arg = repr(axes[0]) if len(axes) == 1 else repr(axes)
            raise ValueError(
                f"`{name}` of this {type(a).__name__} varies along {axes}, "
                f"which {operation} removes, so it would no longer match the "
                f"outputs. Reduce it along those axes first, choosing whether "
                f"to sum or average it, for example "
                f"`dataclasses.replace(a, {name}=a.{name}.sum({arg}))`."
                f"{hint}"
            )
