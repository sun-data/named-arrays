"""
The extra fields of a subclass of :class:`named_arrays.FunctionArray`.

A subclass can carry fields beyond its inputs and outputs, such as the
exposure time of each image. The fields whose values are named arrays are
treated as samples along the axes of the function. The operations which
select, rearrange, or combine elements along an axis do the same to them, an
operation which would remove or resample an axis that one of them varies along
raises an error, since only the subclass knows what should become of the
field, and a function array is checked when it is built, so that an operation
which does not handle the fields fails instead of leaving them stale. Fields
which are not named arrays, such as strings or nested models, are carried over
unchanged.
"""

from typing import Any, Callable, Sequence, cast
import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "names",
    "apply",
    "check",
    "check_axes",
    "stack_like",
    "setitem",
]


@functools.cache
def _candidates(cls: type) -> tuple[str, ...]:
    """
    The names of the fields of `cls`, other than its inputs and outputs, which
    are set by its constructor.
    """
    if not dataclasses.is_dataclass(cls):
        return ()
    return tuple(
        field.name
        for field in dataclasses.fields(cls)
        if field.init and field.name not in ("inputs", "outputs")
    )


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
    candidates = _candidates(type(a))
    if not candidates:
        return ()
    return tuple(
        name for name in candidates
        if isinstance(getattr(a, name), na.AbstractArray)
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


def check(a: Any) -> None:
    """
    Raise an error if a named-array field of `a` does not have the same number
    of elements as the outputs along an axis which they share.

    The outputs may have a single element along an axis, as they do after a
    reduction which keeps the reduced axes, since the inputs and the fields
    then keep their elements along it.

    Parameters
    ----------
    a
        A function array which has just been built.
    """
    fields = names(a)
    if not fields:
        return
    shape_outputs = na.shape(a.outputs)
    for name in fields:
        shape = getattr(a, name).shape
        for axis in shape:
            num = shape_outputs.get(axis)
            if num is not None and num != 1 and num != shape[axis]:
                raise ValueError(
                    f"`{name}` of this {type(a).__name__} has {shape[axis]} "
                    f"elements along {axis!r}, but the outputs have {num}. "
                    f"An operation changed the outputs along {axis!r} without "
                    f"changing `{name}` to match."
                )


def check_axes(
    a: Any,
    axis: Sequence[str],
    operation: str,
    hint: str = "",
) -> None:
    """
    Raise an error if an operation would remove or resample an axis along
    which a named-array field of `a` varies.

    Parameters
    ----------
    a
        The function array being operated on.
    axis
        The axes which the operation removes or resamples.
    operation
        A description of what the operation does to `axis`, for the error
        message, for example ``"np.sum with keepdims=False removes"``.
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
                f"which {operation}, so it would no longer match the outputs. "
                f"Replace it first with a version which does not vary along "
                f"{arg}, reduced the way it should be (a sum or mean of an "
                f"exposure time, `any` or `all` of a mask), for example "
                f"`dataclasses.replace(a, {name}=a.{name}.sum({arg}))`."
                f"{hint}"
            )


def _value(a: Any, name: str, operation: str) -> "na.AbstractArray":
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
        f"`{name}` must be a named array or a scalar to be used by "
        f"{operation}, got {type(value).__name__}"
    )


def _equal(a: Any, b: Any) -> bool:
    """
    Whether two values of a field are the same, without raising if they
    cannot be compared, for example because their units differ.
    """
    if a is b:
        return True
    try:
        # named arrays broadcast against each other by the names of their
        # axes, so a value which is constant along an axis equals one which
        # does not have it
        return bool(np.all(a == b))
    except (TypeError, ValueError, u.UnitsError):
        return False


def _simple(value: Any) -> bool:
    """
    Whether a value which is not a named array can be compared with
    ``==``: a number, a string, a scalar quantity, or a list or tuple of them.
    """
    if value is None or isinstance(value, (bool, int, float, complex, str, np.number)):
        return True
    if isinstance(value, u.Quantity):
        return np.ndim(value) == 0
    if isinstance(value, (list, tuple)):
        return all(_simple(v) for v in value)
    return False


def stack_like(
    func: Callable,
    arrays: Sequence[Any],
    template: Any,
    axis: str,
    lengths: None | Sequence[int],
) -> dict[str, Any]:
    """
    Combine the fields of `arrays` the way `func` combines the arrays
    themselves.

    The named-array fields are stacked or concatenated. The other fields must
    be the same in every array which has them, if they are simple values which
    can be compared, and are taken from `template` otherwise.

    Parameters
    ----------
    func
        Either :func:`numpy.stack` or :func:`numpy.concatenate`.
    arrays
        The function arrays being stacked or concatenated.
    template
        The array whose type, and whose fields which cannot be compared, the
        result takes.
    axis
        The axis along which the arrays are stacked or concatenated.
    lengths
        The number of elements of each array along `axis` when the arrays are
        concatenated, used to broadcast a field which does not vary along it.

    Returns
    -------
        The value of every field of `template` other than its inputs and
        outputs.
    """
    operation = f"np.{func.__name__}"
    array_like = set(n for a in arrays for n in names(a))
    result = dict()
    for name in _candidates(type(template)):

        if name not in array_like:
            values = [getattr(a, name) for a in arrays if hasattr(a, name)]
            if all(_simple(v) for v in values):
                if not all(_equal(v, values[0]) for v in values):
                    raise ValueError(
                        f"`{name}` differs between the arrays combined with "
                        f"{operation}, got {values}"
                    )
            result[name] = getattr(template, name)
            continue

        values = [_value(a, name, operation) for a in arrays if hasattr(a, name)]

        if all(axis not in v.shape for v in values) and all(_equal(v, values[0]) for v in values):
            # the same in every array which has it, and constant along the
            # axis, so it still is
            result[name] = getattr(template, name)
            continue

        if len(values) < len(arrays):
            raise ValueError(
                f"`{name}` differs between the arrays combined with "
                f"{operation}, but some of them do not have it, got "
                f"{[type(a).__name__ for a in arrays]}"
            )

        if lengths is not None:
            shape = na.broadcast_shapes(*[
                {ax: n for ax, n in v.shape.items() if ax != axis} for v in values
            ])
            values = [
                na.broadcast_to(v, shape | {axis: length})
                for v, length in zip(values, lengths)
            ]
        else:
            shape = na.broadcast_shapes(*[v.shape for v in values])
            values = [na.broadcast_to(v, shape) for v in values]

        result[name] = func(values, axis=axis)
    return result


def _writer(
    field: "na.AbstractArray",
    index: Any,
    value: "na.AbstractArray",
) -> Callable[[], None]:
    """A function which writes `value` into `field` at `index` when called."""
    def write() -> None:
        cast("na.AbstractExplicitArray", field)[index] = value
    return write


def setitem(
    a: Any,
    item: "dict[str, Any] | na.AbstractArray",
    value: Any,
) -> list[Callable[[], None]]:
    """
    Check that the fields of `value` can be written into those of `a` at
    `item`, and return the writes, so that they can be made after the outputs
    are written, without leaving `a` half written if one of them fails.

    A field of `a` which does not vary along an axis of `item` can only take a
    value which is the same as the one it has.

    Parameters
    ----------
    a
        The function array being written into.
    item
        A dictionary of the indices along each axis, or a boolean mask, as
        given to the outputs of `a`.
    value
        The function array being written.
    """
    if not isinstance(value, na.AbstractFunctionArray):
        return []

    operation = "assignment"
    writes = []
    for name in dict.fromkeys(names(a) + names(value)):
        if not hasattr(value, name):
            continue
        field = _value(a, name, operation)
        field_value = _value(value, name, operation)

        if isinstance(item, dict):
            axes = tuple(item)
            index = {ax: item[ax] for ax in item if ax in field.shape}
            current = field[index]
        else:
            axes = tuple(item.shape)
            index = item
            current = na.broadcast_to(field, na.broadcast_shapes(field.shape, item.shape))[item]

        extra = set(field_value.shape) - set(current.shape)
        if extra:
            raise ValueError(
                f"`{name}` of the value varies along {sorted(extra)}, which "
                f"`{name}` of this {type(a).__name__} does not have"
            )

        if set(axes).issubset(field.shape) and isinstance(getattr(a, name), na.AbstractArray):
            writes.append(_writer(field, index, field_value))
        elif not _equal(current, field_value):
            missing = sorted(set(axes) - set(field.shape))
            raise ValueError(
                f"`{name}` does not vary along {missing}, so it cannot hold a "
                f"different value for only the elements being set. Broadcast "
                f"it along {missing} first."
            )
    return writes
