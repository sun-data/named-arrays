import abc
import dataclasses
import math
from typing_extensions import Self
import named_arrays as na

__all__ = [
    "Indexable",
]


@dataclasses.dataclass(eq=False)
class Indexable(
    abc.ABC,
):
    """
    A mixin for immutable composite objects whose fields are named arrays.

    This class provides a :attr:`shape` and a named ``__getitem__`` by
    delegating to :func:`named_arrays.shape` and :func:`named_arrays.getitem`,
    both of which recurse into the :mod:`dataclasses` fields of the object.
    A subclass is therefore expected to be an (immutable) :mod:`dataclasses`
    instance whose fields are named arrays, or nested structures thereof.

    The :attr:`ndim`, :attr:`size`, and :attr:`axes` properties are derived
    from :attr:`shape`, so a subclass usually needs to define only its fields.

    .. warning::

        Do not combine this mixin with :class:`named_arrays.AbstractArray` or
        any other object recognized by :func:`named_arrays.named_array_like`.
        For such objects :func:`named_arrays.shape` and
        :func:`named_arrays.getitem` index the object directly, which would
        recurse back into the properties defined here and never terminate.
    """

    @property
    def shape(self: Self) -> dict[str, int]:
        """
        The broadcasted shape of every array-like field of this object.
        """
        return na.shape(self)

    @property
    def ndim(self: Self) -> int:
        """
        The number of dimensions of this object, ``len(self.shape)``.
        """
        return len(self.shape)

    @property
    def size(self: Self) -> int:
        """
        The total number of elements in this object, the product of the
        values of :attr:`shape`.
        """
        return math.prod(self.shape.values())

    @property
    def axes(self: Self) -> tuple[str, ...]:
        """
        The names of the axes of this object, ``tuple(self.shape)``.
        """
        return tuple(self.shape)

    def __getitem__(
        self: Self,
        item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ) -> Self:
        """
        Index every array-like field of this object by ``item``.

        Parameters
        ----------
        item
            The named index to apply to each array-like field, for example a
            :class:`dict` mapping axis names to index arrays.
        """
        return na.getitem(self, item)
