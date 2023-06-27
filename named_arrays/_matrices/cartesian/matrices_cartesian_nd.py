import dataclasses
import named_arrays as na

__all__ = [
    "AbstractCartesianNdMatrixArray",
    "CartesianNdMatrixArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianNdMatrixArray(
    na.AbstractCartesianMatrixArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class CartesianNdMatrixArray(
    AbstractCartesianNdMatrixArray,
    na.AbstractExplicitCartesianMatrixArray,
):
    pass
