import dataclasses
import named_arrays as na

__all__ = []


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectralMatrixArray(
    na.AbstractMatrixArray,
    na.AbstractSpectralVectorArray,
):
    pass

@dataclasses.dataclass(eq=False, repr=False)
class AbstractExplicitSpectralMatrixArray(
    AbstractSpectralMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitSpectralMatrixArray(
    AbstractSpectralMatrixArray,
    na.AbstractImplicitMatrixArray,
):
    pass