import dataclasses
import named_arrays as na

__all__ = [
    'AbstractCartesianMatrixArray',
    'AbstractExplicitCartesianMatrixArray',
    'AbstractImplicitCartesianMatrixArray',
    'AbstractCartesianMatrixRandomSample',
    'AbstractCartesianMatrixUniformRandomSample',
    'AbstractCartesianMatrixNormalRandomSample',
    'AbstractParameterizedCartesianMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianMatrixArray(
    na.AbstractMatrixArray,
    na.AbstractCartesianVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractExplicitCartesianMatrixArray(
    AbstractCartesianMatrixArray,
    na.AbstractExplicitMatrixArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitCartesianMatrixArray(
    AbstractCartesianMatrixArray,
    na.AbstractImplicitMatrixArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianMatrixRandomSample(
    AbstractImplicitCartesianMatrixArray,
    na.AbstractMatrixRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianMatrixUniformRandomSample(
    AbstractCartesianMatrixRandomSample,
    na.AbstractMatrixUniformRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesianMatrixNormalRandomSample(
    AbstractCartesianMatrixRandomSample,
    na.AbstractMatrixNormalRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedCartesianMatrixArray(
    AbstractImplicitCartesianMatrixArray,
    na.AbstractParameterizedMatrixArray,
):
    pass
