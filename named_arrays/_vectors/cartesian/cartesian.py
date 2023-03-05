from __future__ import annotations
from typing import TypeVar, Generic, Type
from typing_extensions import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    'AbstractCartesianVectorArray',
    'AbstractExplicitCartesianVectorArray',
    'AbstractImplicitCartesianVectorArray',
    'AbstractCartesianVectorRandomSample',
    'AbstractCartesianVectorUniformRandomSample',
    'AbstractCartesianVectorNormalRandomSample',
    'AbstractParameterizedCartesianVectorArray',
    'AbstractCartesianVectorArrayRange',
    'AbstractCartesianVectorSpace',
    'AbstractCartesianVectorLinearSpace',
    'AbstractCartesianVectorStratifiedRandomSpace',
    'AbstractCartesianVectorLogarithmicSpace',
    'AbstractCartesianVectorGeometricSpace',
]


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorArray(
    na.AbstractVectorArray
):
    @property
    @abc.abstractmethod
    def type_array(self: Self) -> Type[AbstractExplicitCartesianVectorArray]:
        pass

    @property
    def length(self: Self) -> na.AbstractScalar:
        result = 0
        components = self.components
        for c in components:
            if components[c] is not None:
                result = result + np.square(components[c])
        result = np.sqrt(result)
        return result

    def __mul__(self: Self, other: na.ArrayLike | u.Unit) -> AbstractExplicitCartesianVectorArray:
        if isinstance(other, u.UnitBase):
            result = self.type_array()
            for c in result.components:
                result.components[c] = result.components[c] * other
            return result
        else:
            return super().__mul__(other)

    def __lshift__(self: Self, other: na.ArrayLike | u.UnitBase) -> AbstractExplicitCartesianVectorArray:
        if isinstance(other, u.UnitBase):
            result = self.type_array()
            for c in result.components:
                result.components[c] = result.components[c] << other
            return result
        else:
            return super().__lshift__(other)

    def __truediv__(self: Self, other: na.ArrayLike | u.UnitBase) -> AbstractExplicitCartesianVectorArray:
        if isinstance(other, u.UnitBase):
            result = self.type_array()
            for c in result.components:
                result.components[c] = result.components[c] / other
            return result
        else:
            return super().__truediv__(other)

    def __array_ufunc__(
            self: Self,
            function: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | AbstractExplicitCartesianVectorArray | tuple[AbstractExplicitCartesianVectorArray, ...]:

        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

        components = self.components

        components_inputs = []
        for inp in inputs:
            if isinstance(inp, na.AbstractArray):
                if inp.type_array_abstract == self.type_array_abstract:
                    components_inp = inp.components
                elif isinstance(inp, na.AbstractScalar):
                    components_inp = {c: inp for c in components}
                else:
                    return NotImplemented
            else:
                components_inp = {c: inp for c in components}
            components_inputs.append(components_inp)

        if "out" in kwargs:
            out = kwargs.pop("out")
            components_out = dict()
            for c in components:
                components_out[c] = tuple(o.components[c] if o is not None else o for o in out)
                components_out[c] = tuple(o if isinstance(np.ndarray, na.AbstractArray) else None for o in out)
        else:
            out = (None, ) * function.nout
            components_out = {c: (None, ) * function.nout for c in components}

        if "where" in kwargs:
            where = kwargs.pop("where")
            if isinstance(where, na.AbstractArray):
                if where.type_array_abstract == self.type_array_abstract:
                    components_where = where.components
                elif isinstance(where, na.ScalarArray):
                    components_where = {c: where for c in components}
                else:
                    return where.__array_ufunc__(function, method, *inputs, **kwargs)
            else:
                components_where = {c: where for c in components}
        else:
            components_where = {c: True for c in components}

        components_result = tuple(dict() for _ in range(function.nout))
        for c in components:
            component_result = getattr(function, method)(
                *[inp[c] for inp in components_inputs],
                out=components_out[c],
                where=components_where[c],
                **kwargs,
            )
            if function.nout == 1:
                component_result = (component_result, )
            for i in range(function.nout):
                components_result[i][c] = component_result[i]
        result = list(self.type_array.from_components(components_result[i]) for i in range(function.nout))

        for i in range(function.nout):
            if out[i] is not None:
                out[i].components = result[i].components
                result[i] = out[i]

        if function.nout == 1:
            result = result[0]
        else:
            result = tuple(result)
        return result


@dataclasses.dataclass(eq=False)
class AbstractExplicitCartesianVectorArray(
    AbstractCartesianVectorArray,
    na.AbstractExplicitVectorArray,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractImplicitCartesianVectorArray(
    AbstractCartesianVectorArray,
    na.AbstractImplicitVectorArray,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorRandomSample(
    AbstractImplicitCartesianVectorArray,
    na.AbstractVectorRandomSample,
):
    pass


StartT = TypeVar('StartT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractCartesianVectorArray)
StopT = TypeVar('StopT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractCartesianVectorArray)


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorUniformRandomSample(
    AbstractCartesianVectorRandomSample,
    na.AbstractVectorUniformRandomSample,
    Generic[StartT, StopT],
):
    start: StartT = dataclasses.MISSING
    stop: StopT = dataclasses.MISSING
    shape_random: dict[str, int] = None
    seed: None | int = None

    @property
    def array(self) -> AbstractExplicitCartesianVectorArray:
        start = self.start
        if not isinstance(start, na.AbstractVectorArray):
            start = self.type_array.from_scalar(start)

        stop = self.stop
        if not isinstance(stop, na.AbstractVectorArray):
            stop = self.type_array.from_scalar(stop)

        seed = self.seed

        result = self.type_array()
        components_start = start.components
        components_stop = stop.components

        for c in result.components:
            result.components[c] = na.ScalarUniformRandomSample(
                start=components_start[c],
                stop=components_stop[c],
                shape_random=self.shape_random,
                seed=seed,
            ).array
            seed += 1

        return result


CenterT = TypeVar('CenterT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractCartesianVectorArray)
WidthT = TypeVar('WidthT', bound=float | complex | u.Quantity | na.AbstractScalar | AbstractCartesianVectorArray)


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorNormalRandomSample(
    AbstractCartesianVectorRandomSample,
    na.AbstractVectorNormalRandomSample,
    Generic[CenterT, WidthT],
):
    center: StartT = dataclasses.MISSING
    width: StopT = dataclasses.MISSING
    shape_random: dict[str, int] = None
    seed: None | int = None

    @property
    def array(self) -> AbstractExplicitCartesianVectorArray:
        center = self.center
        if not isinstance(center, na.AbstractVectorArray):
            center = self.type_array.from_scalar(center)

        width = self.width
        if not isinstance(width, na.AbstractVectorArray):
            width = self.type_array.from_scalar(width)

        seed = self.seed

        result = self.type_array()
        components_center = center.components
        components_width = width.components

        for c in result.components:
            result.components[c] = na.ScalarNormalRandomSample(
                center=components_center[c],
                width=components_width[c],
                shape_random=self.shape_random,
                seed=seed,
            ).array
            seed += 1

        return result


@dataclasses.dataclass(eq=False)
class AbstractParameterizedCartesianVectorArray(
    AbstractImplicitCartesianVectorArray,
    na.AbstractParameterizedVectorArray,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorArrayRange(
    AbstractParameterizedCartesianVectorArray,
    na.AbstractVectorArrayRange,
    Generic[StartT, StopT]
):
    @property
    def array(self) -> AbstractExplicitCartesianVectorArray:
        start = self.start
        if not isinstance(start, na.AbstractVectorArray):
            start = self.type_array.from_scalar(start)

        stop = self.stop
        if not isinstance(stop, na.AbstractVectorArray):
            stop = self.type_array.from_scalar(stop)

        result = self.type_array()
        components_start = start.components
        components_stop = stop.components

        # for c in result.components:
        #     result.components[c] = na.
        #
        # return


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorSpace(
    AbstractParameterizedCartesianVectorArray,
    na.AbstractVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorLinearSpace(
    AbstractCartesianVectorSpace,
    na.AbstractVectorLinearSpace,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorStratifiedRandomSpace(
    AbstractCartesianVectorLinearSpace,
    na.AbstractVectorStratifiedRandomSpace,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorLogarithmicSpace(
    AbstractCartesianVectorSpace,
    na.AbstractVectorLogarithmicSpace,
):
    pass


@dataclasses.dataclass(eq=False)
class AbstractCartesianVectorGeometricSpace(
    AbstractCartesianVectorSpace,
    na.AbstractVectorGeometricSpace,
):
    pass