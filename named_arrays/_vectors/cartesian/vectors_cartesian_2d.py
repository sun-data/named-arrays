from __future__ import annotations
from typing import TypeVar, Type, Generic, Sequence
import math
from typing import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractCartesian2dVectorArray",
    "Cartesian2dVectorArray",
    "AbstractImplicitCartesian2dVectorArray",
    "PolarVectorArray",
    "AbstractCartesian2dVectorRandomSample",
    "Cartesian2dVectorUniformRandomSample",
    "Cartesian2dVectorNormalRandomSample",
    "AbstractParameterizedCartesian2dVectorArray",
    "Cartesian2dVectorArrayRange",
    "AbstractCartesian2dVectorSpace",
    "Cartesian2dVectorLinearSpace",
    "Cartesian2dVectorStratifiedRandomSpace",
    "Cartesian2dVectorLogarithmicSpace",
    "Cartesian2dVectorGeometricSpace",
]

XT = TypeVar('XT', bound=na.ArrayLike, covariant=True)
YT = TypeVar('YT', bound=na.ArrayLike, covariant=True)
RadiusT = TypeVar('RadiusT', bound=na.ArrayLike, covariant=True)
AzimuthT = TypeVar('AzimuthT', bound=na.ArrayLike, covariant=True)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dVectorArray(
    na.AbstractCartesianVectorArray,
):
    """
    An interface describing an array of 2D Cartesian vectors.
    """

    @property
    @abc.abstractmethod
    def x(self: Self) -> na.ArrayLike:
        """The :math:`x` component of this vector."""

    @property
    @abc.abstractmethod
    def y(self: Self) -> na.ArrayLike:
        """The :math:`y` component of this vector."""

    @property
    def type_abstract(self: Self) -> Type[AbstractCartesian2dVectorArray]:
        return AbstractCartesian2dVectorArray

    @property
    def type_explicit(self: Self) -> Type[Cartesian2dVectorArray]:
        return Cartesian2dVectorArray

    @property
    def type_matrix(self) -> Type[na.AbstractExplicitMatrixArray]:
        return na.Cartesian2dMatrixArray

    def volume_cell(self, axis: None | str | Sequence[str]) -> na.AbstractScalar:

        if axis is None:
            if self.ndim != 2:
                raise ValueError(
                    f"If {axis=}, then {self.ndim=} must be two-dimensional"
                )
            axis = self.axes

        if not set(axis).issubset(self.shape):
            raise ValueError(
                f"{axis=} should be a subset of {self.shape=}."
            )

        a1, a2 = axis

        slices = [
            {a1: slice(None, ~0), a2: slice(None, ~0)},
            {a1: slice(+1, None), a2: slice(None, ~0)},
            {a1: slice(+1, None), a2: slice(+1, None)},
            {a1: slice(None, ~0), a2: slice(+1, None)},
        ]

        array = self.broadcasted
        x = array.x
        y = array.y

        if not isinstance(x, na.AbstractScalar):    # pragma: nocover
            raise TypeError(
                f"{type(self.x)=} must be a scalar."
            )

        if not isinstance(y, na.AbstractScalar):    # pragma: nocover
            raise TypeError(
                f"{type(self.y)=} must be a scalar."
            )

        x = [x[s] for s in slices]
        y = [y[s] for s in slices]

        result = 0
        n = len(slices)
        for i in range(n):
            result = result + y[i] * (x[i - 1] - x[(i + 1) % n])

        return result / 2


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorArray(
    AbstractCartesian2dVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[XT, YT],
):
    """An array of 2D Cartesian vectors."""

    x: XT = 0
    """The :math:`x` component of this vector."""

    y: YT = 0
    """The :math:`y` component of this vector."""


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitCartesian2dVectorArray(
    AbstractCartesian2dVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):

    @property
    def x(self) -> na.ArrayLike:
        return self.explicit.x

    @property
    def y(self) -> na.ArrayLike:
        return self.explicit.y


@dataclasses.dataclass(eq=False, repr=False)
class PolarVectorArray(
    AbstractImplicitCartesian2dVectorArray,
    Generic[RadiusT, AzimuthT],
):
    r"""
    An array of 2D Cartesian vectors given by their polar coordinates.

    This is an implicit :class:`Cartesian2dVectorArray` whose components are
    :math:`x = r \cos \phi` and :math:`y = r \sin \phi`, so it can be used
    anywhere a 2D Cartesian vector is expected, while being sampled in
    :attr:`radius` and :attr:`azimuth`.
    Its purpose is to sample an annulus or a sector of one with a grid which
    follows its edges, where a rectilinear grid would waste most of its
    samples on the hole and the corners.

    Examples
    --------

    Sample an annulus with a polar grid and plot the samples.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na

        # Define a grid which is linear in radius and in azimuth.
        # The azimuth omits its endpoint so that no sample is repeated,
        # since 360 degrees is the same direction as zero.
        a = na.PolarVectorArray(
            radius=na.linspace(50, 100, axis="radius", num=6) * u.mm,
            azimuth=na.linspace(0, 360, axis="azimuth", num=24, endpoint=False) * u.deg,
        )

        # The Cartesian components are computed from the polar ones.
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        na.plt.scatter(a.x, a.y, ax=ax);

    A grid of samples is not a grid of cells.
    :meth:`volume_cell` reads its arguments as the vertices of the cells
    between them, so `n` vertices describe `n - 1` cells, and a grid built
    with ``endpoint=False`` describes one cell fewer than it appears to:
    the one which wraps past the last sample back to the first is missing.
    Keep the endpoint when the areas matter, which closes the turn.

    .. jupyter-execute::

        b = na.PolarVectorArray(
            radius=na.linspace(50, 100, axis="radius", num=6) * u.mm,
            azimuth=na.linspace(0, 360, axis="azimuth", num=25) * u.deg,
        )

        # the areas of the 5 by 24 cells sum to the area of the annulus
        b.volume_cell(("radius", "azimuth")).sum()
    """

    radius: RadiusT = 0
    """The distance of this vector from the origin."""

    azimuth: AzimuthT = 0
    """The angle of this vector from the :math:`x` axis, toward the :math:`y` axis."""

    @property
    def explicit(self) -> Cartesian2dVectorArray:
        radius = self.radius
        azimuth = self.azimuth
        return Cartesian2dVectorArray(
            x=radius * np.cos(azimuth),
            y=radius * np.sin(azimuth),
        )

    def volume_cell(self, axis: None | str | Sequence[str]) -> na.AbstractScalar:
        r"""
        The exact area of each cell of a polar grid,
        :math:`(r_2^2 - r_1^2)(\phi_2 - \phi_1) / 2`.

        The inherited implementation would take the area of the polygon through
        the four corners of the cell, which replaces each arc with the chord
        joining its ends and so always falls short.
        On an annulus sampled with six cells of azimuth that is an error of
        17%, which only reaches a tenth of a percent at ninety.

        Parameters
        ----------
        axis
            The two axes which parameterize the grid.
            The exact area is used when one of them parameterizes only
            :attr:`radius` and the other only :attr:`azimuth`, and the
            inherited polygon area otherwise, since the cells are then not
            annular sectors.
        """
        radius = na.as_named_array(self.radius)
        azimuth = na.as_named_array(self.azimuth)

        if axis is None:
            if self.ndim != 2:
                raise ValueError(
                    f"If {axis=}, then {self.ndim=} must be two-dimensional"
                )
            axis = self.axes

        if not set(axis).issubset(self.shape):
            raise ValueError(
                f"{axis=} should be a subset of {self.shape=}."
            )

        a1, a2 = axis
        shape_radius = radius.shape
        shape_azimuth = azimuth.shape

        def _separates(ax_r: str, ax_a: str) -> bool:
            return (
                ax_r in shape_radius and ax_r not in shape_azimuth
                and ax_a in shape_azimuth and ax_a not in shape_radius
            )

        if _separates(a1, a2):
            axis_radius, axis_azimuth = a1, a2
        elif _separates(a2, a1):
            axis_radius, axis_azimuth = a2, a1
        else:
            return super().volume_cell(axis)

        lower = {axis_radius: slice(None, ~0)}
        upper = {axis_radius: slice(+1, None)}
        radius_squared = np.square(radius[upper]) - np.square(radius[lower])

        lower = {axis_azimuth: slice(None, ~0)}
        upper = {axis_azimuth: slice(+1, None)}
        angle = azimuth[upper] - azimuth[lower]
        if na.unit(angle) is not None:
            angle = angle.to_value(u.rad)

        return radius_squared * angle / 2


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dVectorRandomSample(
    AbstractImplicitCartesian2dVectorArray,
    na.AbstractCartesianVectorRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorUniformRandomSample(
    AbstractCartesian2dVectorRandomSample,
    na.AbstractCartesianVectorUniformRandomSample,
):
    def volume_cell(self, axis: None | Sequence[str]) -> na.AbstractScalar:

        components = self.components

        axis = na.axis_normalized(self, axis)
        if len(axis) != len(components):
            raise ValueError(
                f"{axis=} must have exactly two elements"
            )

        shape_random = self.shape_random
        if set(axis).issubset(shape_random):
            start = na.asanyarray(self.start, like=self)
            stop = na.asanyarray(self.stop, like=self)
            span = stop - start
            size = math.prod(shape_random[ax] for ax in axis)
            result = math.prod(span.components.values()) / size
        else:
            result = super().volume_cell(axis)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorNormalRandomSample(
    AbstractCartesian2dVectorRandomSample,
    na.AbstractCartesianVectorNormalRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedCartesian2dVectorArray(
    AbstractImplicitCartesian2dVectorArray,
    na.AbstractParameterizedCartesianVectorArray,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorArrayRange(
    AbstractParameterizedCartesian2dVectorArray,
    na.AbstractCartesianVectorArrayRange,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCartesian2dVectorSpace(
    AbstractParameterizedCartesian2dVectorArray,
    na.AbstractCartesianVectorSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorLinearSpace(
    AbstractCartesian2dVectorSpace,
    na.AbstractCartesianVectorLinearSpace,
):
    def volume_cell(self, axis: None | Sequence[str]) -> na.AbstractScalar:

        components = self.components

        axis = na.axis_normalized(self, axis)
        if len(axis) != len(components):
            raise ValueError(
                f"{axis=} must have exactly two elements"
            )

        step = self.step
        if set(axis).issubset(self.axis.components.values()):
            if isinstance(step, na.AbstractVectorArray):
                # fast path for a rectilinear grid: the cell area is the product
                # of the per-component steps.
                result = math.prod(step.components.values())
            else:
                # a scalar step describes a uniform grid with the same spacing
                # along every component, so the cell volume is the step raised
                # to the number of components.
                result = step ** len(components)
        else:
            result = super().volume_cell(axis)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorStratifiedRandomSpace(
    Cartesian2dVectorLinearSpace,
    na.AbstractCartesianVectorStratifiedRandomSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorLogarithmicSpace(
    AbstractCartesian2dVectorSpace,
    na.AbstractCartesianVectorLogarithmicSpace,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class Cartesian2dVectorGeometricSpace(
    AbstractCartesian2dVectorSpace,
    na.AbstractCartesianVectorGeometricSpace,
):
    pass
